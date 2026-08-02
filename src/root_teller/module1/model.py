from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .config import SERVICE_INDEX
from .data import MODALITIES, modality_quality


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int = 48
    embedding_dim: int = 48
    graph_layers: int = 2
    dropout: float = 0.2
    fusion_lambda: float = 0.6
    fusion_mode: str = "adaptive"
    use_availability_mask: bool = True


class MaskedGRUEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.cell1 = nn.GRUCell(input_dim, hidden_dim)
        self.cell2 = nn.GRUCell(hidden_dim, hidden_dim)
        self.attention = nn.Linear(hidden_dim, 1, bias=False)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        nodes, steps, _ = values.shape
        h1 = values.new_zeros((nodes, self.cell1.hidden_size))
        h2 = values.new_zeros((nodes, self.cell2.hidden_size))
        states = []
        for step in range(steps):
            valid = mask[:, step].unsqueeze(1)
            candidate1 = self.cell1(values[:, step], h1)
            h1 = torch.where(valid, candidate1, h1)
            candidate2 = self.cell2(h1, h2)
            h2 = torch.where(valid, candidate2, h2)
            states.append(h2)
        sequence = torch.stack(states, dim=1)
        logits = self.attention(sequence).squeeze(-1)
        logits = logits.masked_fill(~mask, -1e9)
        weights = torch.softmax(logits, dim=1)
        weights = weights * mask.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        pooled = torch.sum(sequence * weights.unsqueeze(-1), dim=1)
        return self.output(pooled)


class PooledMLPEncoder(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.float().unsqueeze(-1)
        pooled = (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.network(pooled)


class RelationalGATLayer(nn.Module):
    RELATIONS = ("invoke", "reverse", "self")

    def __init__(self, dimension: int, dropout: float) -> None:
        super().__init__()
        self.query = nn.Linear(dimension, dimension, bias=False)
        self.keys = nn.ModuleDict(
            {relation: nn.Linear(dimension, dimension, bias=False) for relation in self.RELATIONS}
        )
        self.values = nn.ModuleDict(
            {relation: nn.Linear(dimension, dimension, bias=False) for relation in self.RELATIONS}
        )
        self.relation_bias = nn.ParameterDict(
            {relation: nn.Parameter(torch.zeros(())) for relation in self.RELATIONS}
        )
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)
        self.scale = dimension ** -0.5

    def forward(
        self,
        nodes: torch.Tensor,
        edges: list[tuple[int, int, str]],
        onset: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[tuple[int, int, str], torch.Tensor]]:
        incoming: dict[int, list[tuple[int, str]]] = {
            index: [] for index in range(nodes.shape[0])
        }
        for source, target, relation in edges:
            if relation != "self":
                source_onset = int(onset[source].item())
                target_onset = int(onset[target].item())
                if (
                    source_onset >= 0
                    and target_onset >= 0
                    and source_onset > target_onset + 1
                ):
                    continue
            incoming[target].append((source, relation))

        updated = []
        attentions: dict[tuple[int, int, str], torch.Tensor] = {}
        queries = self.query(nodes)
        for target in range(nodes.shape[0]):
            candidates = incoming[target]
            if not candidates:
                candidates = [(target, "self")]
            scores = []
            messages = []
            for source, relation in candidates:
                score = (
                    queries[target] * self.keys[relation](nodes[source])
                ).sum() * self.scale + self.relation_bias[relation]
                scores.append(F.leaky_relu(score, negative_slope=0.2))
                messages.append(self.values[relation](nodes[source]))
            weights = torch.softmax(torch.stack(scores), dim=0)
            message = torch.sum(torch.stack(messages) * weights.unsqueeze(1), dim=0)
            updated.append(message)
            for (source, relation), weight in zip(candidates, weights):
                attentions[(source, target, relation)] = weight
        output = self.norm(nodes + self.dropout(torch.stack(updated)))
        return F.gelu(output), attentions


class PerceptionRCA(nn.Module):
    def __init__(
        self,
        metric_dim: int,
        log_dim: int,
        trace_dim: int,
        config: ModelConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.metric_encoder = MaskedGRUEncoder(
            metric_dim, config.hidden_dim, config.embedding_dim
        )
        self.log_encoder = PooledMLPEncoder(
            log_dim, config.hidden_dim, config.embedding_dim, config.dropout
        )
        self.trace_encoder = PooledMLPEncoder(
            trace_dim, config.hidden_dim, config.embedding_dim, config.dropout
        )
        self.shared_norm = nn.LayerNorm(config.embedding_dim)
        self.content_scorers = nn.ModuleDict(
            {modality: nn.Linear(config.embedding_dim, 1) for modality in MODALITIES}
        )
        # Start from a neutral content prior (sigmoid(0)=0.5). Random initial
        # modality preferences are especially harmful in the small-data pilot
        # and have no evidential basis.
        for scorer in self.content_scorers.values():
            nn.init.zeros_(scorer.weight)
            nn.init.zeros_(scorer.bias)
        self.graph_layers = nn.ModuleList(
            [
                RelationalGATLayer(config.embedding_dim, config.dropout)
                for _ in range(config.graph_layers)
            ]
        )
        self.role_head = nn.Linear(config.embedding_dim, 3)
        self.localization = nn.Linear(3, 1)

    @staticmethod
    def _graph_edges(case: dict[str, object]) -> list[tuple[int, int, str]]:
        edges: list[tuple[int, int, str]] = []
        for source, target in case["edges"]:
            if source in SERVICE_INDEX and target in SERVICE_INDEX:
                source_index = SERVICE_INDEX[source]
                target_index = SERVICE_INDEX[target]
                edges.append((source_index, target_index, "invoke"))
                edges.append((target_index, source_index, "reverse"))
        for index in range(len(case["services"])):
            edges.append((index, index, "self"))
        return edges

    def forward(self, case: dict[str, object]) -> dict[str, object]:
        encoder_masks = {
            modality: (
                case[f"{modality}_mask"]
                if self.config.use_availability_mask
                else torch.ones_like(case[f"{modality}_mask"], dtype=torch.bool)
            )
            for modality in MODALITIES
        }
        embeddings = {
            "metric": self.metric_encoder(case["metric_x"], encoder_masks["metric"]),
            "log": self.log_encoder(case["log_x"], encoder_masks["log"]),
            "trace": self.trace_encoder(case["trace_x"], encoder_masks["trace"]),
        }
        available = torch.stack(
            [
                encoder_masks[modality].any(dim=1)
                for modality in MODALITIES
            ],
            dim=1,
        )
        quality = torch.stack(
            [
                modality_quality(encoder_masks[modality])
                for modality in MODALITIES
            ],
            dim=1,
        )
        normalized_embeddings = []
        content_scores = []
        for modality in MODALITIES:
            embedding = self.shared_norm(embeddings[modality])
            normalized_embeddings.append(embedding)
            content_scores.append(torch.sigmoid(self.content_scorers[modality](embedding)).squeeze(1))
        embedding_stack = torch.stack(normalized_embeddings, dim=1)
        content = torch.stack(content_scores, dim=1)
        if self.config.fusion_mode == "static":
            fusion_logits = torch.zeros_like(quality)
        else:
            fusion_logits = (
                self.config.fusion_lambda * quality
                + (1.0 - self.config.fusion_lambda) * content
            )
        fusion_logits = fusion_logits.masked_fill(~available, -1e9)
        fusion_weights = torch.softmax(fusion_logits, dim=1)
        fusion_weights = fusion_weights * available.float()
        fusion_weights = fusion_weights / fusion_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        node_state = torch.sum(
            embedding_stack * fusion_weights.unsqueeze(-1), dim=1
        )

        graph_edges = self._graph_edges(case)
        layer_attentions = []
        for layer in self.graph_layers:
            node_state, attention = layer(node_state, graph_edges, case["onset"])
            layer_attentions.append(attention)
        role_logits = self.role_head(node_state)
        role_probabilities = torch.softmax(role_logits, dim=1)
        anomaly = 1.0 - role_probabilities[:, 2]

        phi = node_state.new_zeros(node_state.shape[0])
        for node in range(node_state.shape[0]):
            outgoing = node_state.new_tensor(0.0)
            incoming = node_state.new_tensor(0.0)
            for source, target, relation in graph_edges:
                if relation != "invoke":
                    continue
                weights = [
                    layer_attention.get((source, target, relation))
                    for layer_attention in layer_attentions
                ]
                weights = [weight for weight in weights if weight is not None]
                if not weights:
                    continue
                weight = torch.stack(weights).mean()
                if source == node:
                    outgoing = outgoing + weight * anomaly[target]
                if target == node:
                    incoming = incoming + weight * anomaly[source]
            phi[node] = (outgoing - incoming) / (outgoing + incoming + 1e-6)

        tau = node_state.new_zeros(node_state.shape[0])
        onset = case["onset"]
        for node in range(node_state.shape[0]):
            neighbors = set()
            for source, target, relation in graph_edges:
                if relation != "invoke":
                    continue
                if source == node:
                    neighbors.add(target)
                if target == node:
                    neighbors.add(source)
            comparable = [
                neighbor
                for neighbor in neighbors
                if int(onset[node].item()) >= 0 and int(onset[neighbor].item()) >= 0
            ]
            if comparable:
                tau[node] = torch.stack(
                    [
                        (onset[node] <= onset[neighbor] + 1).float()
                        for neighbor in comparable
                    ]
                ).mean()

        localization_features = torch.stack(
            [role_probabilities[:, 0], phi, tau], dim=1
        )
        localization_logits = self.localization(localization_features).squeeze(1)
        localization_probabilities = torch.softmax(localization_logits, dim=0)
        return {
            "role_logits": role_logits,
            "role_probabilities": role_probabilities,
            "anomaly_scores": anomaly,
            "source_likeness": phi,
            "temporal_lead": tau,
            "localization_logits": localization_logits,
            "localization_probabilities": localization_probabilities,
            "fusion_weights": fusion_weights,
            "node_embeddings": node_state,
            "edge_attentions": layer_attentions,
        }
