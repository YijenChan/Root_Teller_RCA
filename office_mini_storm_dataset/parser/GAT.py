# GAT.py (anonymized & open-source safe version)
# This script encodes graph nodes into embeddings using Graph Attention Networks (GAT).
# It is sanitized for public release: no absolute paths, no raw identifiers, and no real timestamps.

import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import numpy as np
from datetime import datetime
from typing import List, Dict
from config import GAT_CONFIG


# -----------------------------
#   GAT Node Encoder
# -----------------------------
class GATNodeEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 32,
        num_heads: int = 4,
        num_layers: int = 2
    ):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # Input layer
        self.convs.append(GATConv(input_dim, hidden_dim, heads=num_heads, dropout=0.2))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim * num_heads))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=0.2))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim * num_heads))

        # Output layer
        self.convs.append(
            GATConv(hidden_dim * num_heads if num_layers > 1 else input_dim,
                    output_dim, heads=1, concat=False, dropout=0.2)
        )

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            if i < self.num_layers - 1:
                x = self.batch_norms[i](x)
                x = F.elu(x)
                x = F.dropout(x, p=0.2, training=self.training)
        return x


# -----------------------------
#   Graph Embedding Pipeline
# -----------------------------
class GraphEmbeddingPipeline:
    def __init__(self, model_path: str = None):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def load_graphs_from_pkl(self, pkl_path: str) -> List[Dict]:
        """Load serialized graphs (expected after anonymization)."""
        with open(pkl_path, "rb") as f:
            graphs = pickle.load(f)
        return graphs

    def initialize_model(self, input_dim: int, output_dim: int = 32):
        self.model = GATNodeEncoder(
            input_dim=input_dim,
            hidden_dim=GAT_CONFIG["hidden_dim"],
            output_dim=output_dim,
            num_heads=GAT_CONFIG["num_heads"],
            num_layers=GAT_CONFIG["num_layers"]
        ).to(self.device)
        self.output_dim = output_dim

    def embed_single_graph(self, graph_dict: Dict) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not initialized.")
        x = torch.tensor(graph_dict["node_features"], dtype=torch.float32).to(self.device)
        edge_index = torch.tensor(graph_dict["edge_index"], dtype=torch.long).to(self.device)

        self.model.eval()
        with torch.no_grad():
            node_embeddings = self.model(x, edge_index)
        return node_embeddings.cpu().numpy()

    def process_graphs(self, input_pkl: str, output_txt: str, output_npy: str):
        graphs = self.load_graphs_from_pkl(input_pkl)
        if not graphs:
            print("[WARN] No graphs found.")
            return

        if self.model is None:
            input_dim = graphs[0]["node_features"].shape[1]
            self.initialize_model(input_dim, GAT_CONFIG["output_dim"])

        all_embeddings = []

        with open(output_txt, "w", encoding="utf-8") as f_txt:
            for idx, graph_dict in enumerate(graphs):
                timestamp = graph_dict.get("timestamp", 0)
                label = graph_dict.get("label", "N/A")

                # Convert absolute timestamp to relative offset
                relative_time = timestamp - graphs[0].get("timestamp", 0)
                # Mask identifiers for open release
                trace_id = "<trace-" + str(idx + 1) + ">"
                span_ids = [f"<span-{i}>" for i in range(len(graph_dict.get("span_ids", [])))]

                embeddings = self.embed_single_graph(graph_dict)
                all_embeddings.append(embeddings)

                # Write metadata header
                f_txt.write(f"# Graph {idx + 1}\n")
                f_txt.write(f"# Trace ID: {trace_id}\n")
                f_txt.write(f"# Timestamp (relative): {relative_time}\n")
                f_txt.write(f"# Label: {label}\n")
                f_txt.write(f"# Shape: {embeddings.shape}\n")
                f_txt.write("# Node Embeddings:\n")

                for node_idx, node_emb in enumerate(embeddings):
                    emb_str = " ".join([f"{val:.6f}" for val in node_emb])
                    span_tag = span_ids[node_idx] if node_idx < len(span_ids) else f"<span-{node_idx}>"
                    f_txt.write(f"{node_idx} ({span_tag}): {emb_str}\n")

                f_txt.write("\n")

        np.save(output_npy, np.array(all_embeddings, dtype=object))
        print(f"[OK] Embeddings saved to: {output_txt}, {output_npy}")

    def save_model(self, path: str):
        if self.model is None:
            return
        torch.save(
            {"model_state_dict": self.model.state_dict(), "output_dim": self.output_dim},
            path
        )
        print(f"[OK] Model saved to: {path}")

    def load_model(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        if "output_dim" in checkpoint:
            self.output_dim = checkpoint["output_dim"]
        if self.model is not None:
            self.model.load_state_dict(checkpoint["model_state_dict"])


# -----------------------------
#   Main Entry
# -----------------------------
def main():
    pipeline = GraphEmbeddingPipeline()

    input_pkl = GAT_CONFIG["input_pkl"]
    output_txt = GAT_CONFIG["output_txt"]
    output_npy = GAT_CONFIG["output_npy"]
    model_path = GAT_CONFIG["model_path"]

    if not os.path.exists(input_pkl):
        print(f"[WARN] Input file not found: {input_pkl}")
        return

    pipeline.process_graphs(input_pkl, output_txt, output_npy)
    pipeline.save_model(model_path)


if __name__ == "__main__":
    main()
