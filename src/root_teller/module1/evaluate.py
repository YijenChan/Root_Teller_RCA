from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .baseline import metrics
from .config import CONDITIONS, FeatureConfig, Paths, SERVICE_INDEX
from .data import ReferenceStats, to_torch_case
from .features import cache_path, load_case, load_case_specs
from .model import ModelConfig, PerceptionRCA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument(
        "--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS)
    )
    return parser.parse_args()


def opaque_case_id(raw_incident_id: str) -> str:
    digest = hashlib.sha256(
        ("RCAEval RE2-OB\0" + raw_incident_id).encode("utf-8")
    ).hexdigest()
    return f"re2ob-{digest[:16]}"


def _edge_attention(
    output: dict[str, object], source: int, target: int
) -> float:
    values = []
    for layer in output["edge_attentions"]:
        weight = layer.get((source, target, "invoke"))
        if weight is not None:
            values.append(float(weight.detach().cpu().item()))
    return float(np.mean(values)) if values else 0.0


def _chains(
    case: dict[str, object],
    output: dict[str, object],
    candidate: int,
    case_id: str,
    max_hops: int = 4,
    keep: int = 3,
) -> list[dict[str, object]]:
    adjacency: dict[int, list[int]] = {
        index: [] for index in range(len(case["services"]))
    }
    for source, target in case["edges"]:
        if source in SERVICE_INDEX and target in SERVICE_INDEX:
            adjacency[SERVICE_INDEX[source]].append(SERVICE_INDEX[target])
    anomaly = output["anomaly_scores"].detach().cpu().numpy()
    onset = case["onset"].detach().cpu().numpy()
    beam: list[tuple[list[int], float]] = [([candidate], float(anomaly[candidate]))]
    all_paths = list(beam)
    for _ in range(max_hops):
        extended: list[tuple[list[int], float]] = []
        for path, _ in beam:
            source = path[-1]
            for target in adjacency[source]:
                if target in path:
                    continue
                if onset[source] >= 0 and onset[target] >= 0 and onset[source] > onset[target] + 1:
                    continue
                new_path = path + [target]
                edge_strengths = [
                    float(anomaly[v])
                    + _edge_attention(output, u, v)
                    for u, v in zip(new_path[:-1], new_path[1:])
                ]
                score = float(np.mean(edge_strengths))
                extended.append((new_path, score))
        if not extended:
            break
        extended.sort(key=lambda item: item[1], reverse=True)
        beam = extended[:keep]
        all_paths.extend(beam)
    unique: dict[tuple[int, ...], float] = {}
    for path, score in all_paths:
        key = tuple(path)
        unique[key] = max(unique.get(key, -np.inf), score)
    ranked = sorted(unique.items(), key=lambda item: item[1], reverse=True)[:keep]
    return [
        {
            "entities": [case["services"][index] for index in path],
            "score": round(score, 8),
            "temporal_status": (
                "uncertain"
                if any(onset[index] < 0 for index in path)
                else "consistent"
            ),
            "evidence_id": (
                f"{case_id}|{case['condition']}|chain|"
                + "->".join(case["services"][index] for index in path)
            ),
        }
        for path, score in ranked
    ]


def evidence_pack(
    case: dict[str, object],
    output: dict[str, object],
    checkpoint_hash: str,
) -> dict[str, object]:
    probabilities = output["localization_probabilities"].detach().cpu().numpy()
    roles = output["role_probabilities"].detach().cpu().numpy()
    anomaly = output["anomaly_scores"].detach().cpu().numpy()
    source_likeness = output["source_likeness"].detach().cpu().numpy()
    temporal_lead = output["temporal_lead"].detach().cpu().numpy()
    fusion = output["fusion_weights"].detach().cpu().numpy()
    onset = case["onset"].detach().cpu().numpy()
    order = np.argsort(-probabilities)
    case_id = opaque_case_id(str(case["incident_id"]))
    candidates = []
    for rank, index in enumerate(order[:5], start=1):
        service = case["services"][index]
        candidates.append(
            {
                "rank": rank,
                "entity_id": service,
                "evidence_id": (
                    f"{case_id}|{case['condition']}|node|{service}"
                ),
                "localization_probability": round(float(probabilities[index]), 8),
                "diagnostic_role": ["root", "propagation", "normal"][
                    int(np.argmax(roles[index]))
                ],
                "role_probabilities": {
                    "root": round(float(roles[index, 0]), 8),
                    "propagation": round(float(roles[index, 1]), 8),
                    "normal": round(float(roles[index, 2]), 8),
                },
                "anomaly_score": round(float(anomaly[index]), 8),
                "source_likeness": round(float(source_likeness[index]), 8),
                "temporal_lead": round(float(temporal_lead[index]), 8),
                "onset_bin": None if onset[index] < 0 else int(onset[index]),
                "fusion_weights": {
                    "metric": round(float(fusion[index, 0]), 8),
                    "log": round(float(fusion[index, 1]), 8),
                    "trace": round(float(fusion[index, 2]), 8),
                },
                "candidate_chains": _chains(
                    case, output, int(index), case_id=case_id
                ),
            }
        )
    modality_available = {
        modality: bool(case[f"{modality}_mask"].any().item())
        for modality in ("metric", "log", "trace")
    }
    return {
        "schema_version": "module1-evidence-pack-1.0",
        "incident_id": case_id,
        "condition": case["condition"],
        "window": {
            "start_time": case["start_time"],
            "bin_seconds": case["bin_seconds"],
            "bin_count": int(case["metric_x"].shape[1]),
        },
        "ranked_candidates": candidates,
        "dependency_edges": [list(edge) for edge in case["edges"]],
        "quality_notes": {
            "modality_available": modality_available,
            "unavailable_modalities": [
                modality
                for modality, available in modality_available.items()
                if not available
            ],
        },
        "provenance": {
            "checkpoint_sha256": checkpoint_hash,
            "feature_cache_condition": case["condition"],
            "opaque_case_id": case_id,
        },
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    feature_payload = dict(checkpoint["feature_config"])
    if isinstance(feature_payload.get("metric_groups"), list):
        feature_payload["metric_groups"] = tuple(feature_payload["metric_groups"])
    feature_config = FeatureConfig(**feature_payload)
    model_config = ModelConfig(**checkpoint["model_config"])
    reference = ReferenceStats.from_state_dict(checkpoint["reference"])
    model = PerceptionRCA(
        metric_dim=len(feature_config.metric_groups),
        log_dim=feature_config.log_content_dim + feature_config.log_extra_dim,
        trace_dim=feature_config.trace_dim,
        config=model_config,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    paths = Paths()
    specs = [
        spec
        for spec in load_case_specs(paths)
        if spec.split == args.split and (args.split != "test" or spec.eligible)
    ]
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    results: dict[str, object] = {}
    error_count = 0
    for condition in args.conditions:
        ranks = []
        predictions = []
        packs = []
        for spec in specs:
            case = load_case(cache_path(paths, spec, condition, feature_config))
            tensor_case = to_torch_case(case, reference, device)
            with torch.no_grad():
                output = model(tensor_case)
            probabilities = output["localization_probabilities"]
            if not torch.isfinite(probabilities).all():
                error_count += 1
                continue
            order = torch.argsort(probabilities, descending=True).cpu().tolist()
            target = int(tensor_case["target_index"])
            rank = order.index(target) + 1
            ranks.append(rank)
            predictions.append(
                {
                    "incident_id": spec.incident_id,
                    "target": spec.root_cause_service,
                    "rank": rank,
                    "top5": [case["services"][index] for index in order[:5]],
                }
            )
            packs.append(evidence_pack(tensor_case, output, checkpoint_hash))
        results[condition] = {
            "metrics": metrics(ranks),
            "operational_cases": len(ranks),
            "expected_cases": len(specs),
            "predictions": predictions,
        }
        pack_dir = args.output_dir / "evidence_packs" / condition
        pack_dir.mkdir(parents=True, exist_ok=True)
        for pack in packs:
            destination = pack_dir / (pack["incident_id"].replace("/", "__") + ".json")
            destination.write_text(
                json.dumps(pack, indent=2) + "\n", encoding="utf-8"
            )
    summary = {
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "seed": checkpoint["seed"],
        "conditions": {
            condition: {
                key: value
                for key, value in payload.items()
                if key != "predictions"
            }
            for condition, payload in results.items()
        },
        "nan_or_error_cases": error_count,
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    if error_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
