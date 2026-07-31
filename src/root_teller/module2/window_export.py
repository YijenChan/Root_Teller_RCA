from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from root_teller.module1.config import FeatureConfig, Paths, SERVICE_INDEX
from root_teller.module1.data import ReferenceStats, to_torch_case
from root_teller.module1.evaluate import _chains, opaque_case_id
from root_teller.module1.features import cache_path, load_case, load_case_specs
from root_teller.module1.model import ModelConfig, PerceptionRCA

from .config import Module2Paths


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--case-limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _slice_case(case: dict[str, object], bin_index: int) -> dict[str, object]:
    sliced = dict(case)
    for modality in ("metric", "log", "trace"):
        sliced[f"{modality}_x"] = case[f"{modality}_x"][:, bin_index : bin_index + 1].copy()
        sliced[f"{modality}_mask"] = case[f"{modality}_mask"][
            :, bin_index : bin_index + 1
        ].copy()
    sliced["start_time"] = float(case["start_time"]) + bin_index * int(case["bin_seconds"])
    return sliced


def _pack(
    case: dict[str, object],
    output: dict[str, object],
    checkpoint_hash: str,
    window_id: str,
    source_bin_index: int,
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
    evidence_prefix = f"{case_id}|{window_id}|CLEAN"
    role_names = ("root", "propagation", "normal")

    entity_evidence = []
    for index, service in enumerate(case["services"]):
        entity_evidence.append(
            {
                "entity_id": service,
                "evidence_id": f"{evidence_prefix}|entity|{service}",
                "localization_probability": round(float(probabilities[index]), 8),
                "diagnostic_role": role_names[int(np.argmax(roles[index]))],
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
            }
        )

    ranked_candidates = []
    chain_case_id = f"{case_id}|{window_id}"
    for rank, index in enumerate(order[:5], start=1):
        record = dict(entity_evidence[index])
        record["rank"] = rank
        record["evidence_id"] = f"{evidence_prefix}|candidate|{record['entity_id']}"
        chains = _chains(case, output, int(index), case_id=chain_case_id)
        for chain in chains:
            chain["evidence_id"] = chain["evidence_id"].replace(
                f"{chain_case_id}|CLEAN|", f"{evidence_prefix}|"
            )
        record["candidate_chains"] = chains
        ranked_candidates.append(record)

    modality_available = {
        modality: bool(case[f"{modality}_mask"].any().item())
        for modality in ("metric", "log", "trace")
    }
    return {
        "schema_version": "module2-window-evidence-pack-1.0",
        "incident_id": case_id,
        "condition": "CLEAN",
        "window": {
            "window_id": window_id,
            "start_time": float(case["start_time"]),
            "end_time": float(case["start_time"]) + int(case["bin_seconds"]),
            "bin_seconds": int(case["bin_seconds"]),
            "source_bin_index": source_bin_index,
            "activation_order": int(window_id[1:]),
        },
        "ranked_candidates": ranked_candidates,
        "entity_evidence": entity_evidence,
        "dependency_edges": [list(edge) for edge in case["edges"]],
        "quality_notes": {
            "modality_available": modality_available,
            "unavailable_modalities": [
                modality for modality, available in modality_available.items() if not available
            ],
        },
        "provenance": {
            "checkpoint_sha256": checkpoint_hash,
            "feature_cache_condition": "CLEAN",
            "opaque_case_id": case_id,
            "window_exporter": "module2-window-export-v1",
        },
    }


def export(
    split: str,
    case_limit: int | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    module1_paths = Paths()
    paths = Module2Paths()
    checkpoint = torch.load(paths.checkpoint, map_location="cpu", weights_only=False)
    feature_payload = dict(checkpoint["feature_config"])
    if isinstance(feature_payload.get("metric_groups"), list):
        feature_payload["metric_groups"] = tuple(feature_payload["metric_groups"])
    feature_config = FeatureConfig(**feature_payload)
    model_config = ModelConfig(**checkpoint["model_config"])
    reference = ReferenceStats.from_state_dict(checkpoint["reference"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = PerceptionRCA(
        metric_dim=len(feature_config.metric_groups),
        log_dim=feature_config.log_content_dim + feature_config.log_extra_dim,
        trace_dim=feature_config.trace_dim,
        config=model_config,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    specs = [
        spec
        for spec in load_case_specs(module1_paths)
        if spec.split == split and (split != "test" or spec.eligible)
    ]
    if case_limit is not None:
        specs = specs[:case_limit]
    checkpoint_hash = hashlib.sha256(paths.checkpoint.read_bytes()).hexdigest()
    split_root = paths.window_pack_root / split
    split_root.mkdir(parents=True, exist_ok=True)
    labels: dict[str, dict[str, str]] = {}
    exported = 0
    skipped = 0
    for spec in specs:
        raw_case = load_case(cache_path(module1_paths, spec, "CLEAN", feature_config))
        case_id = opaque_case_id(spec.incident_id)
        labels[case_id] = {
            "root_cause_service": spec.root_cause_service,
            "fault_type": spec.fault_type,
        }
        case_root = split_root / case_id
        case_root.mkdir(parents=True, exist_ok=True)
        bins = int(raw_case["metric_x"].shape[1])
        for activation_order, bin_index in enumerate(range(bins - 1, -1, -1)):
            window_id = f"W{activation_order:02d}"
            destination = case_root / f"{window_id}.json"
            if destination.exists() and not overwrite:
                skipped += 1
                continue
            tensor_case = to_torch_case(_slice_case(raw_case, bin_index), reference, device)
            with torch.no_grad():
                output = model(tensor_case)
            pack = _pack(tensor_case, output, checkpoint_hash, window_id, bin_index)
            destination.write_text(json.dumps(pack, indent=2) + "\n", encoding="utf-8")
            exported += 1
    private_root = paths.project / "cache" / "module2_re2ob" / "private_evaluator"
    private_root.mkdir(parents=True, exist_ok=True)
    (private_root / f"{split}_labels.json").write_text(
        json.dumps(labels, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "split": split,
        "cases": len(specs),
        "windows_per_case": 24,
        "exported": exported,
        "skipped": skipped,
        "checkpoint_sha256": checkpoint_hash,
        "label_fields_in_public_packs": False,
    }
    (split_root / "export_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    args = _args()
    print(json.dumps(export(args.split, args.case_limit, args.overwrite), indent=2))


if __name__ == "__main__":
    main()
