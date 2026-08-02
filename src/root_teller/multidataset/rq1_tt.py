from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from root_teller.module1 import data as data_module
from root_teller.module1 import evaluate as evaluate_module
from root_teller.module1 import features as features_module
from root_teller.module1 import model as model_module
from root_teller.module1.baseline import metrics
from root_teller.module1.config import FeatureConfig
from root_teller.module1.data import (
    apply_structured_dropout,
    fit_reference,
    normalize_case,
    to_torch_case,
    weak_role_targets,
)
from root_teller.module1.features import CaseSpec, load_case, save_case
from root_teller.module1.model import ModelConfig, PerceptionRCA
from root_teller.module1.train import role_class_weights, seed_everything
from root_teller.module2 import window_export
from root_teller.paths import workspace_root


PROJECT = workspace_root()
DATASET = PROJECT / "dataset" / "RCAEval RE" / "RE2" / "RE2-TT" / "RE2-TT"
MANIFEST = (
    PROJECT
    / "evaluation"
    / "rq1"
    / "manifests"
    / "case_catalog.csv"
)
RUN_ROOT = PROJECT / "runs" / "rq1_multidataset_clean" / "re2_tt"
CACHE_ROOT = PROJECT / "cache" / "rq1_multidataset_clean" / "re2_tt"


class TTPaths:
    clean_root = DATASET
    corrupted_root = PROJECT / "dataset" / "dataset_corrupted" / "_unused"
    cache_root = CACHE_ROOT / "features"


def specs() -> list[CaseSpec]:
    frame = pd.read_csv(MANIFEST)
    frame = frame.loc[frame["dataset_system"].eq("RCAEval RE2-TT")]
    return [
        CaseSpec(
            incident_id=str(row.incident_id),
            split=str(row.split),
            eligible=bool(row.eligible),
            root_cause_service=str(row.root_cause_service),
            fault_type=str(row.fault_type),
            inject_time=float(row.inject_time),
        )
        for row in frame.itertuples(index=False)
    ]


def discover_services() -> tuple[str, ...]:
    first = DATASET / specs()[0].incident_id / "metrics.csv"
    columns = pd.read_csv(first, nrows=0).columns
    services = set()
    for column in columns:
        if column == "time" or "_" not in column:
            continue
        service = column.split("_", 1)[0].strip().lower()
        # ``ts_container-network-*`` is a namespace-level aggregate, not a
        # deployable service candidate.
        if service.startswith("gke-") or service in {"loadgenerator", "ts"}:
            continue
        services.add(service)
    return tuple(sorted(services))


def configure_services(services: tuple[str, ...]) -> dict[str, int]:
    index = {service: offset for offset, service in enumerate(services)}
    # Preserve the frozen RE2-OB source and patch only this process.
    features_module.SERVICES = services
    for module in (features_module, data_module, model_module, evaluate_module, window_export):
        module.SERVICE_INDEX = index
    return index


def feature_path(spec: CaseSpec, config: FeatureConfig) -> Path:
    identity = hashlib.sha256(
        json.dumps(
            {
                "feature_config": config.to_dict(),
                "services": list(features_module.SERVICES),
                "adapter": "re2-tt-v1",
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:12]
    return CACHE_ROOT / "features" / identity / spec.split / f"{spec.incident_id}.npz"


def build(
    config: FeatureConfig,
    overwrite: bool = False,
    shard_index: int = 0,
    shard_count: int = 1,
) -> dict[str, object]:
    paths = TTPaths()
    completed = 0
    failures = []
    started = time.time()
    selected_specs = specs()[shard_index::shard_count]
    for spec in selected_specs:
        destination = feature_path(spec, config)
        if destination.exists() and not overwrite:
            completed += 1
            continue
        try:
            case = features_module.extract_case(paths, spec, "CLEAN", config)
            save_case(destination, case)
            completed += 1
            print(json.dumps({"built": completed, "case": spec.incident_id}), flush=True)
        except Exception as error:
            failures.append({"case": spec.incident_id, "error": repr(error)})
            print(json.dumps(failures[-1]), flush=True)
    payload = {
        "completed": completed,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "failures": failures,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    (RUN_ROOT / f"build_summary_shard{shard_index}_of_{shard_count}.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def load_split(split: str, config: FeatureConfig) -> list[dict[str, object]]:
    selected = [
        item
        for item in specs()
        if item.split == split and (split != "test" or item.eligible)
    ]
    return [load_case(feature_path(item, config)) for item in selected]


@torch.no_grad()
def evaluate(
    model: PerceptionRCA,
    cases: list[dict[str, object]],
    reference: data_module.ReferenceStats,
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    model.eval()
    ranks = []
    predictions = []
    for case in cases:
        tensor_case = to_torch_case(case, reference, device)
        output = model(tensor_case)
        order = torch.argsort(
            output["localization_probabilities"], descending=True
        ).cpu().tolist()
        target = int(tensor_case["target_index"])
        rank = order.index(target) + 1
        ranks.append(rank)
        predictions.append(
            {
                "incident_id": case["incident_id"],
                "target": case["root_cause_service"],
                "rank": rank,
                "top5": [case["services"][index] for index in order[:5]],
            }
        )
    return metrics(ranks), predictions


def train_and_refit(
    config: FeatureConfig,
    seed: int = 20260724,
    max_epochs: int = 300,
    patience: int = 50,
) -> Path:
    seed_everything(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_cases = load_split("train", config)
    validation_cases = load_split("validation", config)
    reference = fit_reference(train_cases)
    model_config = ModelConfig()
    first = train_cases[0]
    model = PerceptionRCA(
        first["metric_x"].shape[2],
        first["log_x"].shape[2],
        first["trace_x"].shape[2],
        model_config,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    weights = role_class_weights(train_cases, reference, device)
    role_targets = [
        torch.as_tensor(
            weak_role_targets(normalize_case(case, reference), reference),
            dtype=torch.long,
            device=device,
        )
        for case in train_cases
    ]
    tensor_cases = [to_torch_case(case, reference, device) for case in train_cases]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 17)
    best_score = -1.0
    best_epoch = 0
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        loss_sum = 0.0
        for index in torch.randperm(len(tensor_cases), generator=generator).tolist():
            optimizer.zero_grad(set_to_none=True)
            case = apply_structured_dropout(tensor_cases[index], generator, 0.25, 0.25)
            output = model(case)
            role_loss = F.cross_entropy(
                output["role_logits"],
                role_targets[index],
                weight=weights,
                ignore_index=-100,
            )
            target = torch.as_tensor(
                [int(case["target_index"])], dtype=torch.long, device=device
            )
            localization_loss = F.cross_entropy(
                output["localization_logits"].unsqueeze(0), target
            )
            loss = role_loss + localization_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.item())
        if epoch == 1 or epoch % 5 == 0:
            validation_metrics, _ = evaluate(model, validation_cases, reference, device)
            score = validation_metrics["Avg@5"]
            history.append(
                {
                    "epoch": epoch,
                    "loss": round(loss_sum / len(train_cases), 6),
                    "validation": validation_metrics,
                }
            )
            print(json.dumps(history[-1]), flush=True)
            if score > best_score + 1e-9:
                best_score = score
                best_epoch = epoch
            if epoch - best_epoch >= patience:
                break

    # Frozen protocol: select epoch on validation, then refit from scratch on
    # train+validation. No held-out label is read before the refit is complete.
    development = train_cases + validation_cases
    seed_everything(seed)
    reference = fit_reference(development)
    model = PerceptionRCA(
        first["metric_x"].shape[2],
        first["log_x"].shape[2],
        first["trace_x"].shape[2],
        model_config,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    weights = role_class_weights(development, reference, device)
    role_targets = [
        torch.as_tensor(
            weak_role_targets(normalize_case(case, reference), reference),
            dtype=torch.long,
            device=device,
        )
        for case in development
    ]
    tensor_cases = [to_torch_case(case, reference, device) for case in development]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 17)
    for epoch in range(1, best_epoch + 1):
        model.train()
        for index in torch.randperm(len(tensor_cases), generator=generator).tolist():
            optimizer.zero_grad(set_to_none=True)
            case = apply_structured_dropout(tensor_cases[index], generator, 0.25, 0.25)
            output = model(case)
            role_loss = F.cross_entropy(
                output["role_logits"],
                role_targets[index],
                weight=weights,
                ignore_index=-100,
            )
            target = torch.as_tensor(
                [int(case["target_index"])], dtype=torch.long, device=device
            )
            loss = role_loss + F.cross_entropy(
                output["localization_logits"].unsqueeze(0), target
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    run_dir = RUN_ROOT / f"module1_refit_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "reference": reference.state_dict(),
        "feature_config": config.to_dict(),
        "model_config": asdict(model_config),
        "seed": seed,
        "epochs": best_epoch,
        "services": list(first["services"]),
        "protocol": "validation-selected-epoch-refit-train-plus-validation",
    }
    checkpoint_path = run_dir / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    test_metrics, predictions = evaluate(
        model, load_split("test", config), reference, device
    )
    result = {
        "dataset": "RCAEval RE2-TT",
        "stage": "module1",
        "seed": seed,
        "selected_epoch": best_epoch,
        "validation_best_avg5": best_score,
        "test_metrics": test_metrics,
        "test_cases": len(predictions),
        "candidate_services": len(first["services"]),
        "predictions": predictions,
    }
    (run_dir / "test_results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "validation_history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result | {"predictions": "saved"}, indent=2), flush=True)
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["build", "train", "all"], default="all")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-backend", choices=["hash", "sbert"], default="sbert")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    services = discover_services()
    configure_services(services)
    config = FeatureConfig(log_backend=args.log_backend)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    protocol = {
        "dataset": "RCAEval RE2-TT",
        "services": list(services),
        "candidate_count": len(services),
        "feature_config": config.to_dict(),
        "split_counts": {
            split: sum(item.split == split and (split != "test" or item.eligible) for item in specs())
            for split in ("train", "validation", "test")
        },
        "test_labels_used_for_tuning": False,
    }
    (RUN_ROOT / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )
    if args.stage in {"build", "all"}:
        summary = build(
            config, args.overwrite, args.shard_index, args.shard_count
        )
        if summary["failures"]:
            raise SystemExit(1)
    if args.stage in {"train", "all"}:
        train_and_refit(config)


if __name__ == "__main__":
    main()
