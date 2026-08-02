from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .baseline import metrics
from .config import CONDITIONS, FeatureConfig, Paths
from .data import (
    ReferenceStats,
    apply_structured_dropout,
    fit_reference,
    normalize_case,
    to_torch_case,
    weak_role_targets,
)
from .features import cache_path, load_case, load_case_specs
from .model import ModelConfig, PerceptionRCA


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_split(
    split: str, condition: str, paths: Paths, feature_config: FeatureConfig
) -> list[dict[str, object]]:
    specs = [
        spec
        for spec in load_case_specs(paths)
        if spec.split == split and (split != "test" or spec.eligible)
    ]
    return [
        load_case(cache_path(paths, spec, condition, feature_config))
        for spec in specs
    ]


def role_class_weights(
    cases: list[dict[str, object]], reference: ReferenceStats, device: torch.device
) -> torch.Tensor:
    counts = np.zeros(3, dtype=np.float64)
    for case in cases:
        normalized = normalize_case(case, reference)
        targets = weak_role_targets(normalized, reference)
        for role in range(3):
            counts[role] += np.sum(targets == role)
    inverse = 1.0 / np.maximum(counts, 1.0)
    inverse = inverse / inverse.mean()
    return torch.as_tensor(inverse, dtype=torch.float32, device=device)


@torch.no_grad()
def evaluate(
    model: PerceptionRCA,
    cases: list[dict[str, object]],
    reference: ReferenceStats,
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    model.eval()
    ranks: list[int] = []
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
                "condition": case["condition"],
                "target": case["root_cause_service"],
                "rank": rank,
                "top5": [case["services"][index] for index in order[:5]],
                "fusion_weights": {
                    case["services"][index]: {
                        modality: round(
                            float(output["fusion_weights"][index, offset].item()), 6
                        )
                        for offset, modality in enumerate(("metric", "log", "trace"))
                    }
                    for index in order[:5]
                },
            }
        )
    return metrics(ranks), predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--embedding-dim", type=int, default=48)
    parser.add_argument("--graph-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--fusion-lambda", type=float, default=0.6)
    parser.add_argument("--full-dropout-probability", type=float, default=0.25)
    parser.add_argument("--suffix-dropout-probability", type=float, default=0.25)
    parser.add_argument("--log-backend", choices=["hash", "sbert"], default="sbert")
    parser.add_argument("--fusion-mode", choices=["adaptive", "static"], default="adaptive")
    parser.add_argument("--disable-availability-mask", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    paths = Paths()
    feature_config = FeatureConfig(log_backend=args.log_backend)
    model_config = ModelConfig(
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        graph_layers=args.graph_layers,
        dropout=args.dropout,
        fusion_lambda=args.fusion_lambda,
        fusion_mode=args.fusion_mode,
        use_availability_mask=not args.disable_availability_mask,
    )
    train_cases = load_split("train", "CLEAN", paths, feature_config)
    validation = {
        condition: load_split("validation", condition, paths, feature_config)
        for condition in CONDITIONS
    }
    reference = fit_reference(train_cases)
    first = train_cases[0]
    model = PerceptionRCA(
        metric_dim=first["metric_x"].shape[2],
        log_dim=first["log_x"].shape[2],
        trace_dim=first["trace_x"].shape[2],
        config=model_config,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
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
    generator.manual_seed(args.seed + 17)

    history = []
    best_score = -1.0
    best_epoch = 0
    best_state = None
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(tensor_cases), generator=generator).tolist()
        epoch_loss = 0.0
        for index in order:
            optimizer.zero_grad(set_to_none=True)
            augmented = apply_structured_dropout(
                tensor_cases[index],
                generator,
                args.full_dropout_probability,
                args.suffix_dropout_probability,
            )
            output = model(augmented)
            targets = role_targets[index]
            role_loss = F.cross_entropy(
                output["role_logits"],
                targets,
                weight=weights,
                ignore_index=-100,
            )
            target = torch.as_tensor(
                [int(augmented["target_index"])],
                dtype=torch.long,
                device=device,
            )
            localization_loss = F.cross_entropy(
                output["localization_logits"].unsqueeze(0), target
            )
            loss = role_loss + localization_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += float(loss.item())

        if epoch == 1 or epoch % 5 == 0:
            validation_metrics = {}
            for condition, cases in validation.items():
                condition_metrics, _ = evaluate(model, cases, reference, device)
                validation_metrics[condition] = condition_metrics
            clean_score = validation_metrics["CLEAN"]["Avg@5"]
            robust_score = float(
                np.mean(
                    [
                        validation_metrics[condition]["Avg@5"]
                        for condition in CONDITIONS[1:]
                    ]
                )
            )
            selection_score = 0.5 * clean_score + 0.5 * robust_score
            row = {
                "epoch": epoch,
                "loss": round(epoch_loss / len(tensor_cases), 6),
                "selection_score": round(selection_score, 6),
                "validation": validation_metrics,
            }
            history.append(row)
            print(json.dumps(row), flush=True)
            if selection_score > best_score + 1e-9:
                best_score = selection_score
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            if epoch - best_epoch >= args.patience:
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    final_validation = {}
    for condition, cases in validation.items():
        condition_metrics, predictions = evaluate(model, cases, reference, device)
        final_validation[condition] = {
            "metrics": condition_metrics,
            "predictions": predictions,
        }
    config_payload = {
        "seed": args.seed,
        "feature_config": feature_config.to_dict(),
        "model_config": asdict(model_config),
        "training": {
            "epochs_requested": args.epochs,
            "best_epoch": best_epoch,
            "patience": args.patience,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "full_dropout_probability": args.full_dropout_probability,
            "suffix_dropout_probability": args.suffix_dropout_probability,
        },
    }
    config_json = json.dumps(config_payload, sort_keys=True)
    config_hash = hashlib.sha256(config_json.encode()).hexdigest()
    checkpoint = {
        "model_state": best_state,
        "reference": reference.state_dict(),
        "feature_config": feature_config.to_dict(),
        "model_config": asdict(model_config),
        "config_hash": config_hash,
        "seed": args.seed,
        "best_epoch": best_epoch,
    }
    torch.save(checkpoint, args.run_dir / "checkpoint.pt")
    (args.run_dir / "config.json").write_text(
        json.dumps(config_payload | {"config_hash": config_hash}, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.run_dir / "history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    (args.run_dir / "validation_results.json").write_text(
        json.dumps(final_validation, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "status": "complete",
        "device": str(device),
        "best_epoch": best_epoch,
        "best_selection_score": round(best_score, 6),
        "elapsed_seconds": round(time.time() - started, 3),
        "config_hash": config_hash,
        "validation_metrics": {
            condition: payload["metrics"]
            for condition, payload in final_validation.items()
        },
    }
    (args.run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
