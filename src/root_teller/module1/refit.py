from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn import functional as F

from .config import FeatureConfig, Paths
from .data import (
    apply_structured_dropout,
    fit_reference,
    normalize_case,
    to_torch_case,
    weak_role_targets,
)
from .model import ModelConfig, PerceptionRCA
from .train import load_split, role_class_weights, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refit frozen Module 1 config on the full development pool."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--full-dropout-probability", type=float, default=0.25)
    parser.add_argument("--suffix-dropout-probability", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    paths = Paths()
    feature_config = FeatureConfig(log_backend="sbert")
    model_config = ModelConfig(
        hidden_dim=48,
        embedding_dim=48,
        graph_layers=2,
        dropout=0.2,
        fusion_lambda=0.6,
        fusion_mode="adaptive",
        use_availability_mask=True,
    )
    development = load_split("train", "CLEAN", paths, feature_config)
    development += load_split("validation", "CLEAN", paths, feature_config)
    reference = fit_reference(development)
    first = development[0]
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
    class_weights = role_class_weights(development, reference, device)
    tensor_cases = [to_torch_case(case, reference, device) for case in development]
    targets = [
        torch.as_tensor(
            weak_role_targets(normalize_case(case, reference), reference),
            dtype=torch.long,
            device=device,
        )
        for case in development
    ]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + 17)
    history = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(tensor_cases), generator=generator).tolist()
        loss_sum = 0.0
        for index in order:
            optimizer.zero_grad(set_to_none=True)
            case = apply_structured_dropout(
                tensor_cases[index],
                generator,
                args.full_dropout_probability,
                args.suffix_dropout_probability,
            )
            output = model(case)
            role_loss = F.cross_entropy(
                output["role_logits"],
                targets[index],
                weight=class_weights,
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
        history.append(
            {"epoch": epoch, "loss": round(loss_sum / len(development), 6)}
        )
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(json.dumps(history[-1]), flush=True)

    config_payload = {
        "protocol": "final_refit_train_plus_validation",
        "seed": args.seed,
        "epochs": args.epochs,
        "feature_config": feature_config.to_dict(),
        "model_config": asdict(model_config),
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "development_cases": len(development),
        "full_dropout_probability": args.full_dropout_probability,
        "suffix_dropout_probability": args.suffix_dropout_probability,
    }
    config_json = json.dumps(config_payload, sort_keys=True)
    config_hash = hashlib.sha256(config_json.encode()).hexdigest()
    checkpoint = {
        "model_state": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "reference": reference.state_dict(),
        "feature_config": feature_config.to_dict(),
        "model_config": asdict(model_config),
        "config_hash": config_hash,
        "seed": args.seed,
        "epochs": args.epochs,
        "protocol": "final_refit_train_plus_validation",
    }
    checkpoint_path = args.run_dir / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    summary = {
        **config_payload,
        "config_hash": config_hash,
        "checkpoint_sha256": checkpoint_hash,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    (args.run_dir / "config.json").write_text(
        json.dumps(config_payload | {"config_hash": config_hash}, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.run_dir / "history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    (args.run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
