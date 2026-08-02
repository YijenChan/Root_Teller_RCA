from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from root_teller.module1 import data as data_module
from root_teller.module1 import evaluate as evaluate_module
from root_teller.module1 import features as ob_features
from root_teller.module1 import model as model_module
from root_teller.module1.baseline import metrics
from root_teller.module1.config import FeatureConfig, Paths as OBPaths
from root_teller.module1.data import (
    ReferenceStats,
    apply_structured_dropout,
    fit_reference,
    normalize_case,
    to_torch_case,
    weak_role_targets,
)
from root_teller.module1.evaluate import opaque_case_id
from root_teller.module1.model import ModelConfig, PerceptionRCA
from root_teller.module1.train import role_class_weights, seed_everything
from root_teller.module2 import run as run_module
from root_teller.module2 import window_export
from root_teller.paths import workspace_root

from . import rq1_sn, rq1_tt


PROJECT = workspace_root()
RUN_ROOT = PROJECT / "runs" / "rq1_protocol_v2_seed42"
SEED = 42
MODEL_CONFIG = ModelConfig(
    hidden_dim=48,
    embedding_dim=48,
    graph_layers=2,
    dropout=0.2,
    fusion_lambda=0.6,
    fusion_mode="adaptive",
    use_availability_mask=True,
)


def configure_services(services: tuple[str, ...]) -> None:
    index = {service: offset for offset, service in enumerate(services)}
    for module in (
        data_module,
        model_module,
        evaluate_module,
        ob_features,
        window_export,
    ):
        module.SERVICE_INDEX = index


def load_ob() -> list[dict[str, object]]:
    paths = OBPaths()
    config = FeatureConfig(log_backend="sbert")
    cases = []
    for spec in ob_features.load_case_specs(paths):
        path = ob_features.cache_path(paths, spec, "CLEAN", config)
        if not path.exists():
            case = ob_features.extract_case(paths, spec, "CLEAN", config)
            ob_features.save_case(path, case)
        cases.append(ob_features.load_case(path))
    return cases


def load_tt() -> list[dict[str, object]]:
    services = rq1_tt.discover_services()
    rq1_tt.configure_services(services)
    config = FeatureConfig(log_backend="sbert")
    cases = []
    for spec in rq1_tt.specs():
        path = rq1_tt.feature_path(spec, config)
        if not path.exists():
            raise FileNotFoundError(path)
        cases.append(ob_features.load_case(path))
    return cases


def re2_fold_assignments(cases: list[dict[str, object]]) -> dict[str, int]:
    roots = sorted({str(case["root_cause_service"]) for case in cases})
    faults = sorted({str(case["fault_type"]) for case in cases})
    root_index = {value: index for index, value in enumerate(roots)}
    fault_index = {value: index for index, value in enumerate(faults)}
    return {
        str(case["incident_id"]): (
            root_index[str(case["root_cause_service"])]
            + fault_index[str(case["fault_type"])]
        )
        % 3
        for case in cases
    }


def re2_inner_split(
    development: list[dict[str, object]], outer_fold: int
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    roots = sorted({str(case["root_cause_service"]) for case in development})
    families = sorted(
        {
            (str(case["root_cause_service"]), str(case["fault_type"]))
            for case in development
        }
    )
    validation_families: set[tuple[str, str]] = set()
    for root_index, root in enumerate(roots):
        candidates = [family for family in families if family[0] == root]
        validation_families.add(
            candidates[(root_index + outer_fold) % len(candidates)]
        )
    validation = [
        case
        for case in development
        if (str(case["root_cause_service"]), str(case["fault_type"]))
        in validation_families
    ]
    train = [
        case
        for case in development
        if (str(case["root_cause_service"]), str(case["fault_type"]))
        not in validation_families
    ]
    return train, validation


@torch.no_grad()
def predict(
    model: PerceptionRCA,
    cases: list[dict[str, object]],
    reference: ReferenceStats,
    device: torch.device,
) -> list[dict[str, object]]:
    model.eval()
    predictions = []
    for case in cases:
        tensor = to_torch_case(case, reference, device)
        output = model(tensor)
        order = torch.argsort(
            output["localization_probabilities"], descending=True
        ).cpu().tolist()
        rank = order.index(int(tensor["target_index"])) + 1
        predictions.append(
            {
                "incident_id": str(case["incident_id"]),
                "target": str(case["root_cause_service"]),
                "rank": rank,
                "top5": [case["services"][index] for index in order[:5]],
            }
        )
    return predictions


def new_model(case: dict[str, object], device: torch.device) -> PerceptionRCA:
    return PerceptionRCA(
        metric_dim=int(case["metric_x"].shape[2]),
        log_dim=int(case["log_x"].shape[2]),
        trace_dim=int(case["trace_x"].shape[2]),
        config=MODEL_CONFIG,
    ).to(device)


def train_epochs(
    train_cases: list[dict[str, object]],
    reference: ReferenceStats,
    device: torch.device,
    epochs: int,
    seed: int,
    validation_cases: list[dict[str, object]] | None = None,
    patience: int = 50,
) -> tuple[PerceptionRCA, int, list[dict[str, object]]]:
    seed_everything(seed)
    model = new_model(train_cases[0], device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    weights = role_class_weights(train_cases, reference, device)
    tensor_cases = [to_torch_case(case, reference, device) for case in train_cases]
    role_targets = [
        torch.as_tensor(
            weak_role_targets(normalize_case(case, reference), reference),
            dtype=torch.long,
            device=device,
        )
        for case in train_cases
    ]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 17)
    history = []
    best_score = -1.0
    best_epoch = epochs
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for index in torch.randperm(len(tensor_cases), generator=generator).tolist():
            optimizer.zero_grad(set_to_none=True)
            augmented = apply_structured_dropout(
                tensor_cases[index], generator, 0.25, 0.25
            )
            output = model(augmented)
            role_loss = F.cross_entropy(
                output["role_logits"],
                role_targets[index],
                weight=weights,
                ignore_index=-100,
            )
            target = torch.as_tensor(
                [int(augmented["target_index"])], dtype=torch.long, device=device
            )
            loss = role_loss + F.cross_entropy(
                output["localization_logits"].unsqueeze(0), target
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += float(loss.item())
        if validation_cases is not None and (epoch == 1 or epoch % 5 == 0):
            validation_predictions = predict(
                model, validation_cases, reference, device
            )
            score = metrics(
                [int(item["rank"]) for item in validation_predictions]
            )["Avg@5"]
            row = {
                "epoch": epoch,
                "loss": round(epoch_loss / len(train_cases), 6),
                "validation_Avg@5": score,
            }
            history.append(row)
            print(json.dumps(row), flush=True)
            if score > best_score + 1e-9:
                best_score = score
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            if epoch - best_epoch >= patience:
                break
    if validation_cases is not None:
        assert best_state is not None
        model.load_state_dict(best_state)
    return model, best_epoch, history


def save_checkpoint(
    model: PerceptionRCA,
    reference: ReferenceStats,
    services: tuple[str, ...],
    selected_epoch: int,
    fold_dir: Path,
) -> Path:
    fold_dir.mkdir(parents=True, exist_ok=True)
    destination = fold_dir / "checkpoint.pt"
    torch.save(
        {
            "model_state": {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            },
            "reference": reference.state_dict(),
            "model_config": asdict(MODEL_CONFIG),
            "services": list(services),
            "selected_epoch": selected_epoch,
            "seed": SEED,
            "protocol": "rq1-protocol-v2",
        },
        destination,
    )
    return destination


def run_re2(
    dataset: str, cases: list[dict[str, object]], device: torch.device
) -> list[tuple[dict[str, object], Path]]:
    services = tuple(cases[0]["services"])
    configure_services(services)
    assignments = re2_fold_assignments(cases)
    manifest = []
    output: list[tuple[dict[str, object], Path]] = []
    all_predictions = []
    for outer_fold in range(3):
        test = [
            case
            for case in cases
            if assignments[str(case["incident_id"])] == outer_fold
        ]
        development = [
            case
            for case in cases
            if assignments[str(case["incident_id"])] != outer_fold
        ]
        inner_train, validation = re2_inner_split(development, outer_fold)
        selection_reference = fit_reference(inner_train)
        _, selected_epoch, history = train_epochs(
            inner_train,
            selection_reference,
            device,
            epochs=300,
            seed=SEED,
            validation_cases=validation,
            patience=50,
        )
        refit_reference = fit_reference(development)
        refit_model, _, _ = train_epochs(
            development,
            refit_reference,
            device,
            epochs=selected_epoch,
            seed=SEED,
        )
        fold_dir = RUN_ROOT / dataset / f"fold_{outer_fold}"
        checkpoint = save_checkpoint(
            refit_model, refit_reference, services, selected_epoch, fold_dir
        )
        fold_predictions = predict(refit_model, test, refit_reference, device)
        (fold_dir / "selection_history.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )
        (fold_dir / "module1_predictions.json").write_text(
            json.dumps(fold_predictions, indent=2) + "\n", encoding="utf-8"
        )
        all_predictions.extend(fold_predictions)
        output.extend((case, checkpoint) for case in test)
        for case in cases:
            family = [
                str(case["root_cause_service"]),
                str(case["fault_type"]),
            ]
            role = (
                "test"
                if case in test
                else ("validation" if case in validation else "train")
            )
            manifest.append(
                {
                    "outer_fold": outer_fold,
                    "incident_id": str(case["incident_id"]),
                    "family": family,
                    "role": role,
                }
            )
        fold_metrics = metrics(
            [int(item["rank"]) for item in fold_predictions]
        )
        print(
            json.dumps(
                {
                    "dataset": dataset,
                    "fold": outer_fold,
                    "selected_epoch": selected_epoch,
                    "module1_metrics": fold_metrics,
                }
            ),
            flush=True,
        )
    dataset_dir = RUN_ROOT / dataset
    (dataset_dir / "fold_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (dataset_dir / "module1_all_predictions.json").write_text(
        json.dumps(
            sorted(all_predictions, key=lambda item: item["incident_id"]),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return output


def run_sn(
    cases: list[dict[str, object]], device: torch.device
) -> list[tuple[dict[str, object], Path]]:
    dataset = "eadro_sn"
    services = tuple(cases[0]["services"])
    rq1_sn.configure(services)
    configure_services(services)
    reference = rq1_sn.fit_healthy_reference(services)
    output: list[tuple[dict[str, object], Path]] = []
    all_predictions = []
    manifest = []
    for outer_fold in range(4):
        validation_fold = (outer_fold + 1) % 4
        capture = lambda case: int(str(case["incident_id"])[4:6])
        test = [case for case in cases if capture(case) == outer_fold]
        validation = [case for case in cases if capture(case) == validation_fold]
        inner_train = [
            case
            for case in cases
            if capture(case) not in {outer_fold, validation_fold}
        ]
        development = [case for case in cases if capture(case) != outer_fold]
        _, selected_epoch, history = train_epochs(
            inner_train,
            reference,
            device,
            epochs=300,
            seed=SEED,
            validation_cases=validation,
            patience=50,
        )
        refit_model, _, _ = train_epochs(
            development, reference, device, selected_epoch, SEED
        )
        fold_dir = RUN_ROOT / dataset / f"fold_{outer_fold}"
        checkpoint = save_checkpoint(
            refit_model, reference, services, selected_epoch, fold_dir
        )
        fold_predictions = predict(refit_model, test, reference, device)
        (fold_dir / "selection_history.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )
        (fold_dir / "module1_predictions.json").write_text(
            json.dumps(fold_predictions, indent=2) + "\n", encoding="utf-8"
        )
        all_predictions.extend(fold_predictions)
        output.extend((case, checkpoint) for case in test)
        for case in cases:
            case_fold = capture(case)
            role = (
                "test"
                if case_fold == outer_fold
                else ("validation" if case_fold == validation_fold else "train")
            )
            manifest.append(
                {
                    "outer_fold": outer_fold,
                    "incident_id": str(case["incident_id"]),
                    "capture_id": str(case["capture_id"]),
                    "role": role,
                }
            )
        print(
            json.dumps(
                {
                    "dataset": dataset,
                    "fold": outer_fold,
                    "selected_epoch": selected_epoch,
                    "module1_metrics": metrics(
                        [int(item["rank"]) for item in fold_predictions]
                    ),
                }
            ),
            flush=True,
        )
    dataset_dir = RUN_ROOT / dataset
    (dataset_dir / "fold_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (dataset_dir / "module1_all_predictions.json").write_text(
        json.dumps(
            sorted(all_predictions, key=lambda item: item["incident_id"]),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return output


@dataclass
class Module2Paths:
    dataset: str

    @property
    def project(self) -> Path:
        return RUN_ROOT / self.dataset / "module2_workspace"

    @property
    def api_config(self) -> Path:
        return PROJECT / "config" / "API_KEY.txt"

    @property
    def window_pack_root(self) -> Path:
        return self.project / "cache" / "module2_re2ob" / "window_evidence_packs"

    @property
    def response_cache(self) -> Path:
        return RUN_ROOT / "shared_llm_responses"

    @property
    def run_root(self) -> Path:
        return self.project / "runs" / "module2_re2ob"


def load_checkpoint(
    checkpoint_path: Path, raw_case: dict[str, object], device: torch.device
) -> tuple[PerceptionRCA, ReferenceStats, str]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    configure_services(tuple(checkpoint["services"]))
    model = new_model(raw_case, device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    reference = ReferenceStats.from_state_dict(checkpoint["reference"])
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    return model, reference, digest


def export_cases(
    dataset: str,
    cases_with_checkpoints: list[tuple[dict[str, object], Path]],
) -> dict[str, object]:
    paths = Module2Paths(dataset)
    split_root = paths.window_pack_root / "test"
    split_root.mkdir(parents=True, exist_ok=True)
    private_root = paths.project / "cache" / "module2_re2ob" / "private_evaluator"
    private_root.mkdir(parents=True, exist_ok=True)
    labels = {}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loaded = {}
    exported = 0
    for raw_case, checkpoint_path in cases_with_checkpoints:
        key = str(checkpoint_path)
        if key not in loaded:
            loaded[key] = load_checkpoint(checkpoint_path, raw_case, device)
        model, reference, checkpoint_hash = loaded[key]
        configure_services(tuple(raw_case["services"]))
        case_id = opaque_case_id(str(raw_case["incident_id"]))
        labels[case_id] = {
            "root_cause_service": raw_case["root_cause_service"],
            "fault_type": raw_case["fault_type"],
            "raw_incident_id": raw_case["incident_id"],
        }
        case_root = split_root / case_id
        case_root.mkdir(parents=True, exist_ok=True)
        bins = int(raw_case["metric_x"].shape[1])
        for activation_order, bin_index in enumerate(range(bins - 1, -1, -1)):
            destination = case_root / f"W{activation_order:02d}.json"
            sliced = window_export._slice_case(raw_case, bin_index)
            tensor_case = to_torch_case(sliced, reference, device)
            with torch.no_grad():
                model_output = model(tensor_case)
            pack = window_export._pack(
                tensor_case,
                model_output,
                checkpoint_hash,
                f"W{activation_order:02d}",
                bin_index,
            )
            destination.write_text(
                json.dumps(pack, indent=2) + "\n", encoding="utf-8"
            )
            exported += 1
    (private_root / "test_labels.json").write_text(
        json.dumps(labels, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "dataset": dataset,
        "cases": len(cases_with_checkpoints),
        "windows_exported": exported,
        "label_fields_in_public_packs": False,
    }
    (split_root / "export_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def pairs_from_artifacts(dataset: str) -> list[tuple[dict[str, object], Path]]:
    if dataset == "re2_ob":
        cases = load_ob()
        assignments = re2_fold_assignments(cases)
        return [
            (
                case,
                RUN_ROOT
                / dataset
                / f"fold_{assignments[str(case['incident_id'])]}"
                / "checkpoint.pt",
            )
            for case in cases
        ]
    if dataset == "re2_tt":
        cases = load_tt()
        assignments = re2_fold_assignments(cases)
        return [
            (
                case,
                RUN_ROOT
                / dataset
                / f"fold_{assignments[str(case['incident_id'])]}"
                / "checkpoint.pt",
            )
            for case in cases
        ]
    cases = rq1_sn.build_cases(False)
    return [
        (
            case,
            RUN_ROOT
            / dataset
            / f"fold_{int(str(case['incident_id'])[4:6])}"
            / "checkpoint.pt",
        )
        for case in cases
    ]


def run_module2(dataset: str, workers: int, resume: bool) -> dict[str, object]:
    paths = Module2Paths(dataset)
    run_module.Module2Paths = lambda: paths
    return run_module.run(
        split="test",
        run_id=f"rq1_protocol_v2_seed{SEED}",
        offline=False,
        workers=workers,
        protocol="default",
        resume=resume,
    )


def verify(dataset: str) -> dict[str, object]:
    paths = Module2Paths(dataset)
    run_dir = paths.run_root / f"rq1_protocol_v2_seed{SEED}"
    evaluated = json.loads(
        (run_dir / "evaluation_private.json").read_text(encoding="utf-8")
    )
    ranks = [int(row["default_exhaustive_rank"]) for row in evaluated]
    recomputed = metrics(ranks)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if recomputed != summary["default_exhaustive_metrics"]:
        raise RuntimeError("independent metric recomputation disagrees with summary")
    result = {
        "dataset": dataset,
        "seed": SEED,
        "cases": len(ranks),
        "A@1": recomputed["A@1"],
        "A@3": recomputed["A@3"],
        "Avg@5": recomputed["Avg@5"],
        "evaluation_sha256": hashlib.sha256(
            (run_dir / "evaluation_private.json").read_bytes()
        ).hexdigest(),
        "checks": [
            "group-disjoint outer evaluation",
            "each incident evaluated exactly once",
            "duplicate-free ordered service ranking",
            "independent metric recomputation matched",
        ],
    }
    (RUN_ROOT / dataset / "verified_rq1_result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    global RUN_ROOT, SEED
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=["re2_ob", "re2_tt", "eadro_sn"], required=True
    )
    parser.add_argument(
        "--stage", choices=["train", "export", "module2", "verify", "all"], default="all"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, choices=[41, 42, 43], default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    SEED = args.seed
    RUN_ROOT = PROJECT / "runs" / f"rq1_protocol_v2_seed{SEED}"
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print(
        json.dumps(
            {
                "experiment": "Root-Teller RQ1 Protocol V2",
                "dataset": args.dataset,
                "seed": SEED,
                "selection": "predeclared; no seed or result cherry-picking",
                "device": "cuda:0" if torch.cuda.is_available() else "cpu",
            }
        ),
        flush=True,
    )
    if args.stage in {"train", "all"}:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if args.dataset == "re2_ob":
            run_re2(args.dataset, load_ob(), device)
        elif args.dataset == "re2_tt":
            run_re2(args.dataset, load_tt(), device)
        else:
            services = rq1_sn.services()
            rq1_sn.configure(services)
            run_sn(rq1_sn.build_cases(False), device)
    if args.stage in {"export", "all"}:
        print(json.dumps(export_cases(args.dataset, pairs_from_artifacts(args.dataset)), indent=2))
    if args.stage in {"module2", "all"}:
        run_module2(args.dataset, args.workers, args.resume)
    if args.stage in {"verify", "all"}:
        verify(args.dataset)
    print(
        json.dumps({"elapsed_seconds": round(time.time() - started, 3)}),
        flush=True,
    )


if __name__ == "__main__":
    main()
