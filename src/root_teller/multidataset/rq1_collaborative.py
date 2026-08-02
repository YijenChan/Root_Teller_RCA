from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from root_teller.module1 import data as data_module
from root_teller.module1 import evaluate as evaluate_module
from root_teller.module1 import model as model_module
from root_teller.module1.config import FeatureConfig
from root_teller.module1.data import ReferenceStats, to_torch_case
from root_teller.module1.evaluate import opaque_case_id
from root_teller.module1.model import ModelConfig, PerceptionRCA
from root_teller.module2 import run as run_module
from root_teller.module2 import window_export
from root_teller.paths import workspace_root

from . import rq1_sn, rq1_tt


PROJECT = workspace_root()
ROOT = PROJECT / "runs" / "rq1_multidataset_clean"


@dataclass
class Paths:
    dataset: str

    @property
    def project(self) -> Path:
        return ROOT / self.dataset / "module2_workspace"

    @property
    def api_config(self) -> Path:
        return PROJECT / "config" / "API_KEY.txt"

    @property
    def window_pack_root(self) -> Path:
        return self.project / "cache" / "module2_re2ob" / "window_evidence_packs"

    @property
    def response_cache(self) -> Path:
        return ROOT / "shared_llm_responses"

    @property
    def run_root(self) -> Path:
        return self.project / "runs" / "module2_re2ob"


def configure_services(services: tuple[str, ...]) -> None:
    index = {service: offset for offset, service in enumerate(services)}
    for module in (
        data_module,
        model_module,
        evaluate_module,
        window_export,
    ):
        module.SERVICE_INDEX = index


def load_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[PerceptionRCA, ReferenceStats, str]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    services = tuple(checkpoint["services"])
    configure_services(services)
    model = PerceptionRCA(
        metric_dim=10,
        log_dim=387,
        trace_dim=6,
        config=ModelConfig(**checkpoint["model_config"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    reference = ReferenceStats.from_state_dict(checkpoint["reference"])
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    return model, reference, digest


def export_cases(
    dataset: str,
    cases_with_checkpoints: list[tuple[dict[str, object], Path]],
    overwrite: bool = False,
) -> dict[str, object]:
    paths = Paths(dataset)
    split_root = paths.window_pack_root / "test"
    split_root.mkdir(parents=True, exist_ok=True)
    private_root = paths.project / "cache" / "module2_re2ob" / "private_evaluator"
    private_root.mkdir(parents=True, exist_ok=True)
    labels = {}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loaded: dict[Path, tuple[PerceptionRCA, ReferenceStats, str]] = {}
    exported = 0
    for raw_case, checkpoint_path in cases_with_checkpoints:
        if checkpoint_path not in loaded:
            loaded[checkpoint_path] = load_model(checkpoint_path, device)
        model, reference, checkpoint_hash = loaded[checkpoint_path]
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
            if destination.exists() and not overwrite:
                continue
            sliced = window_export._slice_case(raw_case, bin_index)
            tensor_case = to_torch_case(sliced, reference, device)
            with torch.no_grad():
                output = model(tensor_case)
            pack = window_export._pack(
                tensor_case,
                output,
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


def prepare_tt() -> dict[str, object]:
    services = rq1_tt.discover_services()
    rq1_tt.configure_services(services)
    config = FeatureConfig(log_backend="sbert")
    cases = rq1_tt.load_split("test", config)
    checkpoint = (
        rq1_tt.RUN_ROOT / "module1_refit_seed20260724" / "checkpoint.pt"
    )
    return export_cases("re2_tt", [(case, checkpoint) for case in cases])


def prepare_sn() -> dict[str, object]:
    candidate_services = rq1_sn.services()
    rq1_sn.configure(candidate_services)
    cases = rq1_sn.build_cases(False)
    pairs = []
    for case in cases:
        fold = int(str(case["incident_id"])[4:6])
        checkpoint = rq1_sn.RUN_ROOT / f"fold_{fold}" / "checkpoint.pt"
        pairs.append((case, checkpoint))
    return export_cases("eadro_sn", pairs)


def run_live(
    dataset: str,
    workers: int,
    case_id: str | None = None,
    run_id: str = "rq1_clean_default_exhaustive_live",
    resume: bool = False,
) -> dict[str, object]:
    paths = Paths(dataset)
    run_module.Module2Paths = lambda: paths
    return run_module.run(
        split="test",
        run_id=run_id,
        case_id=case_id,
        offline=False,
        workers=workers,
        protocol="default",
        resume=resume,
    )


def recover_failed(dataset: str) -> list[dict[str, object]]:
    paths = Paths(dataset)
    primary = (
        paths.run_root
        / "rq1_clean_default_exhaustive_live"
        / "summary.json"
    )
    failures = json.loads(primary.read_text(encoding="utf-8"))["failures"]
    recovered = []
    for index, failure in enumerate(failures):
        case_id = failure["incident_id"]
        summary = run_live(
            dataset,
            workers=1,
            case_id=case_id,
            run_id=f"rq1_clean_recovery_{index:02d}",
        )
        recovered.append(summary)
    return recovered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["re2_tt", "eadro_sn"], required=True)
    parser.add_argument(
        "--stage", choices=["export", "run", "recover", "all"], default="all"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--case-id")
    parser.add_argument("--run-id", default="rq1_clean_default_exhaustive_live")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.stage in {"export", "all"}:
        result = prepare_tt() if args.dataset == "re2_tt" else prepare_sn()
        print(json.dumps(result, indent=2), flush=True)
    if args.stage in {"run", "all"}:
        print(
            json.dumps(
                run_live(
                    args.dataset,
                    args.workers,
                    case_id=args.case_id,
                    run_id=args.run_id,
                    resume=args.resume,
                ),
                indent=2,
            ),
            flush=True,
        )
    if args.stage == "recover":
        print(json.dumps(recover_failed(args.dataset), indent=2), flush=True)


if __name__ == "__main__":
    main()
