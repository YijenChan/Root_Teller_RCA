"""Common-protocol baseline evaluation on the clean Eadro-SN dataset.

The 36 annotated intervals are evaluated with grouped four-fold
leave-one-capture-out, matching Root-Teller's Eadro-SN protocol.  Fault
annotations define interval boundaries and evaluator labels only; no fault
name or fault type is exposed to any ranker.
"""
from __future__ import annotations

import os

import argparse
import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


PROJECT = Path(os.environ.get("ROOTTELLER_WORKSPACE", Path.cwd())).expanduser().resolve()
sys.path.insert(0, str(PROJECT / "src"))

from root_teller.module1.data import normalize_case  # noqa: E402
from root_teller.multidataset import rq1_sn  # noqa: E402


SERVICES = ()
SERVICE_INDEX = {}
RUN_ROOT = PROJECT / "baselines"
RUN_ID = "eadro_sn_clean_grouped_loco_2026-07-25"


def order_from(scores: np.ndarray, incident_id: str = "") -> list[str]:
    tie_key = {
        service: hashlib.sha256(
            f"{incident_id}\0{service}".encode("utf-8")
        ).hexdigest()
        for service in SERVICES
    }
    return [
        SERVICES[index]
        for index in sorted(
            range(len(SERVICES)),
            key=lambda i: (-float(scores[i]), tie_key[SERVICES[i]]),
        )
    ]


def metrics(predictions: list[dict]) -> dict[str, float | int]:
    hits = np.zeros(5, dtype=np.float64)
    for item in predictions:
        hits += [item["rank"] <= cutoff for cutoff in range(1, 6)]
    hits /= max(len(predictions), 1)
    return {
        "A@1": float(hits[0]),
        "A@5": float(hits[4]),
        "Avg@5": float(hits.mean()),
        "cases": len(predictions),
    }


def save(method: str, predictions: list[dict], extra: dict | None = None) -> None:
    destination = (
        RUN_ROOT / method / "runs" / "frozen_test" /
        RUN_ID
    )
    destination.mkdir(parents=True, exist_ok=True)
    config = {
        "dataset": "Eadro-SN",
        "protocol": "grouped-4-fold-leave-one-capture-out",
        "cases": 36,
        "candidate_services": 12,
        "window_seconds": 60,
        "healthy_reference": "three released no-fault captures",
        "uses_fault_label_at_inference": False,
        "uses_fault_type_at_inference": False,
    }
    if extra:
        config.update(extra)
    (destination / "summary.json").write_text(
        json.dumps({"metrics": metrics(predictions), "config": config}, indent=2) + "\n",
        encoding="utf-8",
    )
    (destination / "predictions_private.json").write_text(
        json.dumps(predictions, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics(predictions), indent=2), flush=True)


def load_eadro_module():
    path = Path(__file__).resolve().with_name("eadro.py")
    spec = importlib.util.spec_from_file_location("eadro_sn_compat", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.SERVICES = SERVICES
    module.SERVICE_INDEX = SERVICE_INDEX
    return module


def eadro_tensor(case: dict, reference) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    normalized = normalize_case(case, reference)
    metric = np.asarray(normalized["metric_x"], np.float32)[:, :, :7]
    trace_raw = np.asarray(normalized["trace_x"], np.float32)
    trace = np.mean(trace_raw, axis=2, keepdims=True)
    repeats = int(np.ceil(60 / metric.shape[1]))
    metric = np.repeat(metric, repeats, axis=1)[:, :60]
    trace = np.repeat(trace, repeats, axis=1)[:, :60]
    logs = np.asarray(normalized["log_x"], np.float32).mean(axis=1)
    logs = logs[:, :256]
    return metric, trace, logs, SERVICE_INDEX[case["root_cause_service"]]


def run_eadro(cases: list[dict], reference, epochs: int) -> None:
    module = load_eadro_module()
    predictions = []
    fold_summaries = []
    for fold in range(4):
        random.seed(42 + fold)
        np.random.seed(42 + fold)
        torch.manual_seed(42 + fold)
        torch.cuda.manual_seed_all(42 + fold)
        train_cases = [
            case for case in cases if case["incident_id"][:6] != f"sn-c{fold:02d}"
        ]
        test_cases = [
            case for case in cases if case["incident_id"][:6] == f"sn-c{fold:02d}"
        ]
        edges = set()
        for case in train_cases:
            for source, target in case.get("edges", ()):
                if source in SERVICE_INDEX and target in SERVICE_INDEX:
                    edges.add((SERVICE_INDEX[source], SERVICE_INDEX[target]))
        train_tensors = [eadro_tensor(case, reference) for case in train_cases]
        test_tensors = [eadro_tensor(case, reference) for case in test_cases]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = module.EadroCompat(256, sorted(edges)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        metric = torch.tensor(np.stack([item[0] for item in train_tensors]), device=device)
        trace = torch.tensor(np.stack([item[1] for item in train_tensors]), device=device)
        logs = torch.tensor(np.stack([item[2] for item in train_tensors]), device=device)
        labels = torch.tensor([item[3] for item in train_tensors], dtype=torch.long, device=device)
        for _ in range(epochs):
            model.train()
            detector, locator = model(metric, trace, logs)
            detection_labels = torch.ones(len(labels), dtype=torch.long, device=device)
            loss = 0.5 * F.cross_entropy(detector, detection_labels) + 0.5 * F.cross_entropy(locator, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        fold_predictions = []
        with torch.no_grad():
            for case, tensors in zip(test_cases, test_tensors):
                metric_x = torch.tensor(tensors[0][None], device=device)
                trace_x = torch.tensor(tensors[1][None], device=device)
                log_x = torch.tensor(tensors[2][None], device=device)
                _, locator = model(metric_x, trace_x, log_x)
                ranking = order_from(locator[0].detach().cpu().numpy())
                position = ranking.index(case["root_cause_service"]) + 1
                item = {
                    "incident_id": case["incident_id"],
                    "capture_id": case["capture_id"],
                    "ground_truth_service": case["root_cause_service"],
                    "ranking": ranking[:5],
                    "rank": position,
                    "fold": fold,
                }
                fold_predictions.append(item)
                predictions.append(item)
        fold_summaries.append({"fold": fold, "metrics": metrics(fold_predictions)})
        print(json.dumps(fold_summaries[-1]), flush=True)
    save(
        "eadro",
        predictions,
        {
            "variant": "Eadro TCN-GAT-Hawkes architecture compatibility port",
            "epochs": epochs,
            "folds": fold_summaries,
            "grouped_split_consequence": "test-fold culprit services are absent from supervised training folds",
        },
    )


def main() -> None:
    global RUN_ID, SERVICES, SERVICE_INDEX
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "method",
        choices=("eadro", "nezha", "multisource_rcd", "torai", "thinkfl", "rclagent"),
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--run-id", default=RUN_ID)
    args, remaining = parser.parse_known_args()
    if args.method in {"nezha", "multisource_rcd", "torai"}:
        from root_teller.baselines.classical_run import main as run
        sys.argv = [sys.argv[0], "--method", args.method, "--dataset", "eadro_sn", *remaining]
        run()
        return
    if args.method in {"thinkfl", "rclagent"}:
        from root_teller.baselines.run import main as run
        sys.argv = [sys.argv[0], "--method", args.method, "--dataset", "eadro_sn", *remaining]
        run()
        return
    if remaining:
        parser.error("Unrecognized arguments: " + " ".join(remaining))
    SERVICES = rq1_sn.services()
    SERVICE_INDEX = {service: index for index, service in enumerate(SERVICES)}
    RUN_ID = args.run_id
    rq1_sn.configure(SERVICES)
    cases = rq1_sn.build_cases(False)
    reference = rq1_sn.fit_healthy_reference(SERVICES)
    run_eadro(cases, reference, args.epochs)


if __name__ == "__main__":
    main()
