"""Faithful RCAEval Multi-source RCD rerun with a label-free midpoint anchor."""
from __future__ import annotations

import os

import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib.util
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

if not hasattr(np, "mat"):
    np.mat = np.asmatrix

PROJECT = Path(os.environ.get("ROOTTELLER_WORKSPACE", Path.cwd())).expanduser().resolve()
UPSTREAM = PROJECT / "baselines" / "torai" / "upstream" / "RCAEval-main"
MANIFEST = Path(os.environ.get("ROOTTELLER_ACTIVE_SPLIT_MANIFEST", PROJECT / "evaluation" / "rq1" / "manifests" / "active_split_manifest.csv"))
DATASETS = {
    "re2ob": (
        "RCAEval RE2-OB",
        PROJECT / "dataset" / "RCAEval RE" / "RE2" / "RE2-OB" / "RE2-OB",
    ),
    "re2tt": (
        "RCAEval RE2-TT",
        PROJECT / "dataset" / "RCAEval RE" / "RE2" / "RE2-TT" / "RE2-TT",
    ),
}


def patch_causallearn() -> None:
    from causallearn.utils.PCUtils import SkeletonDiscovery

    if hasattr(SkeletonDiscovery, "local_skeleton_discovery"):
        return
    source = UPSTREAM / "lib" / "causallearn" / "utils" / "PCUtils" / "SkeletonDiscovery.py"
    spec = importlib.util.spec_from_file_location("mmrcd_skeleton_patch", source)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    graph_source = UPSTREAM / "lib" / "causallearn" / "graph" / "GraphClass.py"
    graph_spec = importlib.util.spec_from_file_location("mmrcd_graph_patch", graph_source)
    graph_module = importlib.util.module_from_spec(graph_spec)
    assert graph_spec and graph_spec.loader
    graph_spec.loader.exec_module(graph_module)
    module.CausalGraph = graph_module.CausalGraph
    SkeletonDiscovery.local_skeleton_discovery = module.local_skeleton_discovery


patch_causallearn()
sys.path.insert(0, str(UPSTREAM))
e2e_package = types.ModuleType("RCAEval.e2e")
e2e_package.__path__ = [str(UPSTREAM / "RCAEval" / "e2e")]
sys.modules.setdefault("RCAEval.e2e", e2e_package)
from RCAEval.e2e.mmrcd import mmrcd  # noqa: E402


def canonical_service(value: object) -> str:
    value = str(value).strip().lower().replace("_", "-")
    if value.startswith("ts-") and value.endswith(("-mongo", "-mysql")):
        owner = value.rsplit("-", 1)[0]
        value = owner if owner.endswith("-service") else owner + "-service"
    aliases = {
        "frontendservice": "frontend",
        "frontend-external": "frontend",
        "redis-cart": "redis",
    }
    return aliases.get(value, value)


@dataclass(frozen=True)
class Record:
    incident_id: str
    root: str
    directory: Path


def records(dataset: str, split: str) -> list[Record]:
    system, raw = DATASETS[dataset]
    frame = pd.read_csv(MANIFEST)
    frame = frame[(frame.dataset_system == system) & (frame.split == split)]
    if split == "test":
        frame = frame[frame.eligible]
    return [
        Record(
            str(row.incident_id),
            canonical_service(row.root_cause_service),
            raw / Path(str(row.incident_id)),
        )
        for row in frame.itertuples(index=False)
    ]


def service_from_indicator(indicator: object) -> str:
    text = str(indicator).strip().lower()
    return canonical_service(text.rsplit("_", 1)[0])


def input_for(
    record: Record, dataset: str, length_minutes: int
) -> tuple[dict[str, pd.DataFrame], float]:
    metric = pd.read_csv(record.directory / "simple_metrics.csv", low_memory=False)
    metric = metric.loc[:, ~metric.columns.str.endswith("_latency-50")]
    if dataset == "re2tt":
        time_col = metric["time"]
        metric = metric.loc[:, metric.columns.str.startswith("ts-")]
        metric["time"] = time_col
    metric = metric.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
    metric["time"] = pd.to_numeric(metric["time"], errors="coerce")
    metric = metric.dropna(subset=["time"]).sort_values("time")
    boundary = (float(metric.time.min()) + float(metric.time.max())) / 2.0

    # Match RCAEval main.py: each side receives length * 60 / 2 samples.
    # Multi-source RCD uses main.py's native default --length 20.
    half_samples = length_minutes * 60 // 2
    before = metric[metric.time < boundary].tail(half_samples)
    after = metric[metric.time >= boundary].head(half_samples)
    metric = pd.concat([before, after], ignore_index=True)
    empty = pd.DataFrame()
    return {
        "metric": metric,
        "logts": empty,
        "tracets_err": empty,
        "tracets_lat": empty,
    }, boundary


def diagnose(payload: tuple[Record, str, int]) -> dict:
    record, dataset, length_minutes = payload
    data, boundary = input_for(record, dataset, length_minutes)
    result = mmrcd(
        data,
        inject_time=boundary,
        dataset=dataset,
        gamma=5,
        localized=True,
        bins=5,
        seed=0,
    )
    ordered = []
    for indicator in result["ranks"]:
        service = service_from_indicator(indicator)
        if service and service not in ordered:
            ordered.append(service)
    # The native method may return fewer than five candidates. Absence is a
    # miss at every reported cutoff, not the next numerical rank.
    position = ordered.index(record.root) + 1 if record.root in ordered else 1_000_000
    return {
        "incident_id": record.incident_id,
        "ground_truth_service": record.root,
        "ranking": ordered[:5],
        "rank": position,
        "raw_indicator_ranking": [str(value) for value in result["ranks"]],
        "boundary": boundary,
    }


def run(args: argparse.Namespace) -> None:
    rows = records(args.dataset, args.split)
    hits = np.zeros(5)
    predictions = []
    payloads = [(record, args.dataset, args.length) for record in rows]
    if args.workers == 1:
        iterator = map(diagnose, payloads)
    else:
        executor = ProcessPoolExecutor(max_workers=args.workers)
        iterator = executor.map(diagnose, payloads)
    try:
        for index, item in enumerate(iterator, 1):
            position = item["rank"]
            hits += [position <= k for k in range(1, 6)]
            predictions.append(item)
            print(
                f"[{index}/{len(rows)}] {item['incident_id']}: "
                f"rank={position} top5={item['ranking']}",
                flush=True,
            )
    finally:
        if args.workers != 1:
            executor.shutdown()

    metrics = {
        "A@1": float(hits[0] / len(rows)),
        "A@5": float(hits[4] / len(rows)),
        "Avg@5": float(hits.mean() / len(rows)),
        "cases": len(rows),
    }
    config = {
        "variant": "RCAEval official mmrcd",
        "upstream_function": "RCAEval.e2e.mmrcd.mmrcd",
        "official_parameters": {"gamma": 5, "localized": True, "bins": 5, "seed": 0},
        "length_minutes": args.length,
        "reference_policy": "predefined observation-range midpoint",
        "uses_injection_time_at_inference": False,
        "uses_labels_at_inference": False,
        "workers": args.workers,
        "telemetry_note": "The released mmrcd function currently ranks from metrics; its log/trace inputs are unused.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps({"metrics": metrics, "config": config}, indent=2),
        encoding="utf-8",
    )
    (args.output / "predictions_private.json").write_text(
        json.dumps(predictions, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--length", type=int, default=20)
    run(parser.parse_args())
