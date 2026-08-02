"""Run the official TORAI artifact on RE2-OB with a label-free time split."""
from __future__ import annotations

import os

import argparse
import importlib.util
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# RCAEval's package initializer imports legacy causal-learn modules that still
# reference np.mat.  This compatibility alias is limited to this runner.
if not hasattr(np, "mat"):
    np.mat = np.asmatrix

PROJECT = Path(os.environ.get("ROOTTELLER_WORKSPACE", Path.cwd())).expanduser().resolve()
ROOT = PROJECT / "baselines" / "torai" / "upstream" / "RCAEval-main"
RAW = PROJECT / "dataset" / "RCAEval RE" / "RE2" / "RE2-OB" / "RE2-OB"
MANIFEST = Path(os.environ.get("ROOTTELLER_ACTIVE_SPLIT_MANIFEST", PROJECT / "evaluation" / "rq1" / "manifests" / "active_split_manifest.csv"))
SERVICES = ("adservice", "cartservice", "checkoutservice", "currencyservice", "emailservice", "frontend", "paymentservice", "productcatalogservice", "recommendationservice", "redis", "shippingservice")


def patch_causallearn() -> None:
    """Use the localized-PC helper shipped in the official TORAI artifact."""
    from causallearn.utils.PCUtils import SkeletonDiscovery
    if hasattr(SkeletonDiscovery, "local_skeleton_discovery"):
        return
    source = ROOT / "lib" / "causallearn" / "utils" / "PCUtils" / "SkeletonDiscovery.py"
    spec = importlib.util.spec_from_file_location("torai_skeleton_patch", source)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    graph_source = ROOT / "lib" / "causallearn" / "graph" / "GraphClass.py"
    graph_spec = importlib.util.spec_from_file_location("torai_graph_patch", graph_source)
    graph_module = importlib.util.module_from_spec(graph_spec)
    assert graph_spec and graph_spec.loader
    graph_spec.loader.exec_module(graph_module)
    module.CausalGraph = graph_module.CausalGraph
    SkeletonDiscovery.local_skeleton_discovery = module.local_skeleton_discovery


patch_causallearn()
sys.path.insert(0, str(ROOT))
# The official e2e package initializer eagerly imports every optional baseline.
# Create a namespace package so importing TORAI itself does not pull unrelated
# methods (and their optional dependencies) into this isolated runner.
e2e_package = types.ModuleType("RCAEval.e2e")
e2e_package.__path__ = [str(ROOT / "RCAEval" / "e2e")]
sys.modules.setdefault("RCAEval.e2e", e2e_package)
from RCAEval.e2e.torai import torai  # noqa: E402


def canonical_service(value: object) -> str:
    return {"frontendservice": "frontend", "frontend-external": "frontend", "redis-cart": "redis"}.get(str(value).strip().lower(), str(value).strip().lower())


@dataclass(frozen=True)
class Record:
    incident_id: str
    root: str

    @property
    def directory(self) -> Path:
        return RAW / Path(self.incident_id)


def records_for(split: str) -> list[Record]:
    f = pd.read_csv(MANIFEST)
    f = f.loc[(f.dataset_system == "RCAEval RE2-OB") & (f.split == split)]
    if split == "test":
        f = f.loc[f.eligible]
    return [Record(str(row.incident_id), canonical_service(row.root_cause_service)) for row in f.itertuples(index=False)]


def data_for(record: Record) -> tuple[dict[str, pd.DataFrame], float]:
    files = {"metric": "simple_metrics.csv", "logts": "logts.csv", "tracets_err": "tracets_err.csv", "tracets_lat": "tracets_lat.csv"}
    data = {key: pd.read_csv(record.directory / file, low_memory=False) for key, file in files.items()}
    for frame in data.values():
        frame["time"] = pd.to_numeric(frame["time"], errors="coerce")
        frame.dropna(subset=["time"], inplace=True)
    # The official artifact requires a normal/anomalous boundary.  RCAEval cases
    # contain a fixed pre/post observation range, so use its temporal midpoint.
    # This is independent of inject_time.txt, labels, service, and fault metadata.
    metric_times = data["metric"]["time"]
    boundary = (float(metric_times.min()) + float(metric_times.max())) / 2.0
    if int((data["metric"]["time"] < boundary).sum()) < 15 or int((data["metric"]["time"] >= boundary).sum()) < 15:
        raise ValueError(f"insufficient fixed reference window: {record.incident_id}")
    return data, boundary


def rank(record: Record) -> tuple[list[str], dict]:
    data, boundary = data_for(record)
    result = torai(data, inject_time=boundary, dataset="online-boutique")
    raw = [canonical_service(str(item).rsplit("_", 1)[0]) for item in result["ranks"]]
    ordered = []
    for service in raw + list(SERVICES):
        if service in SERVICES and service not in ordered:
            ordered.append(service)
    return ordered, {"fixed_boundary": boundary, "raw_ranks": result["ranks"][:20]}


def run(rows: list[Record], output: Path) -> None:
    predictions, hit = [], np.zeros(5)
    for i, record in enumerate(rows, 1):
        ranking, diag = rank(record)
        position = ranking.index(record.root) + 1
        hit += [position <= k for k in range(1, 6)]
        predictions.append({"incident_id": record.incident_id, "ground_truth_service": record.root, "ranking": ranking[:5], "rank": position, "diagnostics": diag})
        print(f"[{i}/{len(rows)}] {record.incident_id}: rank={position} top5={ranking[:5]}", flush=True)
    metrics = {"A@1": float(hit[0] / len(rows)), "A@5": float(hit[4] / len(rows)), "Avg@5": float(hit.mean() / len(rows)), "cases": len(rows)}
    config = {"variant": "TORAI-RE2 compatibility variant", "official_artifact": "Figshare 31938495", "window_seconds": 60, "reference_policy": "predefined observation-range midpoint", "production_policy": "second half of the same observation range", "uses_injection_time_at_inference": False, "uses_labels_at_inference": False, "official_defaults": {"gamma": 5, "bins": 5, "localized": True}}
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps({"metrics": metrics, "config": config}, indent=2), encoding="utf-8")
    (output / "predictions_private.json").write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=("validation", "test"), required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    run(records_for(args.split), args.output)
