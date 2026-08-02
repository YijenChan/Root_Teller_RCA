from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path()
TRACE_SHIFT = 8 * 60 * 60
LOG_TIME = re.compile(
    r"\[(\d{4}-[A-Za-z]{3}-\d{2} \d\d:\d\d:\d\d(?:\.\d+)?)\]"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_log_time(line: str) -> float | None:
    match = LOG_TIME.search(line)
    if not match:
        return None
    for fmt in ("%Y-%b-%d %H:%M:%S.%f", "%Y-%b-%d %H:%M:%S"):
        try:
            return datetime.strptime(match.group(1), fmt).timestamp()
        except ValueError:
            pass
    return None


def read_csv_rows(path: Path) -> int:
    return sum(len(chunk) for chunk in pd.read_csv(path, chunksize=250_000))


def check_rca_metric(
    view: Path, condition: str, onset: float, errors: list[str]
) -> None:
    archive = np.load(view / "metric_availability_mask.npz", allow_pickle=False)
    mask = archive["mask"].astype(bool)
    times = archive["time"].astype(float)
    if condition == "GMO_METRIC" and mask.any():
        errors.append(f"GMO metric retains values: {view}")
    if condition == "IAMI_METRIC":
        if mask[times >= onset].any():
            errors.append(f"IAMI metric retains post-onset values: {view}")
        if not np.any(times >= onset):
            errors.append(f"IAMI metric has no post-onset rows to mask: {view}")


def check_sn_metric(
    view: Path, condition: str, onset: float, errors: list[str]
) -> None:
    archives = sorted((view / "metric_masks").glob("*.npz"))
    if not archives:
        errors.append(f"missing Eadro metric masks: {view}")
        return
    saw_post = False
    for path in archives:
        archive = np.load(path, allow_pickle=False)
        mask = archive["mask"].astype(bool)
        times = archive["timestamp"].astype(float)
        if condition == "GMO_METRIC" and mask.any():
            errors.append(f"GMO Eadro metric retains values: {path}")
        if condition == "IAMI_METRIC":
            saw_post |= bool(np.any(times >= onset))
            if mask[times >= onset].any():
                errors.append(f"IAMI Eadro metric retains post-onset values: {path}")
    if condition == "IAMI_METRIC" and not saw_post:
        errors.append(f"IAMI Eadro metric has no post-onset rows: {view}")


def check_rca_log(
    view: Path, condition: str, onset: float, errors: list[str]
) -> None:
    path = view / "logs.csv"
    if condition == "GMO_LOG":
        if read_csv_rows(path) != 0:
            errors.append(f"GMO log is not empty: {view}")
        return
    for chunk in pd.read_csv(path, usecols=["timestamp"], chunksize=250_000):
        times = pd.to_numeric(chunk["timestamp"], errors="coerce")
        if (times >= onset * 1e9).any():
            errors.append(f"IAMI log retains post-onset rows: {view}")
            return


def check_sn_log(
    view: Path, condition: str, onset: float, errors: list[str]
) -> None:
    payload = json.loads((view / "logs.json").read_text(encoding="utf-8"))
    lines = [str(line) for values in payload.values() for line in values]
    if condition == "GMO_LOG" and lines:
        errors.append(f"GMO Eadro log is not empty: {view}")
    if condition == "IAMI_LOG":
        for line in lines:
            timestamp = parse_log_time(line)
            if timestamp is None or timestamp >= onset:
                errors.append(f"IAMI Eadro log boundary violation: {view}")
                return


def check_rca_trace(
    view: Path, condition: str, onset: float, errors: list[str]
) -> None:
    path = view / "traces.csv"
    if condition == "GMO_TRACE":
        if read_csv_rows(path) != 0:
            errors.append(f"GMO trace is not empty: {view}")
        return
    for chunk in pd.read_csv(path, usecols=["startTimeMillis"], chunksize=250_000):
        times = pd.to_numeric(chunk["startTimeMillis"], errors="coerce")
        if (times >= onset * 1e3).any():
            errors.append(f"IAMI trace retains post-onset spans: {view}")
            return


def check_sn_trace(
    view: Path, condition: str, onset: float, errors: list[str]
) -> None:
    traces = json.loads((view / "spans.json").read_text(encoding="utf-8"))
    spans = [span for trace in traces for span in trace.get("spans", [])]
    if condition == "GMO_TRACE" and spans:
        errors.append(f"GMO Eadro trace is not empty: {view}")
    if condition == "IAMI_TRACE":
        for span in spans:
            timestamp = float(span.get("startTime", 0)) / 1e6 - TRACE_SHIFT
            if timestamp >= onset:
                errors.append(f"IAMI Eadro trace boundary violation: {view}")
                return


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate materialized GMO/IAMI views against their semantics."
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Output directory produced by prepare.py",
    )
    args = parser.parse_args()
    global ROOT
    ROOT = args.root.expanduser().resolve()
    if not ROOT.exists():
        parser.error(f"output directory does not exist: {ROOT}")

    with (ROOT / "_manifests" / "private_manifest.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        private_rows = list(csv.DictReader(handle))
    private = {
        (row["dataset"], row["opaque_id"]): row for row in private_rows
    }
    with (ROOT / "_manifests" / "condition_manifest.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        conditions = list(csv.DictReader(handle))
    errors: list[str] = []
    checked: dict[str, int] = defaultdict(int)
    if len(private_rows) != 216:
        errors.append(f"private manifest rows={len(private_rows)}")
    if len(conditions) != 1296:
        errors.append(f"condition manifest rows={len(conditions)}")
    topology_hashes = {
        (dataset_dir.name.replace("_", " "), int(path.stem.split("_")[1])): sha256(path)
        for dataset_dir in (ROOT / "_manifests" / "topology").iterdir()
        for path in dataset_dir.glob("fold_*.csv")
    }
    # Restore exact dataset names after the filesystem-safe substitution.
    topology_hashes = {
        (
            {
                "RCAEval RE2-OB": "RCAEval RE2-OB",
                "RCAEval RE2-TT": "RCAEval RE2-TT",
                "Eadro-SN": "Eadro-SN",
            }.get(dataset, dataset),
            fold,
        ): value
        for (dataset, fold), value in topology_hashes.items()
    }
    for index, row in enumerate(conditions, 1):
        dataset = row["dataset"]
        opaque = row["opaque_incident_id"]
        condition = row["condition"]
        key = (dataset, opaque)
        if key not in private:
            errors.append(f"missing private mapping: {key}")
            continue
        record = private[key]
        onset = float(record["inject_time"])
        view = ROOT / row["view_relative_path"]
        if condition == "GMO_TRACE":
            expected = topology_hashes[(dataset, int(row["outer_fold"]))]
            actual = sha256(view / "topology_override.csv")
            if actual != expected or actual != row["topology_sha256"]:
                errors.append(f"GMO topology hash mismatch: {view}")
        if condition.endswith("_METRIC"):
            if dataset.startswith("RCAEval"):
                check_rca_metric(view, condition, onset, errors)
            else:
                check_sn_metric(view, condition, onset, errors)
        elif condition.endswith("_LOG"):
            if dataset.startswith("RCAEval"):
                check_rca_log(view, condition, onset, errors)
            else:
                check_sn_log(view, condition, onset, errors)
        else:
            if dataset.startswith("RCAEval"):
                check_rca_trace(view, condition, onset, errors)
            else:
                check_sn_trace(view, condition, onset, errors)
        checked[f"{dataset}/{condition}"] += 1
        if index % 100 == 0:
            print(json.dumps({"checked": index, "total": len(conditions)}), flush=True)
    report = {
        "status": "PASS" if not errors else "FAIL",
        "private_incidents": len(private_rows),
        "condition_views": len(conditions),
        "checked": dict(sorted(checked.items())),
        "errors": errors,
    }
    destination = ROOT / "_manifests" / "semantic_validation_report.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
