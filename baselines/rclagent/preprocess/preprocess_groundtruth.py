"""
preprocess_groundtruth.py
=========================
For every error trace in ``error_traces.txt``, find the nearest
ground-truth row in ``groundtruth.csv`` (by timestamp) and write a
per-trace label file:

  <data_root>/labels/trace_<i>_label.json   {"level": "...", "cmdb_id": "..."}

This enables the evaluator to look up the label without re-reading the
full groundtruth CSV for every prediction. Both Unix-second and ISO-8601
timestamps are accepted.

Usage
-----
  python preprocess/preprocess_groundtruth.py [data_root]
"""

import json
import os
import sys
from datetime import datetime

import pandas as pd


def _parse_unix(ts_str: str) -> datetime:
    """Convert a Unix timestamp string (possibly millisecond precision) to datetime."""
    if "T" in ts_str:
        return datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%SZ")
    ts_str = ts_str[:10]   # truncate to seconds
    return datetime.utcfromtimestamp(int(ts_str))


def find_nearest(timestamp_str: str, groundtruth_df: pd.DataFrame) -> pd.Series:
    dt = _parse_unix(timestamp_str)
    diffs = abs(groundtruth_df["timestamp"] - dt)
    return groundtruth_df.loc[diffs.idxmin()]


def build_labels(data_root: str) -> None:
    gt_path    = os.path.join(data_root, "groundtruth.csv")
    error_path = os.path.join(data_root, "error_traces.txt")
    out_dir    = os.path.join(data_root, "labels")

    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"groundtruth.csv not found at {gt_path}")
    if not os.path.exists(error_path):
        raise FileNotFoundError(f"error_traces.txt not found at {error_path}")

    os.makedirs(out_dir, exist_ok=True)

    groundtruth = pd.read_csv(gt_path)
    groundtruth["timestamp"] = pd.to_datetime(groundtruth["timestamp"], unit="s")

    with open(error_path) as f:
        lines = f.readlines()

    written = 0
    for i, line in enumerate(lines[1:], start=1):   # skip header
        try:
            parts = line.split()
            if len(parts) < 4:
                continue
            timestamp_str = parts[1]   # column index 1 in DataFrame str output
            nearest = find_nearest(timestamp_str, groundtruth)
            label_path = os.path.join(out_dir, f"trace_{i}_label.json")
            with open(label_path, "w") as fout:
                json.dump(
                    {"level": nearest["level"], "cmdb_id": nearest["cmdb_id"]},
                    fout,
                )
            written += 1
        except Exception as exc:
            print(f"  [warn] line {i}: {exc}")

    print(f"[preprocess_groundtruth] Written {written} label files → {out_dir}/")


if __name__ == "__main__":
    data_root_arg = sys.argv[1] if len(sys.argv) > 1 else "sample_data"
    build_labels(data_root_arg)
