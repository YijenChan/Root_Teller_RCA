"""
preprocessing_logs.py
=====================
Combine per-service log CSV files into a single ``log/all/logs.csv``, with
INFO / DEBUG lines filtered out by default to reduce context size. The
script is idempotent and skips rebuild when the output already exists
(pass ``--force`` to redo, ``--keep-info`` to retain all severities).

Expected raw layout under <data_root>:
  log/
    <service_or_any_name>/   *.csv  – columns: log_id, timestamp, cmdb_id, log_name, value
    ...
  log/all/                   – created by this script

Outputs:
  log/all/logs.csv           – combined, severity-filtered log table

Usage
-----
  python preprocess/preprocessing_logs.py [data_root] [--keep-info] [--force]
"""

import os
import sys
import argparse

import pandas as pd

EXPECTED_COLUMNS = ["log_id", "timestamp", "cmdb_id", "log_name", "value"]
INFO_DEBUG_PATTERN = r"severity:\s*(info|debug)"


def build_logs(data_root: str, keep_info: bool = False, force: bool = False) -> str:
    """Combine and filter log CSVs. Returns path to output logs.csv."""
    out_csv = os.path.join(data_root, "log", "all", "logs.csv")

    if not force and os.path.exists(out_csv):
        print("[preprocessing_logs] Output already exists, skipping. (use --force to rebuild)")
        return out_csv

    os.makedirs(os.path.join(data_root, "log", "all"), exist_ok=True)

    log_base = os.path.join(data_root, "log")
    out_name = os.path.basename(out_csv)  # "logs.csv" — never re-ingest our own output
    frames = []

    # Case 1: per-service subdirectories, e.g. log/<service>/*.csv
    # Case 2: AIOPS-2022 style — all raw CSVs directly under log/all/ alongside the output.
    scan_dirs = []
    for entry in os.scandir(log_base):
        if not entry.is_dir():
            continue
        scan_dirs.append(entry.path)

    for scan_dir in scan_dirs:
        for fname in os.listdir(scan_dir):
            if not fname.endswith(".csv") or fname == out_name:
                continue
            fpath = os.path.join(scan_dir, fname)
            try:
                df = pd.read_csv(fpath)
                # Ensure expected schema (add missing columns as empty).
                for col in EXPECTED_COLUMNS:
                    if col not in df.columns:
                        df[col] = ""
                df = df[EXPECTED_COLUMNS]
                frames.append(df)
            except Exception as exc:
                print(f"  [warn] skipping {fpath}: {exc}")

    if not frames:
        print("[preprocessing_logs] No raw log CSVs found; nothing to do.")
        return out_csv

    combined = pd.concat(frames, ignore_index=True)

    # Severity filter.
    if not keep_info and "value" in combined.columns:
        mask_noise = (
            combined["value"]
            .astype(str)
            .str.lower()
            .str.contains(INFO_DEBUG_PATTERN, regex=True, na=False)
        )
        before = len(combined)
        combined = combined[~mask_noise]
        print(f"  [severity filter] dropped {before - len(combined)} INFO/DEBUG rows")

    combined = combined.sort_values("timestamp").reset_index(drop=True)
    combined.to_csv(out_csv, index=False)
    print(f"[preprocessing_logs] Written {len(combined)} rows → {out_csv}")
    return out_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", nargs="?", default="sample_data")
    parser.add_argument("--keep-info", action="store_true", help="Keep INFO/DEBUG log lines")
    parser.add_argument("--force",     action="store_true", help="Rebuild even if output exists")
    args = parser.parse_args()
    build_logs(args.data_root, keep_info=args.keep_info, force=args.force)
