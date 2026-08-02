"""
preprocessing_metrics.py
========================
Merge raw container / node / service metric CSVs into a single unified
``metrics.csv``, and build node↔service mapping pickle files consumed by
the tool server.

Expected raw layout under <data_root>:
  metric/
    container/   *.csv  – columns: timestamp, cmdb_id (node.service), kpi_name, value
    node/        *.csv  – columns: timestamp, cmdb_id (node_id),       kpi_name, value
    service/     *.csv  – columns: timestamp, service, rr, sr, mrt, count
  metric/all/           – created by this script

Outputs:
  metric/all/metrics.csv
  metric/node_service_map.pkl   node_id → set(service_name)
  metric/service_node_map.pkl   service_name → node_id

The script is idempotent: it skips the rebuild when the output already
exists (pass ``--force`` to redo).

Usage
-----
  python preprocess/preprocessing_metrics.py [data_root]
"""

import os
import sys
import pickle
import argparse

import pandas as pd


SERVICE_KPI_COLUMNS = ["rr", "sr", "mrt", "count"]


# ── helpers ───────────────────────────────────────────────────────────────────

def _read_csv_dir(directory: str) -> pd.DataFrame:
    """Concatenate every CSV in directory; return empty DataFrame on missing dir."""
    if not os.path.isdir(directory):
        return pd.DataFrame()
    files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(".csv")
    ]
    if not files:
        return pd.DataFrame()
    frames = []
    for path in files:
        try:
            frames.append(pd.read_csv(path))
        except Exception as exc:
            print(f"  [warn] skipping {path}: {exc}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _process_container(df: pd.DataFrame, node_service_map: dict, service_node_map: dict):
    """Explode cmdb_id = 'node.service' into separate columns."""
    rows = []
    for _, row in df.iterrows():
        parts = str(row["cmdb_id"]).split(".", 1)
        if len(parts) != 2:
            continue
        node_id, service_name = parts
        rows.append(
            {
                "timestamp":    row["timestamp"],
                "node_id":      node_id,
                "service_name": service_name,
                "kpi_name":     row["kpi_name"],
                "value":        row["value"],
            }
        )
        node_service_map.setdefault(node_id, set()).add(service_name)
        service_node_map[service_name] = node_id
    return pd.DataFrame(rows)


def _process_node(df: pd.DataFrame):
    """Node-level metrics: no associated service."""
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "timestamp":    row["timestamp"],
                "node_id":      row["cmdb_id"],
                "service_name": "",
                "kpi_name":     row["kpi_name"],
                "value":        row["value"],
            }
        )
    return pd.DataFrame(rows)


def _process_service(df: pd.DataFrame):
    """Pivot service-level KPI columns into long format."""
    rows = []
    for _, row in df.iterrows():
        for kpi in SERVICE_KPI_COLUMNS:
            if kpi not in df.columns:
                continue
            rows.append(
                {
                    "timestamp":    row["timestamp"],
                    "node_id":      "",
                    "service_name": row.get("service", row.get("service_name", "")),
                    "kpi_name":     kpi,
                    "value":        row[kpi],
                }
            )
    return pd.DataFrame(rows)


# ── main logic ────────────────────────────────────────────────────────────────

def build_metrics(data_root: str, force: bool = False) -> str:
    """Build unified metrics.csv and mapping pickles. Returns output CSV path."""
    out_csv  = os.path.join(data_root, "metric", "all", "metrics.csv")
    out_ns   = os.path.join(data_root, "metric", "node_service_map.pkl")
    out_sn   = os.path.join(data_root, "metric", "service_node_map.pkl")

    if not force and os.path.exists(out_csv) and os.path.exists(out_ns):
        print(f"[preprocessing_metrics] Output already exists, skipping. (use --force to rebuild)")
        return out_csv

    os.makedirs(os.path.join(data_root, "metric", "all"), exist_ok=True)

    node_service_map: dict = {}
    service_node_map: dict = {}
    frames = []

    # ── container ──
    container_raw = _read_csv_dir(os.path.join(data_root, "metric", "container"))
    if not container_raw.empty:
        container_df = _process_container(container_raw, node_service_map, service_node_map)
        frames.append(container_df)
        print(f"  container rows: {len(container_df)}")

    # ── node ──
    node_raw = _read_csv_dir(os.path.join(data_root, "metric", "node"))
    if not node_raw.empty:
        node_df = _process_node(node_raw)
        frames.append(node_df)
        print(f"  node rows:      {len(node_df)}")

    # ── service ──
    service_raw = _read_csv_dir(os.path.join(data_root, "metric", "service"))
    if not service_raw.empty:
        service_df = _process_service(service_raw)
        frames.append(service_df)
        print(f"  service rows:   {len(service_df)}")

    if not frames:
        print("[preprocessing_metrics] No raw metric CSVs found; nothing to do.")
        return out_csv

    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(out_csv, index=False)
    print(f"[preprocessing_metrics] Written {len(all_df)} rows → {out_csv}")

    with open(out_ns, "wb") as f:
        pickle.dump(node_service_map, f)
    with open(out_sn, "wb") as f:
        pickle.dump(service_node_map, f)
    print(f"[preprocessing_metrics] Written mapping pickles → {os.path.dirname(out_ns)}/")

    return out_csv


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", nargs="?", default="sample_data")
    parser.add_argument("--force", action="store_true", help="Rebuild even if output exists")
    args = parser.parse_args()
    build_metrics(args.data_root, force=args.force)
