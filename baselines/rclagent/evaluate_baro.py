"""
evaluate_baro.py — Run BARO on RE2-OB, AIOPS 2022 subsets, and Nezha-30,
compute R@k and MRR so results are comparable to our RCLAgent evaluation.

Usage:
  python evaluate_baro.py re2ob
  python evaluate_baro.py aiops 2022-03-20-cloudbed2
  python evaluate_baro.py nezha nezha-2023-01-30
  python evaluate_baro.py all
"""

import os, sys, re, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

from baro.root_cause_analysis import robust_scorer

DATA_BASE = Path(__file__).parent / "data"

# ── helpers ────────────────────────────────────────────────────────────────────

def _k8s_service_base(name: str) -> str:
    low = name.lower()
    m = re.match(r"^(.+?)-[a-z0-9]{5,}-[a-z0-9]{4,}$", low)
    if m: return m.group(1)
    if low[-1:].isdigit(): return low.rsplit("-", 1)[0]
    return low

def _label_hit(label: str, ranked: list) -> int:
    """0-based rank of first match, or -1."""
    label_base = _k8s_service_base(label)
    for i, svc in enumerate(ranked):
        svc_base = _k8s_service_base(svc)
        if label_base == svc_base or label_base in svc_base or svc_base in label_base:
            return i
    return -1

def mrr_and_recall(hits: list, ks=(1, 3, 5, 10)):
    n = len(hits)
    if n == 0: return {}
    recip = [1.0/(h+1) if h >= 0 else 0.0 for h in hits]
    result = {"MRR": np.mean(recip)}
    for k in ks:
        result[f"R@{k}"] = np.mean([1 if 0 <= h < k else 0 for h in hits])
    return result

def print_results(name, metrics, n_faults, n_gt):
    print(f"\nResults  ({n_faults} faults evaluated out of {n_gt} groundtruth, data_root={name})")
    if n_faults < n_gt:
        print(f"  WARNING: {n_gt-n_faults} groundtruth fault(s) skipped (insufficient metric window)")
    for k in [1,3,5,10]:
        r = metrics.get(f"R@{k}", 0)
        cnt = int(round(r * n_faults))
        print(f"  R@{k}  : {r:.4f}  ({cnt}/{n_faults})")
    print(f"  MRR  : {metrics.get('MRR',0):.4f}")

# ── RE2-OB ─────────────────────────────────────────────────────────────────────

def _get_window_wide(raw_df: pd.DataFrame, inject_time: int,
                     time_col="time", svc_col="service_name", kpi_col="kpi_name",
                     before=1200, after=600) -> pd.DataFrame:
    """Filter a window around inject_time and pivot to wide format."""
    t_start = inject_time - before
    t_end   = inject_time + after
    window = raw_df[(raw_df[time_col] >= t_start) & (raw_df[time_col] <= t_end)].copy()
    if len(window) < 5:
        return pd.DataFrame()
    window["col"] = window[svc_col].str.lower() + "_" + window[kpi_col]
    wide = window.pivot_table(index=time_col, columns="col", values="value", aggfunc="mean")
    wide = wide.reset_index().rename(columns={time_col: "time"}).sort_values("time")
    return wide

def _run_baro(wide: pd.DataFrame, inject_time: int) -> list:
    """Run BARO and return ranked service names."""
    result = robust_scorer(wide, inject_time=inject_time)
    ranks_raw = result.get("ranks", [])
    # ranks_raw is a list of column-name strings like "checkoutservice_cpu"
    ranked_svcs = []
    seen = set()
    for item in ranks_raw:
        col = item[0] if isinstance(item, (list, tuple)) else item
        svc = _k8s_service_base(str(col).rsplit("_", 1)[0])
        if svc not in seen:
            ranked_svcs.append(svc)
            seen.add(svc)
    return ranked_svcs

def eval_re2ob(data_root: Path):
    print(f"Loading RE2-OB metrics (9M rows, may take ~30s)...")
    raw = pd.read_csv(data_root / "metric" / "all" / "metrics.csv")
    raw = raw.rename(columns={"timestamp": "time"})
    gt = pd.read_csv(data_root / "groundtruth.csv")

    hits = []
    skipped = 0
    for i, row in gt.iterrows():
        inject_time = int(row["timestamp"])
        label = str(row["cmdb_id"]).lower()
        wide = _get_window_wide(raw, inject_time,
                                svc_col="service_name", kpi_col="kpi_name")
        if wide.empty:
            skipped += 1
            continue
        try:
            ranked = _run_baro(wide, inject_time)
            hits.append(_label_hit(label, ranked))
        except Exception:
            skipped += 1
        if (i+1) % 10 == 0:
            print(f"  {i+1}/{len(gt)} faults done...")

    n_faults = len(hits)
    n_gt = len(gt)
    metrics = mrr_and_recall(hits)
    print_results("re2ob", metrics, n_faults, n_gt)

# ── AIOPS 2022 ─────────────────────────────────────────────────────────────────

def load_aiops_raw(data_root: Path) -> pd.DataFrame:
    """Load AIOPS container metrics into a raw long-format df."""
    frames = []
    for subdir in ["container"]:   # container metrics most relevant
        metric_dir = data_root / "metric" / subdir
        if not metric_dir.exists():
            continue
        for f in metric_dir.glob("*.csv"):
            try:
                df = pd.read_csv(f)
                if "timestamp" not in df.columns or "cmdb_id" not in df.columns:
                    continue
                df = df.rename(columns={"timestamp": "time"})
                # cmdb_id like "node-5.checkoutservice2-0" → take part after last dot
                df["service_name"] = df["cmdb_id"].apply(
                    lambda x: x.split(".")[-1] if "." in str(x) else str(x)
                ).str.lower()
                frames.append(df[["time","service_name","kpi_name","value"]])
            except Exception:
                pass
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

def eval_aiops(data_root: Path):
    print(f"Loading AIOPS metrics from {data_root.name}...")
    raw = load_aiops_raw(data_root)
    if raw.empty:
        print("  No metric data found, skipping.")
        return

    gt = pd.read_csv(data_root / "groundtruth.csv")
    if "timestamp" not in gt.columns:
        print("  No timestamp in groundtruth, skipping.")
        return
    gt = gt.sort_values("timestamp").reset_index(drop=True)

    hits = []
    skipped = 0
    for i, row in gt.iterrows():
        inject_time = int(row["timestamp"])
        label = str(row["cmdb_id"]).lower()
        wide = _get_window_wide(raw, inject_time,
                                svc_col="service_name", kpi_col="kpi_name")
        if wide.empty:
            skipped += 1
            continue
        try:
            ranked = _run_baro(wide, inject_time)
            hits.append(_label_hit(label, ranked))
        except Exception:
            skipped += 1

    n_faults = len(hits)
    n_gt = len(gt)
    metrics = mrr_and_recall(hits)
    print_results(data_root.name, metrics, n_faults, n_gt)
    return metrics

# ── Nezha ──────────────────────────────────────────────────────────────────────

def load_nezha_raw(data_root: Path) -> pd.DataFrame:
    """Load Nezha metrics into long-format df."""
    frames = []
    for f in (data_root / "metric").rglob("*.csv"):
        try:
            df = pd.read_csv(f)
            if "timestamp" in df.columns:
                df = df.rename(columns={"timestamp": "time"})
                svc_col = "service_name" if "service_name" in df.columns else "cmdb_id"
                df["service_name"] = df[svc_col].str.lower()
                frames.append(df[["time","service_name","kpi_name","value"]])
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

def eval_nezha(data_root: Path):
    print(f"Loading Nezha metrics from {data_root.name}...")
    raw = load_nezha_raw(data_root)
    if raw.empty:
        print("  No metric data found, skipping.")
        return

    gt = pd.read_csv(data_root / "groundtruth.csv")
    if "timestamp" not in gt.columns:
        print("  No timestamp in groundtruth, skipping.")
        return

    hits = []
    skipped = 0
    for _, row in gt.iterrows():
        inject_time = int(row["timestamp"])
        label = str(row["cmdb_id"]).lower()
        wide = _get_window_wide(raw, inject_time,
                                svc_col="service_name", kpi_col="kpi_name")
        if wide.empty:
            skipped += 1
            continue
        try:
            ranked = _run_baro(wide, inject_time)
            hits.append(_label_hit(label, ranked))
        except Exception:
            skipped += 1

    n_faults = len(hits)
    n_gt = len(gt)
    metrics = mrr_and_recall(hits)
    print_results(data_root.name, metrics, n_faults, n_gt)
    return metrics

# ── main ───────────────────────────────────────────────────────────────────────

AIOPS_SUBSETS = [
    "2022-03-20-cloudbed2",
    "2022-03-20-cloudbed3",
    "2022-03-21-cloudbed1",
    "2022-03-21-cloudbed2",
    "2022-03-21-cloudbed3",
    "2022-03-24-cloudbed3",
]

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    if mode == "re2ob":
        eval_re2ob(DATA_BASE / "re2ob")

    elif mode == "aiops":
        subset = sys.argv[2] if len(sys.argv) > 2 else None
        if subset:
            eval_aiops(DATA_BASE / subset)
        else:
            all_hits = []
            for s in AIOPS_SUBSETS:
                print(f"\n{'='*60}")
                eval_aiops(DATA_BASE / s)

    elif mode == "nezha":
        subset = sys.argv[2] if len(sys.argv) > 2 else "nezha-2023-01-30"
        eval_nezha(DATA_BASE / subset)

    elif mode == "all":
        print("="*60)
        print("BARO Evaluation — RE2-OB")
        print("="*60)
        eval_re2ob(DATA_BASE / "re2ob")

        print("\n" + "="*60)
        print("BARO Evaluation — AIOPS 2022")
        print("="*60)
        for s in AIOPS_SUBSETS:
            print(f"\n--- {s} ---")
            eval_aiops(DATA_BASE / s)

        print("\n" + "="*60)
        print("BARO Evaluation — Nezha-30")
        print("="*60)
        eval_nezha(DATA_BASE / "nezha-2023-01-30")
