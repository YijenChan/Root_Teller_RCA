"""
evaluate.py – Compute R@k and MRR for RCLAgent results.

Usage
-----
  python evaluate.py [data_root] [result_sub_dir]

  data_root       path to dataset dir (default: DATA_ROOT from config.py / env)
  result_sub_dir  subdirectory under data_root for result txt files
                  (default: result)

Evaluation methodology
----------------------
* Each dataset has N fault-injection cases (groundtruth entries).
* Each fault may produce multiple error traces (observed symptoms).
* Evaluation is **per fault**: for each groundtruth entry, we take the
  **best rank** across all traces that map to that fault (nearest-timestamp
  matching).  This follows the standard RCL evaluation protocol where one
  fault = one evaluation unit.
* Label matching is case-insensitive and handles service-level labels matching
  pod-level candidates (e.g. "recommendationservice" matches
  "recommendationservice-0").
* Prints per-case misses for diagnosis.
"""

import json
import os
import re
import sys
import warnings
from collections import defaultdict
from datetime import datetime

import pandas as pd

warnings.filterwarnings("ignore")

import config


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_unix(ts_str: str) -> datetime:
    ts_str = str(ts_str)
    if "T" in ts_str:
        return datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%SZ")
    return datetime.utcfromtimestamp(int(ts_str[:10]))


def find_nearest_groundtruth_idx(timestamp_str: str, groundtruth_df: pd.DataFrame) -> int:
    """Return the index of the nearest groundtruth entry by timestamp."""
    dt    = _parse_unix(timestamp_str)
    diffs = abs(groundtruth_df["timestamp"] - dt)
    return diffs.idxmin()


def _k8s_service_base(name: str) -> str:
    """
    Extract deployment/service base from a K8s pod name or AIOPS pod name.
    - 'ts-contacts-service-866bd68c97-xcqfx' → 'ts-contacts-service'
    - 'checkoutservice-0' → 'checkoutservice'
    - 'recommendationservice' → 'recommendationservice'
    """
    low = name.lower()
    # K8s pod name pattern: {service}-{replicaset_hash}-{pod_hash}
    # The hash segments are 5+ chars of [a-z0-9]
    m = re.match(r"^(.+?)-[a-z0-9]{5,}-[a-z0-9]{4,}$", low)
    if m:
        return m.group(1)
    # AIOPS pattern: {service}-{digit}
    if low[-1:].isdigit():
        return low.rsplit("-", 1)[0]
    return low


def _label_hit(label: str, candidates: list) -> int:
    """
    0-based rank of first hit, or -1.
    Matching rules (following the paper §V-A):
    - Exact match
    - Service-base match: label and candidate share the same deployment/service name
      (handles K8s pod names, AIOPS pod indices, and service-level names)
    """
    label_low = label.lower()
    label_base = _k8s_service_base(label_low)

    for rank, candidate in enumerate(candidates):
        cand_low = str(candidate).lower()
        cand_base = _k8s_service_base(cand_low)

        # Exact match
        if label_low == cand_low:
            return rank
        # Same service base (covers pod↔service, pod↔pod, k8s pod↔service)
        if label_base == cand_base and label_base:
            return rank
        # Prefix match: label=service, candidate=pod (or vice versa)
        if cand_low.startswith(label_low) and (len(cand_low) == len(label_low) or cand_low[len(label_low)] == '-'):
            return rank
        if label_low.startswith(cand_low) and (len(label_low) == len(cand_low) or label_low[len(cand_low)] == '-'):
            return rank

    return -1


# ── main ──────────────────────────────────────────────────────────────────────

def evaluate(data_root: str, result_sub_dir: str = "result"):
    gt_path    = os.path.join(data_root, "groundtruth.csv")
    error_path = os.path.join(data_root, "error_traces.txt")
    result_dir = os.path.join(data_root, result_sub_dir)

    groundtruth = pd.read_csv(gt_path)
    groundtruth["timestamp"] = pd.to_datetime(groundtruth["timestamp"], unit="s")

    with open(error_path) as f:
        error_lines = f.readlines()

    result_files = sorted(
        [f for f in os.listdir(result_dir) if f.endswith(".txt")],
        key=lambda x: int(x.split("conversation_trace_")[1].split(".")[0]),
    )

    # ── Phase 1: collect per-trace ranks, grouped by groundtruth fault ────────
    # Key: groundtruth row index  →  list of (rank, result_file) tuples
    fault_ranks = defaultdict(list)

    for result_file in result_files:
        index = int(result_file.split("conversation_trace_")[1].split(".")[0])
        if index >= len(error_lines):
            continue
        error_line    = error_lines[index]
        timestamp_str = error_line.split()[1]

        result_path = os.path.join(result_dir, result_file)
        try:
            with open(result_path) as f:
                data = json.load(f)
            root_causes = data.get("root_causes", [])
        except Exception as exc:
            print(f"[warn] cannot load {result_file}: {exc}")
            continue

        gt_idx = find_nearest_groundtruth_idx(timestamp_str, groundtruth)
        label  = str(groundtruth.loc[gt_idx, "cmdb_id"]).lower()
        rank   = _label_hit(label, root_causes)

        fault_ranks[gt_idx].append((rank, result_file, root_causes))

    # ── Phase 2: per-fault best-of-N aggregation ─────────────────────────────
    r1 = r3 = r5 = r10 = total = 0
    mrr_sum = 0.0

    for gt_idx in sorted(fault_ranks.keys()):
        label = str(groundtruth.loc[gt_idx, "cmdb_id"]).lower()
        entries = fault_ranks[gt_idx]
        n_traces = len(entries)

        # Best rank across all traces for this fault (lowest non-negative)
        valid = [(r, f, rc) for r, f, rc in entries if r >= 0]
        if valid:
            best_rank, best_file, _ = min(valid, key=lambda x: x[0])
        else:
            best_rank = -1
            best_file = entries[0][1] if entries else "?"

        total += 1
        if best_rank >= 0:
            if best_rank < 1:  r1  += 1
            if best_rank < 3:  r3  += 1
            if best_rank < 5:  r5  += 1
            if best_rank < 10: r10 += 1
            mrr_sum += 1.0 / (best_rank + 1)
        else:
            # Show top5 from the first trace for diagnosis
            top5 = entries[0][2][:5] if entries else []
            print(f"  [miss] gt[{gt_idx}] label={label}  "
                  f"traces={n_traces}  best_file={best_file}  top5={top5}")

    if total == 0:
        print("No results found.")
        return

    n_gt = len(groundtruth)
    uncovered = n_gt - total
    print(f"\nResults  ({total} faults evaluated out of {n_gt} groundtruth, "
          f"{sum(len(v) for v in fault_ranks.values())} traces, "
          f"data_root={data_root})")
    if uncovered:
        print(f"  WARNING: {uncovered} groundtruth fault(s) have no matching "
              f"result traces — run more traces to cover them.")
    print(f"  R@1  : {r1  / total:.4f}  ({r1}/{total})")
    print(f"  R@3  : {r3  / total:.4f}  ({r3}/{total})")
    print(f"  R@5  : {r5  / total:.4f}  ({r5}/{total})")
    print(f"  R@10 : {r10 / total:.4f}  ({r10}/{total})")
    print(f"  MRR  : {mrr_sum / total:.4f}")


if __name__ == "__main__":
    data_root_arg  = sys.argv[1] if len(sys.argv) > 1 else config.DATA_ROOT
    result_sub_arg = sys.argv[2] if len(sys.argv) > 2 else config.RESULT_SUB_DIR
    evaluate(data_root_arg, result_sub_arg)
