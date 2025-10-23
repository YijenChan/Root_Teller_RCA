# -*- coding: utf-8 -*-
"""
feedback_profiler.py
---------------------------------
Root-Teller Feedback Profiler

This module evaluates the consistency between
  - the LLM-generated RCA report (pred_root / pred_chain)
  - and the ground-truth RCA labels (true_root / true_chain).

It outputs a lightweight diagnostic JSON file for later visualization.

Usage:
    python stage_c_reporter/feedback_profiler.py \
        --report outputs/reports/llm_report_raw.json \
        --gt outputs/reasoning/single_case.json \
        --out outputs/reports/feedback_stats.json
"""

import os
import json
import argparse
from typing import Dict, Any, List, Tuple


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------
def load_json(path: str) -> Dict[str, Any]:
    """Load a JSON file safely."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def jaccard_similarity(a: List[Any], b: List[Any]) -> float:
    """Compute Jaccard index between two lists."""
    sa, sb = set(a or []), set(b or [])
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return round(len(sa & sb) / len(sa | sb), 4)


def compare_roots(pred_root: Any, true_root: Any) -> bool:
    """Return True if predicted root matches the ground truth."""
    try:
        return int(pred_root) == int(true_root)
    except Exception:
        return str(pred_root) == str(true_root)


def evaluate_feedback(report_json: Dict[str, Any],
                      gt_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute root accuracy, chain similarity and pass statistics.
    """
    pred_root = report_json.get("pred_root")
    pred_chain = report_json.get("pred_chain", [])
    true_root = gt_json.get("root") or gt_json.get("true_root")
    true_chain = gt_json.get("chain_nodes") or gt_json.get("true_chain", [])

    # Metrics
    root_match = compare_roots(pred_root, true_root)
    chain_jacc = jaccard_similarity(pred_chain, true_chain)
    pass_rate = round(0.5 * int(root_match) + 0.5 * chain_jacc, 4)

    diff_example = {
        "pred": pred_chain,
        "true": true_chain,
        "common": list(set(pred_chain) & set(true_chain)),
        "missing": list(set(true_chain) - set(pred_chain)),
        "extra": list(set(pred_chain) - set(true_chain)),
    }

    stats = {
        "root_match": root_match,
        "chain_jaccard": chain_jacc,
        "pass_rate": pass_rate,
        "diff_example": diff_example,
        "pred_root": pred_root,
        "true_root": true_root,
    }
    return stats


# ---------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True,
                    help="Path to llm_report_raw.json")
    ap.add_argument("--gt", required=True,
                    help="Path to single_case.json (ground truth)")
    ap.add_argument("--out", default="outputs/reports/feedback_stats.json",
                    help="Output JSON path")
    args = ap.parse_args()

    report = load_json(args.report)
    gt = load_json(args.gt)
    stats = evaluate_feedback(report, gt)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("[OK] Feedback statistics saved to:", args.out)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
