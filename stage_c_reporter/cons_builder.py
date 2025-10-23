# -*- coding: utf-8 -*-
"""
stage_c_reporter/cons_builder.py

Builds structured constraint and evidence objects for RCA verification.
- Extracts node/edge/top-k root candidates from Stage-B reasoning output.
- Normalizes graph structure for downstream consistency checking.
- Loads LLM-predicted report outputs for comparison.
"""

import json
from typing import Dict, Any, List, Tuple


# ---------------------------------------------------------------------
# Basic JSON utility
# ---------------------------------------------------------------------
def load_json(path: str) -> Dict[str, Any]:
    """Load a JSON file with UTF-8 encoding."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------
# Constraint builder (from reasoning results)
# ---------------------------------------------------------------------
def build_constraints_from_reasoning(
    reasoning_json_path: str,
    k_root: int = 10,
    use_all_edges: bool = True,
) -> Dict[str, Any]:
    """
    Build a normalized constraint object from Stage-B reasoning results.

    Expected structure of `single_case.json`:
    {
        "nodes": ["n0","n1",...],               # or [{"id":"n0"}, ...]
        "edges": [["u","v"], ...],              # or [{"u":"n0","v":"n1"}, ...]
        "scores": {"node_p": {"n0":{"root":0.83,"chain":0.1}, ...}},
        "meta": {...}                           # optional
    }
    """
    result = load_json(reasoning_json_path)

    # Normalize node and edge structure
    nodes = [n["id"] if isinstance(n, dict) else n for n in result.get("nodes", [])]
    raw_edges = result.get("edges", [])
    edges: List[Tuple[str, str]] = []
    for e in raw_edges:
        if isinstance(e, dict):
            edges.append((e.get("u"), e.get("v")))
        else:
            edges.append((e[0], e[1]))

    # Sort candidate roots by root-probability
    node_p = (result.get("scores", {}) or {}).get("node_p", {})
    items = []
    for nid in nodes:
        pr = node_p.get(nid, {}).get("root", 0.0)
        items.append((nid, pr))
    items.sort(key=lambda x: x[1], reverse=True)
    root_candidates = [nid for nid, _ in items[:k_root]] if items else nodes[:k_root]

    constraints = {
        "root_candidates": root_candidates,
        "allowed_edges": edges if use_all_edges else edges[:],
    }

    evidence = {
        "nodes": [{"id": n} for n in nodes],
        "edges": [{"u": u, "v": v} for (u, v) in edges],
        "scores": {"node_p": node_p},
        "constraints": constraints,
        "metadata": result.get("meta", {}),
    }
    return evidence


# ---------------------------------------------------------------------
# LLM prediction loader
# ---------------------------------------------------------------------
def build_llm_pred_from_report(report_json_path: str) -> Dict[str, Any]:
    """
    Load predicted roots and chains from the LLM report.

    Expected structure of `llm_report_raw.json`:
    {
        "pred_root": ["nX"],
        "pred_chain": [["nX","nY",...]],
        "reason": [...],
        "uncertainty": {...}
    }
    """
    report = load_json(report_json_path)
    pred_root = report.get("pred_root", [])
    pred_chain = report.get("pred_chain", [])

    # Backward-compatible fallbacks
    if not pred_chain:
        pred_chain = report.get("chains", report.get("paths", []))

    return {
        "pred_root": pred_root,
        "pred_chain": pred_chain,
        "reason": report.get("reason", []),
        "uncertainty": report.get("uncertainty", {}),
    }
