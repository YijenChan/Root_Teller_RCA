# -*- coding: utf-8 -*-
"""
stage_c_reporter/consistency.py

Consistency verification module for the Reporter Agent.

It checks whether the LLM-predicted root causes and chains
are valid under the graph constraints derived from reasoning outputs.
It also performs a lightweight adversarial backtrace placeholder and
produces a unified consistency score.
"""

from typing import Dict, Any, List, Tuple


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _edges_to_set(edges: List[List[str]]) -> set:
    """Convert a list of edges into a set of tuples for fast lookup."""
    return set(tuple(e) for e in edges)


# ----------------------------------------------------------------------
# Core consistency checks
# ----------------------------------------------------------------------
def check_graph_consistency(llm_pred: Dict[str, Any],
                            constraints: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate LLM predictions (root, chains) against allowed graph constraints.

    Args:
        llm_pred: Dict containing LLM predictions
            - pred_root: list of node IDs
            - pred_chain: list of node sequences
        constraints: Dict containing
            - allowed_edges: list of [u, v]
            - root_candidates: list of node IDs

    Returns:
        A dict summarizing validation status and metrics.
    """
    roots = llm_pred.get("pred_root", []) or []
    chains = llm_pred.get("pred_chain", []) or []
    allowed = _edges_to_set(constraints.get("allowed_edges", []))
    root_cands = set(constraints.get("root_candidates", []))

    graph_valid = True
    illegal_edges, illegal_roots = [], []
    edge_total, edge_hit = 0, 0
    has_cycle = False

    # Check roots
    for r in roots:
        if r not in root_cands:
            illegal_roots.append(r)
            graph_valid = False

    # Check chains and edge coverage
    for path in chains:
        seen = set()
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edge_total += 1
            if (u, v) in allowed:
                edge_hit += 1
            else:
                illegal_edges.append([u, v])
                graph_valid = False
            if v in seen:  # simple cycle detection
                has_cycle = True
            seen.add(u)

    edge_cov = edge_hit / max(1, edge_total)
    return {
        "graph_valid": graph_valid,
        "illegal_roots": illegal_roots,
        "illegal_edges": illegal_edges,
        "edge_coverage": round(edge_cov, 4),
        "has_cycle": has_cycle,
    }


# ----------------------------------------------------------------------
# Adversarial placeholder & scoring
# ----------------------------------------------------------------------
def adversarial_backtrace(root: str,
                          constraints: Dict[str, Any],
                          k: int = 3) -> Dict[str, Any]:
    """
    Placeholder for adversarial backtrace.
    (Future extension: controlled prompts or graph simulation.)

    Returns:
        Dummy alignment results for compatibility.
    """
    return {"agree_ratio": 1.0, "divergent_edges": []}


def score_consistency(cons: Dict[str, Any], adv: Dict[str, Any]) -> float:
    """
    Compute an overall consistency score based on structure and alignment.

    Formula (bounded to [0, 1]):
        score = 0.5 * validity + 0.3 * edge_coverage + 0.2 * agree_ratio
                - penalty(cycles, illegal edges/roots)
    """
    s = 0.0
    s += 0.5 if cons["graph_valid"] else 0.0
    s += 0.3 * cons["edge_coverage"]
    s += 0.2 * float(adv.get("agree_ratio", 1.0))

    if cons["has_cycle"]:
        s -= 0.15
    if cons["illegal_edges"] or cons["illegal_roots"]:
        s -= 0.2

    return max(0.0, min(1.0, round(s, 4)))


# ----------------------------------------------------------------------
# Unified entry point
# ----------------------------------------------------------------------
def run_consistency(evidence: Dict[str, Any],
                    llm_pred: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run consistency verification and compute a unified result.

    Args:
        evidence: evidence pack containing constraints
        llm_pred: LLM prediction results

    Returns:
        Dict with fields:
            - consistency (structure check)
            - adversarial_llm (placeholder backtrace)
            - score (float [0,1])
    """
    cons = check_graph_consistency(llm_pred, evidence.get("constraints", {}))
    roots = llm_pred.get("pred_root", [])
    adv = (
        adversarial_backtrace(roots[0], evidence.get("constraints", {}), k=3)
        if roots else {"agree_ratio": 0.0}
    )

    return {
        "consistency": cons,
        "adversarial_llm": adv,
        "score": score_consistency(cons, adv),
    }
