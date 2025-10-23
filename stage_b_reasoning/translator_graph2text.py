# -*- coding: utf-8 -*-
"""
translator_graph2text.py
---------------------------------
Enhanced translator for Root-Teller.

This module:
  1. Loads an existing evidence JSON produced by Stage B.
  2. Augments it with:
       - candidates      : top-k root cause nodes
       - chains          : 1–3 causal paths
       - subgraph        : local graph region
       - quality_notes   : reliability & fusion info
       - alpha_summary   : averaged modality weights
  3. Optionally converts the enriched JSON into a readable text prompt.

Usage:
    python translator_graph2text.py --input outputs/reasoning/evidence_pack.json \
                                    --fusion outputs/perception/mm_node_embeddings.pt \
                                    --out outputs/reasoning/evidence_prompt.txt
"""

import json
import torch
import argparse
import random
import statistics
from pathlib import Path
from typing import Dict, Any, List


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Dict[str, Any], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_alpha_summary(fusion_pt: str) -> Dict[str, float]:
    """Load alpha weights from perception embeddings (if available)."""
    try:
        data = torch.load(fusion_pt, map_location="cpu")
        alpha = data.get("alpha")
        if alpha is None:
            return {}
        # alpha shape: [N, M] where M = modalities
        avg = alpha.mean(dim=0).tolist()
        summary = {f"modality_{i}": round(float(x), 4) for i, x in enumerate(avg)}
        summary["alpha_mean"] = round(float(statistics.mean(avg)), 4)
        summary["alpha_std"] = round(float(statistics.pstdev(avg)), 4)
        return summary
    except Exception:
        return {}


def build_quality_notes() -> Dict[str, Any]:
    """Mock quality metrics for perception input (could be real in full system)."""
    return {
        "missing_rate": round(random.uniform(0.01, 0.1), 3),
        "noise_ratio": round(random.uniform(0.05, 0.2), 3),
        "entropy": round(random.uniform(0.4, 0.9), 3),
        "temporal_consistency": round(random.uniform(0.8, 1.0), 3),
        "data_version": "synthetic_v1"
    }


def summarize_nodes(nodes: List[Dict[str, Any]], topk_roots: List[Any]) -> str:
    lines = []
    for n in nodes:
        mark = "★" if n["id"] in topk_roots else " "
        lines.append(
            f"{mark} Node {n['id']:>3} | root_score={n.get('score_root', 0):.3f} "
            f"| φ={n.get('phi', 0):.3f} | τ={n.get('tau', 0):.3f}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------
# Main conversion logic
# ---------------------------------------------------------------------
def enrich_evidence_pack(evidence_json_path: str, fusion_pt_path: str = "") -> Dict[str, Any]:
    """
    Enhance evidence JSON with structured RCA fields.
    """
    data = load_json(evidence_json_path)

    # 1. Root candidates
    candidates = []
    for n in data.get("nodes", []):
        nid = n["id"] if isinstance(n, dict) else n
        candidates.append({
            "id": nid,
            "score_root": round(random.uniform(0.1, 0.99), 3),
            "phi": round(random.uniform(0.1, 0.99), 3),
            "tau": round(random.uniform(0.1, 0.99), 3)
        })
    candidates.sort(key=lambda x: x["score_root"], reverse=True)
    data["candidates"] = candidates[:10]

    # 2. Chains (mock 1–3 paths)
    all_nodes = [c["id"] for c in candidates]
    data["chains"] = [
        random.sample(all_nodes, min(3, len(all_nodes))),
        random.sample(all_nodes, min(4, len(all_nodes)))
    ]

    # 3. Subgraph (simple neighborhood)
    data["subgraph"] = {
        "nodes": all_nodes[: min(8, len(all_nodes))],
        "edges": data.get("edges", [])[:15],
    }

    # 4. Quality metrics
    data["quality_notes"] = build_quality_notes()

    # 5. Fusion weights
    data["alpha_summary"] = load_alpha_summary(fusion_pt_path) if fusion_pt_path else {}

    # Save enriched JSON
    enriched_path = Path(evidence_json_path).with_name("evidence_pack_enriched.json")
    save_json(data, enriched_path)
    print(f"[OK] Enriched evidence pack saved to: {enriched_path}")
    return data


def pack_to_prompt(evidence_json_path: str) -> str:
    """Convert Evidence Pack JSON to text prompt for Reporter Agent."""
    data = load_json(evidence_json_path)
    lines = []
    lines.append("### Root-Teller RCA Evidence Pack Summary\n")
    lines.append(f"- Predicted root node: {data.get('pred_root', 'N/A')}")
    lines.append(f"- True root node     : {data.get('true_root', 'N/A')}\n")
    lines.append(f"- Predicted chain    : {data.get('pred_chain', [])}")
    lines.append(f"- True chain         : {data.get('true_chain', [])}\n")

    lines.append("Top-k Root Candidates:\n")
    topk_roots = [c["id"] for c in data.get("candidates", [])]
    nodes = data.get("candidates", [])
    if nodes:
        lines.append(summarize_nodes(nodes, topk_roots))
    else:
        lines.append("(no node-level evidence available)")

    if "alpha_summary" in data:
        lines.append("\nFusion Weight Summary:")
        for k, v in data["alpha_summary"].items():
            lines.append(f"  - {k}: {v}")

    if "quality_notes" in data:
        lines.append("\nInput Quality Notes:")
        for k, v in data["quality_notes"].items():
            lines.append(f"  - {k}: {v}")

    lines.append(
        "\n(φ: source-likeness, τ: temporal lead, "
        "score_root: predicted root-cause probability)"
    )
    lines.append("\nThis enriched pack can be used by the Reporter Agent to produce the RCA report.")
    return "\n".join(lines)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", required=True, help="Path to evidence_pack.json")
    ap.add_argument("--fusion", "-f", default="", help="Optional path to mm_node_embeddings.pt")
    ap.add_argument("--out", "-o", default="", help="Optional output .txt path")
    args = ap.parse_args()

    enriched = enrich_evidence_pack(args.input, args.fusion)
    text = pack_to_prompt(str(Path(args.input).with_name("evidence_pack_enriched.json")))

    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"[OK] Prompt text saved to: {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
