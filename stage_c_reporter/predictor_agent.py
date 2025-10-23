# -*- coding: utf-8 -*-
"""
stage_c_reporter/predictor_agent.py

Generates an RCA (Root Cause Analysis) report.
LLM-based generation is preferred; if unavailable, falls back to a static Markdown template.

Aligned with the current architecture:
- Reads modality contribution summary (alpha) from Stage-A: mm_node_embeddings.pt
- Reads eval_summary / meta / ground_truth from Stage-B: single_case.json
- Computes Top1–Top2 confidence margin
- Writes or fills missing keys (pred_root, pred_chain) in llm_report_raw.json for the Verifier
"""

import os
import sys
import json
import argparse
import torch

# ---------------------------------------------------------------------
# Path setup for both package and direct script execution
# ---------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))   # .../stage_c_reporter
ROOT = os.path.dirname(THIS_DIR)                        # .../agentic_rca
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from .llm_client import build_client_from_cfg
except Exception:
    from stage_c_reporter.llm_client import build_client_from_cfg

PERCEP_PATH = os.path.join(ROOT, "outputs", "perception", "mm_node_embeddings.pt")
REASON_JSON = os.path.join(ROOT, "outputs", "reasoning", "single_case.json")
REPORT_MD   = os.path.join(ROOT, "outputs", "reports", "llm_report.md")
REPORT_JSON = os.path.join(ROOT, "outputs", "reports", "llm_report_raw.json")
CFG_LLM     = os.path.join(ROOT, "configs", "llm_openai_compat.yaml")


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------
def _safe_torch_load(path, map_location="cpu"):
    """Try torch.load with weights_only=True; fallback to regular load if unsupported."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _to_int_or_none(x):
    """Convert to int if possible, otherwise None."""
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return None


def _to_int_list(obj):
    """Convert an object or iterable into a list of ints (skipping invalid values)."""
    if obj is None:
        return []
    if isinstance(obj, (list, tuple)):
        out = []
        for v in obj:
            iv = _to_int_or_none(v)
            if iv is not None:
                out.append(iv)
        return out
    iv = _to_int_or_none(obj)
    return [iv] if iv is not None else []


# ---------------------------------------------------------------------
# Data loading and preprocessing
# ---------------------------------------------------------------------
def _load_alpha_summary(path: str):
    """Load modality contribution (alpha) statistics from Stage-A output."""
    if not os.path.exists(path):
        return None
    try:
        obj = _safe_torch_load(path, map_location="cpu")
        A = obj.get("alpha", None)  # [B,3] => logs, metrics, traces
        if A is None or A.numel() == 0:
            return None
        A = A.detach().cpu().float()
        means = A.mean(dim=0).tolist()
        p25 = A.quantile(0.25, dim=0).tolist()
        p50 = A.quantile(0.50, dim=0).tolist()
        p75 = A.quantile(0.75, dim=0).tolist()
        return {
            "mean": {"logs": means[0], "metrics": means[1], "traces": means[2]},
            "p25":  {"logs": p25[0],   "metrics": p25[1],   "traces": p25[2]},
            "p50":  {"logs": p50[0],   "metrics": p50[1],   "traces": p50[2]},
            "p75":  {"logs": p75[0],   "metrics": p75[1],   "traces": p75[2]},
        }
    except Exception:
        return None


def _load_reasoning_result(path: str):
    """
    Load reasoning results from Stage-B.

    Expected structure of single_case.json:
    {
      "nodes": [...],
      "edges": [[u,v], ...],
      "scores": {...},
      "meta": {...},
      "ground_truth": { "root": int, "chain_nodes": [int] },
      "eval_summary": {
         "test_acc": float, "root_hit@1": float, "root_hit@5": float,
         "chain_jaccard": float, "pred_root": int,
         "topk_roots": [int], "topk_scores": [float],
         "pred_chain": [int]
      }
    }
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    meta = obj.get("meta", {}) or {}
    gt   = obj.get("ground_truth", {}) or {}
    ev   = obj.get("eval_summary", {}) or {}

    res = {
        "test_acc": ev.get("test_acc"),
        "root_hit@1": ev.get("root_hit@1"),
        "root_hit@5": ev.get("root_hit@5"),
        "chain_jaccard": ev.get("chain_jaccard"),
        "pred_root": ev.get("pred_root"),
        "pred_chain": ev.get("pred_chain") or [],
        "topk_roots": ev.get("topk_roots") or [],
        "topk_scores": ev.get("topk_scores") or [],
    }
    cfg = {
        "num_nodes": meta.get("num_nodes"),
        "time_dim":  meta.get("time_dim"),
        "heads":     meta.get("heads"),
        "hidden":    meta.get("hidden"),
    }
    reason_prompt_view = {
        "true_root":  gt.get("root"),
        "true_chain": gt.get("chain_nodes") or []
    }
    return obj, res, cfg, reason_prompt_view


def _format_modal_contrib(mc):
    """Format modality contribution summary for text prompts."""
    if not mc:
        return "No alpha summary available."
    return (
        f"- logs    : {mc['mean']['logs']:.3f} "
        f"(p25={mc['p25']['logs']:.3f}, p50={mc['p50']['logs']:.3f}, p75={mc['p75']['logs']:.3f})\n"
        f"- metrics : {mc['mean']['metrics']:.3f} "
        f"(p25={mc['p25']['metrics']:.3f}, p50={mc['p50']['metrics']:.3f}, p75={mc['p75']['metrics']:.3f})\n"
        f"- traces  : {mc['mean']['traces']:.3f} "
        f"(p25={mc['p25']['traces']:.3f}, p50={mc['p50']['traces']:.3f}, p75={mc['p75']['traces']:.3f})\n"
    )


# ---------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------
def _build_messages(res, cfg, modal_contrib, reason_prompt_view):
    """Compose system/user messages for the LLM."""
    topk_scores = res.get("topk_scores") or []
    margin = None
    if isinstance(topk_scores, list) and len(topk_scores) >= 2:
        try:
            margin = float(topk_scores[0] - topk_scores[1])
        except Exception:
            margin = None

    system = (
        "You are a cybersecurity RCA copilot. Produce an actionable, concise Markdown report. "
        "Anchor every claim to concrete evidence (scores, margins, chains). Avoid hallucinations."
    )

    user = f"""
# Inputs
- num_nodes: {cfg.get('num_nodes')}, time_dim: {cfg.get('time_dim')}
- heads/hidden: {cfg.get('heads')}/{cfg.get('hidden')}

# Metrics
- test_acc: {res.get('test_acc')}
- root_hit@1: {res.get('root_hit@1')}, chain_jaccard: {res.get('chain_jaccard')}

# Predictions
- pred_root: {res.get('pred_root')}
- pred_chain: {res.get('pred_chain')}

# Ground Truth
- true_root: {reason_prompt_view.get('true_root')}
- true_chain: {reason_prompt_view.get('true_chain')}

# Top-K Root Candidates
- topk_roots: {res.get('topk_roots')}
- topk_scores: {res.get('topk_scores')}
- confidence_margin(top1 - top2): {margin if margin is not None else "N/A"}

# Modal Contributions (alpha from Stage-A)
{_format_modal_contrib(modal_contrib)}

Please write a Markdown report with sections:
1) Suspected Root Cause (quantify confidence using the margin and scores)
2) Chain Comparison (point out exact matches/mismatches)
3) Evidence Bullets (use modal contributions and any temporal/structural cues)
4) Remediation Steps & Monitoring Items (short, prioritized)
5) Limitations & Next Checks (what to verify next)
"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _fallback_md(res, cfg, modal_contrib, reason_prompt_view, err):
    """Fallback Markdown report used when LLM call fails."""
    tks = res.get("topk_scores") or []
    margin = "N/A"
    if isinstance(tks, list) and len(tks) >= 2:
        try:
            margin = float(tks[0]) - float(tks[1])
        except Exception:
            margin = "N/A"

    return f"""# Agentic RCA Report (Fallback)

> LLM call failed or was not configured; a static offline report is generated.
> Reason: `{str(err)}`

## Inputs
- num_nodes: {cfg.get('num_nodes')}, time_dim: {cfg.get('time_dim')}, heads/hidden: {cfg.get('heads')}/{cfg.get('hidden')}

## Metrics
- test_acc: {res.get('test_acc')}
- root_hit@1: {res.get('root_hit@1')}, chain_jaccard: {res.get('chain_jaccard')}

## Predictions
- pred_root: **{res.get('pred_root')}**
- pred_chain: {res.get('pred_chain')}

## Ground Truth
- true_root: **{reason_prompt_view.get('true_root')}**
- true_chain: {reason_prompt_view.get('true_chain')}

## Modal Contributions (alpha)
{_format_modal_contrib(modal_contrib)}

## Top-K Root Candidates
- topk_roots: {res.get('topk_roots')}
- topk_scores: {res.get('topk_scores')}
- Confidence margin (Top1-Top2): {margin}

## Suggested Actions
- Replay metrics/logs for predicted root and its direct neighbors.
- Investigate time-consistent edges to confirm propagation direction and speed.
- Prioritize modalities with higher alpha contributions for deeper inspection.

## Limitations
- This report is offline-only and lacks LLM-driven synthesis.
"""


# ---------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg_llm", default=CFG_LLM)
    ap.add_argument("--reason_json", default=REASON_JSON)
    ap.add_argument("--percep_pt", default=PERCEP_PATH)
    ap.add_argument("--out_md", default=REPORT_MD)
    ap.add_argument("--out_json", default=REPORT_JSON)
    args = ap.parse_args()

    # Load reasoning results and modality summary
    reason_obj, res, cfg, reason_prompt_view = _load_reasoning_result(args.reason_json)
    modal_contrib = _load_alpha_summary(args.percep_pt)

    # Ensure valid fields for Verifier
    pred_root = _to_int_or_none(res.get("pred_root"))
    pred_chain = _to_int_list(res.get("pred_chain"))
    if pred_root is None:
        tkr = res.get("topk_roots") or []
        if tkr:
            pred_root = _to_int_or_none(tkr[0])

    # Build LLM prompt messages
    messages = _build_messages(res, cfg, modal_contrib, reason_prompt_view)

    ok, err = True, None
    try:
        client = build_client_from_cfg(args.cfg_llm)
        md = client.chat(messages, temperature=0.2, max_tokens=1200)
    except Exception as e:
        ok, err = False, e
        md = _fallback_md(res, cfg, modal_contrib, reason_prompt_view, e)

    # Save Markdown + JSON outputs
    os.makedirs(os.path.dirname(args.out_md), exist_ok=True)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(md)

    raw = {}
    if os.path.exists(args.out_json):
        try:
            with open(args.out_json, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception:
            raw = {}

    payload = {
        "ok": ok,
        "error": str(err) if err else None,
        "messages": messages,
        "pred_root": pred_root,
        "pred_chain": pred_chain,
    }
    raw.update(payload)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)

    print(f"Saved: {args.out_md}")
    if not ok:
        print(f"(fallback) Reason: {err}")


if __name__ == "__main__":
    main()
