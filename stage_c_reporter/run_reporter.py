# -*- coding: utf-8 -*-
"""
pipelines/run_reporter.py

Reporter pipeline (Stage C):
1) Run predictor_agent to produce an RCA report (Markdown + structured JSON).
2) Build evidence pack from Stage-B outputs.
3) Verify graph/text consistency and write a final summary.

Outputs:
- outputs/reports/llm_report.md
- outputs/reports/llm_report_raw.json
- outputs/reasoning/evidence_pack.json
- outputs/reports/consistency.json
- outputs/reports/single_case_report.txt
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------
# Import from project root
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Stage-C modules
from stage_c_reporter import predictor_agent as PA
from stage_c_reporter.cons_builder import build_constraints_from_reasoning
from stage_c_reporter.consistency import run_consistency

# ---------------------------------------------------------------------
# Paths (defaults)
# ---------------------------------------------------------------------
PERCEP_PT     = ROOT / "outputs" / "perception" / "mm_node_embeddings.pt"
REASON_JSON   = ROOT / "outputs" / "reasoning" / "single_case.json"
EVIDENCE_JSON = ROOT / "outputs" / "reasoning" / "evidence_pack.json"
REPORT_DIR    = ROOT / "outputs" / "reports"
LLM_MD        = REPORT_DIR / "llm_report.md"
LLM_RAW       = REPORT_DIR / "llm_report_raw.json"
CFG_LLM       = ROOT / "configs" / "llm_openai_compat.yaml"
CONS_JSON     = REPORT_DIR / "consistency.json"
FINAL_TXT     = REPORT_DIR / "single_case_report.txt"


# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def load_json(p: Path, default=None):
    if not p.exists():
        return default
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj, p: Path):
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return p

def banner(msg: str):
    print(f"\n=== {msg} ===")


# ---------------------------------------------------------------------
# 1) Run predictor_agent (LLM or fallback)
# ---------------------------------------------------------------------
def run_predictor(cfg_llm: Path, reason_json: Path, percep_pt: Path,
                  out_md: Path, out_json: Path, force: bool):
    if out_md.exists() and out_json.exists() and not force:
        print(f"[skip] predictor_agent outputs exist: {out_md.name}, {out_json.name}")
        return
    # Emulate CLI for predictor_agent.main()
    argv_backup = list(sys.argv)
    try:
        sys.argv = [
            "predictor_agent",
            "--cfg_llm", str(cfg_llm),
            "--reason_json", str(reason_json),
            "--percep_pt", str(percep_pt),
            "--out_md", str(out_md),
            "--out_json", str(out_json),
        ]
        PA.main()
    finally:
        sys.argv = argv_backup


# ---------------------------------------------------------------------
# 2) Build evidence pack from Stage-B results
# ---------------------------------------------------------------------
def build_evidence(reason_json: Path, out_json: Path):
    ev = build_constraints_from_reasoning(str(reason_json), k_root=10, use_all_edges=True)
    save_json(ev, out_json)
    print(f"[ok] evidence saved: {out_json}")


# ---------------------------------------------------------------------
# 3) Consistency verification
# ---------------------------------------------------------------------
def verify_consistency(evidence_json: Path, llm_raw: Path, out_json: Path):
    evidence = load_json(evidence_json, default={}) or {}
    llm_pred = load_json(llm_raw, default={}) or {}
    res = run_consistency(evidence, llm_pred)
    save_json(res, out_json)
    print(f"[ok] consistency saved: {out_json}")
    return res


# ---------------------------------------------------------------------
# 4) Final text report
# ---------------------------------------------------------------------
def write_final_report(final_txt: Path, llm_md: Path, cons: dict):
    ensure_dir(final_txt.parent)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append("Agentic RCA: Single-Case Final Report")
    lines.append("=====================================")
    lines.append(f"Generated at: {ts}")
    lines.append("")
    lines.append("Section A — Consistency Summary")
    lines.append("--------------------------------")
    lines.append(f"- overall_score     : {cons.get('score')}")
    c = cons.get("consistency", {})
    lines.append(f"- graph_valid       : {c.get('graph_valid')}")
    lines.append(f"- edge_coverage     : {c.get('edge_coverage')}")
    lines.append(f"- has_cycle         : {c.get('has_cycle')}")
    lines.append(f"- illegal_roots     : {c.get('illegal_roots')}")
    lines.append(f"- illegal_edges     : {c.get('illegal_edges')}")
    lines.append("")
    lines.append("Section B — LLM Report")
    lines.append("--------------------------------")
    lines.append(f"Markdown path: {llm_md}")
    lines.append("")
    lines.append("Notes")
    lines.append("--------------------------------")
    lines.append("- The LLM report is generated by stage_c_reporter/predictor_agent.py.")
    lines.append("- Consistency metrics are computed by stage_c_reporter/consistency.py.")
    lines.append("")

    with open(final_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[ok] final summary saved: {final_txt}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg_llm", default=str(CFG_LLM))
    ap.add_argument("--reason_json", default=str(REASON_JSON))
    ap.add_argument("--percep_pt", default=str(PERCEP_PT))
    ap.add_argument("--force", action="store_true", help="force regenerate predictor outputs")
    args = ap.parse_args()

    # Ensure dirs
    ensure_dir(REPORT_DIR)
    ensure_dir(EVIDENCE_JSON.parent)

    banner("Stage C | Predictor (LLM / Fallback)")
    run_predictor(
        cfg_llm=Path(args.cfg_llm),
        reason_json=Path(args.reason_json),
        percep_pt=Path(args.percep_pt),
        out_md=LLM_MD,
        out_json=LLM_RAW,
        force=args.force,
    )

    banner("Stage C | Build Evidence Pack")
    build_evidence(reason_json=Path(args.reason_json), out_json=EVIDENCE_JSON)

    banner("Stage C | Consistency Verification")
    cons = verify_consistency(evidence_json=EVIDENCE_JSON, llm_raw=LLM_RAW, out_json=CONS_JSON)

    banner("Stage C | Final Report")
    write_final_report(final_txt=FINAL_TXT, llm_md=LLM_MD, cons=cons)

    # EventBus-like log (stdout only)
    evt = {
        "sender": "ReporterAgent",
        "type": "REPORT_READY",
        "payload": {
            "report_md": str(LLM_MD),
            "report_txt": str(FINAL_TXT),
            "consistency_json": str(CONS_JSON),
            "evidence_json": str(EVIDENCE_JSON),
        },
    }
    print(f"\n[EventBus] -> {json.dumps(evt, indent=2)}")


if __name__ == "__main__":
    main()
