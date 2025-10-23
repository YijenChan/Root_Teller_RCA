# -*- coding: utf-8 -*-
"""
pipelines/run_single_case.py

Single-case end-to-end pipeline:
  Stage A → Perception and Fusion
  Stage B → Graph Reasoning (R-GAT+time)
  Stage C → Reporting (Predictor + Feedback + EventBus)

Artifacts:
  outputs/perception/mm_node_embeddings.pt
  outputs/perception/meta.json
  outputs/reasoning/rgat_time_min.pt
  outputs/reasoning/single_case.json
  outputs/reasoning/evidence_pack.json
  outputs/reasoning/evidence_prompt.txt
  outputs/reports/llm_report.md
  outputs/reports/llm_report_raw.json
  outputs/reports/feedback_stats.json
  outputs/event_log.jsonl
"""

import os
import sys
import json
import shutil
import argparse
from types import SimpleNamespace
from pathlib import Path

import yaml
import torch

# ---------------------------------------------------------------------
# Import from project root
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.seed import set_seed
from stage_a_perception.adapters import from_demo_dir
from stage_a_perception import fusion_mid_raw as fm
from stage_b_reasoning import train_rgat as rgat
from stage_b_reasoning.model_rgat import RGATTimeNet
from stage_b_reasoning.data_rca import make_synthetic_rca
from stage_c_reporter.event_bus import EventBus

# stage_c_reporter tools
from stage_c_reporter import predictor_agent, feedback_profiler

try:
    from stage_b_reasoning.translator_graph2text import pack_to_prompt
except Exception:
    pack_to_prompt = None


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def copy_if_exists(src: str, dst: str):
    if os.path.exists(src):
        ensure_dir(os.path.dirname(dst))
        shutil.copy2(src, dst)
        print(f"  -> copied: {dst}")
        return True
    return False


def auto_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _to_float(x, default):
    try:
        return float(x)
    except Exception:
        return float(default)


def _to_int(x, default):
    try:
        return int(float(x))
    except Exception:
        return int(default)


def _to_bool(x, default=False):
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        return x.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(default)


# ---------------------------------------------------------------------
# Stage A: Perception + Mid-Fusion
# ---------------------------------------------------------------------
def run_stage_a(cfg_run, cfg_llm, workdirs, bus=None):
    """Run perception and fusion (Stage A)."""
    print("\n=== Stage A | Perception & Fusion ===")

    device = auto_device()
    sc = cfg_run.get("single_case", {}) or {}
    use_demo = bool(sc.get("use_demo_input", False))
    demo_dir = sc.get("demo_dir", str(ROOT / "outputs" / "single_case_demo"))

    args_a = SimpleNamespace(
        device=device,
        seed=_to_int(cfg_run.get("seed", 7), 7),
        epochs=_to_int(cfg_run.get("epochs_a", 8), 8),
        batch_size=_to_int(cfg_run.get("batch_size_a", 128), 128),
        lr=_to_float(cfg_run.get("lr_a", 1e-3), 1e-3),
        num_samples=_to_int(cfg_run.get("num_samples_a", 4000), 4000),
        d_node=_to_int(cfg_run.get("d_node", 128), 128),
        d_log=_to_int(cfg_run.get("d_log", 256), 256),
        d_met=_to_int(cfg_run.get("d_met", 128), 128),
        d_trc=_to_int(cfg_run.get("d_trc", 128), 128),
        use_openai_logs=_to_int(cfg_llm.get("enable", 0), 0),
        openai_api_base=str(cfg_llm.get("api_base", "")),
        openai_api_key=str(cfg_llm.get("api_key", "")),
    )

    # Encode-only mode
    if use_demo:
        print(f" [Stage A] Using demo input (encode-only): {demo_dir}")

        model = fm.FusionFromRaw(
            d_node=args_a.d_node,
            d_log=args_a.d_log,
            d_met=args_a.d_met,
            d_trc=args_a.d_trc,
            metrics_inC=4,
            use_openai_logs=bool(args_a.use_openai_logs),
            openai_api_base=args_a.openai_api_base,
            openai_api_key=args_a.openai_api_key,
        ).to(device).eval()

        logs, M, spans, lengths, Q, Y, meta = from_demo_dir(demo_dir)
        M, spans, lengths, Q = M.to(device), spans.to(device), lengths.to(device), Q.to(device)

        with torch.no_grad():
            _, H, aux = model(logs, M, spans, lengths, Q)

        per_dir = Path(workdirs["perception"])
        ensure_dir(str(per_dir))
        emb_path = per_dir / "mm_node_embeddings.pt"
        torch.save({"h": H.cpu(), "alpha": aux["alpha"].cpu(), "y": Y.cpu(), "meta": meta}, emb_path)
        (per_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"  -> saved embeddings: {emb_path}")

        if bus:
            bus.emit("StageA", "StageB", "DATA_READY", {"embeddings_path": str(emb_path), "meta": meta})

        return {"embeddings_path": str(emb_path)}

    # Training mode
    fm.train(args_a)
    emb_path = Path(workdirs["perception"]) / "mm_node_embeddings.pt"

    if not emb_path.exists():
        candidates = list(ROOT.glob("**/mm_node_embeddings.pt"))
        if candidates:
            newest = max(candidates, key=os.path.getmtime)
            copy_if_exists(newest, emb_path)
        else:
            print("  ! mm_node_embeddings.pt not found after training.")
    else:
        print(f"  -> fused embeddings: {emb_path}")

    meta_path = Path(workdirs["perception"]) / "meta.json"
    meta = json.load(open(meta_path, "r", encoding="utf-8")) if meta_path.exists() else {}

    if bus:
        bus.emit("StageA", "StageB", "DATA_READY", {"embeddings_path": str(emb_path), "meta": meta})

    return {"embeddings_path": str(emb_path)}


# ---------------------------------------------------------------------
# Stage B: Graph Reasoning (R-GAT + time)
# ---------------------------------------------------------------------
def run_stage_b(cfg_run, workdirs, bus=None):
    """Train, evaluate, and export reasoning results."""
    print("\n=== Stage B | Graph Reasoning (R-GAT + time) ===")
    device = auto_device()

    cfg_b = {
        "epochs": _to_int(cfg_run.get("epochs_b", 120), 120),
        "lr": _to_float(cfg_run.get("lr_b", 1e-3), 1e-3),
        "hidden": _to_int(cfg_run.get("hidden_b", 128), 128),
        "heads": _to_int(cfg_run.get("heads_b", 4), 4),
        "dropout": _to_float(cfg_run.get("dropout_b", 0.1), 0.1),
        "time_dim": _to_int(cfg_run.get("time_dim_b", 8), 8),
        "num_nodes": _to_int(cfg_run.get("num_nodes_b", 400), 400),
        "base_feat_dim": _to_int(cfg_run.get("base_feat_dim_b", 16), 16),
        "chain_len": _to_int(cfg_run.get("chain_len_b", 10), 10),
        "seed": _to_int(cfg_run.get("seed", 7), 7),
        "device": device,
    }

    rgat.train(**cfg_b)
    src = Path("rgat_time_min.pt")
    dst = Path(workdirs["reasoning"]) / "rgat_time_min.pt"
    copy_if_exists(src, dst)

    g = make_synthetic_rca(
        num_nodes=cfg_b["num_nodes"],
        base_feat_dim=cfg_b["base_feat_dim"],
        chain_len=cfg_b["chain_len"],
        seed=cfg_b["seed"],
    )
    model = RGATTimeNet(
        g.x.size(1),
        hidden=cfg_b["hidden"],
        num_classes=3,
        heads=cfg_b["heads"],
        time_dim=cfg_b["time_dim"],
        edge_dim=3,
        dropout=cfg_b["dropout"],
    ).to(device)

    try:
        state = torch.load(dst, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(dst, map_location=device)
    model.load_state_dict(state)

    res = rgat.evaluate(model, g, device=device, topk=5)

    reasoning_dir = Path(workdirs["reasoning"])
    metrics_path = reasoning_dir / "single_case.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {"cfg": cfg_b, "result": res, "true_root": g.root, "true_chain": g.chain_nodes},
            f,
            indent=2,
        )
    print(f"  -> saved metrics: {metrics_path}")

    if bus:
        bus.emit("StageB", "StageC", "REASONING_COMPLETE", {"metrics_path": str(metrics_path), "summary": res})

    if "evidence_pack" in res:
        evidence_path = reasoning_dir / "evidence_pack.json"
        with open(evidence_path, "w", encoding="utf-8") as f:
            json.dump(res["evidence_pack"], f, indent=2)
        print(f"  -> saved evidence pack: {evidence_path}")

        if pack_to_prompt is not None:
            try:
                prompt_txt = pack_to_prompt(str(evidence_path))
                prompt_path = reasoning_dir / "evidence_prompt.txt"
                prompt_path.write_text(prompt_txt, encoding="utf-8")
                print(f"  -> saved evidence prompt: {prompt_path}")
            except Exception as e:
                print(f"  ! translator_graph2text failed: {e}")

    return {"model_path": str(dst), "metrics_path": str(metrics_path)}


# ---------------------------------------------------------------------
# Stage C: Reporting + Feedback
# ---------------------------------------------------------------------
def run_stage_c(paths, cfg_llm, bus=None):
    """Execute Reporter Agent: predictor + feedback profiler."""
    print("\n=== Stage C | Reporter Agent ===")

    predictor_agent.main()
    feedback_profiler.main()

    if bus:
        bus.emit("StageC", None, "REPORT_GENERATED", {"ok": True})

    print("  -> Stage C completed: RCA report and feedback stats generated.")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_run", default=str(ROOT / "configs" / "single_case.yaml"))
    parser.add_argument("--cfg_llm", default=str(ROOT / "configs" / "llm_openai_compat.yaml"))
    args = parser.parse_args()

    cfg_run = load_yaml(args.cfg_run) or {}
    cfg_llm_all = load_yaml(args.cfg_llm) or {}
    llm_cfg = cfg_llm_all.get("llm", {})

    if os.getenv("USE_OPENAI_LOGS"):
        llm_cfg["enable"] = 1
        llm_cfg["api_key"] = os.getenv("OPENAI_API_KEY", llm_cfg.get("api_key", ""))
        llm_cfg["api_base"] = os.getenv("OPENAI_API_BASE", llm_cfg.get("api_base", ""))

    run_cfg = cfg_run.get("run", {})
    set_seed(_to_int(run_cfg.get("seed", 7), 7))

    paths = {
        "workdir": cfg_run.get("run", {}).get("workdir", "outputs"),
        "perception": cfg_run.get("paths", {}).get("perception", "outputs/perception"),
        "reasoning": cfg_run.get("paths", {}).get("reasoning", "outputs/reasoning"),
        "reports": cfg_run.get("paths", {}).get("reports", "outputs/reports"),
    }
    for k, v in paths.items():
        p = Path(ROOT) / v if not os.path.isabs(v) else Path(v)
        paths[k] = str(ensure_dir(str(p)))

    print(">>> Running single-case pipeline")
    print(f" - ROOT    : {ROOT}")
    print(f" - Workdir : {paths['workdir']}")
    print(f" - Device  : {auto_device()}")

    bus = EventBus(console=True)

    res_a = run_stage_a(run_cfg, llm_cfg, paths, bus)
    res_b = run_stage_b(run_cfg, paths, bus)
    run_stage_c(paths, llm_cfg, bus)

    print("\nPipeline completed successfully.")
    print(f"Perception outputs: {res_a}")
    print(f"Reasoning outputs : {res_b}")


if __name__ == "__main__":
    main()
