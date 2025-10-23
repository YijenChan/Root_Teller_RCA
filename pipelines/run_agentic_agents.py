# -*- coding: utf-8 -*-
# pipelines/run_agentic_agents.py
"""
使用 EventBus 串联三类 Agent，跑通单样例的 agentic 流程：
  Stage-A: DataProcessAgent.build_embeddings
  Stage-B: GraphReasonerAgent.train_and_eval
  Stage-C: LLMReporterAgent.generate_report
并将 Stage-B 的真实产物喂给 VerifierAgent 做一致性校验。
同时读取 agents/memory/profile.yaml，作为 profile_patch 注入 Reporter。
"""

import os
import sys
import json
import yaml

# —— 1) 注入项目根到 sys.path（在任何项目内 import 之前）——
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# —— 2) 项目内 import ——
from agents.bus import EventBus
from agents.data_process_agent import DataProcessAgent
from agents.graph_reasoner_agent import GraphReasonerAgent
from agents.llm_reporter_agent import LLMReporterAgent
from agents.verifier_agent import VerifierAgent


# ========== I/O & 安全工具 ==========
def load_yaml(p: str):
    if not os.path.exists(p):
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def load_json_safe(p: str):
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def pretty(obj) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return str(obj)

def to_int_or_none(x):
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return None

def to_int_list(obj):
    """把混合类型的 list 转为 int list，过滤无效元素"""
    if obj is None:
        return []
    if isinstance(obj, (list, tuple)):
        out = []
        for v in obj:
            iv = to_int_or_none(v)
            if iv is not None:
                out.append(iv)
        return out
    iv = to_int_or_none(obj)
    return [iv] if iv is not None else []

def to_edge_list_int(edges):
    """把 [[u,v], ...] 转成 int 对列表，过滤坏数据"""
    out = []
    if isinstance(edges, (list, tuple)):
        for e in edges:
            if isinstance(e, (list, tuple)) and len(e) == 2:
                u = to_int_or_none(e[0]); v = to_int_or_none(e[1])
                if u is not None and v is not None:
                    out.append([u, v])
    return out


def main():
    print(">>> Running agentic pipeline via EventBus")

    # 配置
    cfg_run = load_yaml(os.path.join(ROOT, "configs", "single_case.yaml"))
    cfg_llm_all = load_yaml(os.path.join(ROOT, "configs", "llm_openai_compat.yaml"))
    llm_cfg = cfg_llm_all.get("llm", {}) or {}

    # 读取 profile（你已有的旧版结构：style/thresholds/sections）
    profile_path = os.path.join(ROOT, "agents", "memory", "profile.yaml")
    profile_patch = load_yaml(profile_path)  # 若不存在则为空字典

    # —— 初始化总线与各 Agent ——
    bus = EventBus()
    a = DataProcessAgent("perception", bus=bus)
    b = GraphReasonerAgent("reasoner", bus=bus)
    c = LLMReporterAgent("reporter", bus=bus)
    v = VerifierAgent("verifier", bus=bus)

    # —— Stage A ——
    res_a = a.on_build_embeddings(
        type("M", (), {"payload": {"cfg_run": cfg_run.get("run", {}), "cfg_llm": llm_cfg}})
    )
    if not res_a.get("ok"):
        print("Stage-A failed:", pretty(res_a)); return
    print("Stage-A:", pretty(res_a["data"]))

    # —— Stage B ——
    res_b = b.on_train_and_eval(
        type("M", (), {"payload": {"cfg_run": cfg_run.get("run", {})}})
    )
    if not res_b.get("ok"):
        print("Stage-B failed:", pretty(res_b)); return
    print("Stage-B:", pretty(res_b["data"]))

    # —— Stage C ——
    # 将 profile_patch 传给报告生成（若 reporter 已支持，可在 on_generate_report 读取 payload.profile_patch）
    res_c = c.on_generate_report(
        type("M", (), {"payload": {"profile_patch": profile_patch}})
    )
    if not res_c.get("ok"):
        print("Stage-C failed:", pretty(res_c)); return
    print("Stage-C:", pretty(res_c["data"]))

    # ====== Verifier：读取真实产物进行一致性校验（仅用 Stage-B 的 single_case.json） ======
    reason_json = os.path.join(ROOT, "outputs", "reasoning", "single_case.json")
    reason = load_json_safe(reason_json)
    if reason is None:
        print("[WARN] missing reasoning json, skip VERIFY.")
        print("All done."); return

    # --- 取图与候选根 ---
    edges_raw = reason.get("edges", [])
    nodes = reason.get("nodes", []) or []
    edges = to_edge_list_int(edges_raw)

    gt_root = ((reason.get("ground_truth") or {}).get("root", None))
    eval_summary = (reason.get("eval_summary") or {})
    topk_roots = to_int_list(eval_summary.get("topk_roots", []))

    # —— evidence ——
    root_candidates = list({x for x in ([to_int_or_none(gt_root)] + topk_roots) if x is not None})
    evidence = {
        "constraints": {
            "allowed_edges": edges,               # [[u,v], ...]
            "root_candidates": root_candidates    # 候选根集合
        },
        "meta": {
            "num_nodes": len(nodes),
            "seed": (reason.get("meta") or {}).get("seed", None),
        }
    }

    # —— llm_pred ——
    # 直接用 Stage-B 的预测结果作为校验输入，稳定可用
    pred_root_b = to_int_or_none(eval_summary.get("pred_root"))
    pred_chain_b = to_int_list(eval_summary.get("pred_chain"))
    llm_pred = {
        "pred_root": [pred_root_b] if pred_root_b is not None else [],
        "pred_chain": [pred_chain_b] if pred_chain_b else []
    }

    # 空保护：若两者都为空，就不跑校验
    if not edges or not llm_pred["pred_root"]:
        print("[WARN] insufficient data for VERIFY. edges or pred_root missing.")
        print("All done."); return

    verify_res = v.handle(type("M", (), {"type": "VERIFY", "payload": {
        "evidence": evidence,
        "llm_pred": llm_pred
    }}))
    print("VERIFY:", pretty(verify_res))

    # 依据 profile 阈值做门槛提示（兼容你旧版字段名）
    thr = (profile_patch.get("thresholds") or {})
    gate = float(thr.get("min_consistency_score", thr.get("consistency_gate", 0.65)))
    if verify_res.get("ok") and verify_res["data"].get("score", 0.0) < gate:
        print(f">>> 标记：需人工复核 / 可触发反馈流程 (score<{gate:.2f})")

    print("All done.")


if __name__ == "__main__":
    main()
