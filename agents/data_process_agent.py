# -*- coding: utf-8 -*-
# agents/data_process_agent.py
import os, sys, shutil, json
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agents.base import BaseAgent, Message
from stage_a_perception import fusion_mid_raw as fm

def _to_float(x, d):
    try: return float(x)
    except: return float(d)

def _to_int(x, d):
    try: return int(float(x))
    except: return int(d)

class DataProcessAgent(BaseAgent):
    """
    职责：调用 Stage-A 训练并导出 mm_node_embeddings.pt
    输入 payload:
      - cfg_run: dict（可选：epochs_a/batch_size_a/lr_a/...）
      - cfg_llm: dict（可选：enable/api_base/api_key）
    返回:
      { ok: True, data: {"embeddings_path": ".../outputs/perception/mm_node_embeddings.pt"} }
    """
    def handle(self, msg: Message):
        return self.reply_err(f"unknown message type: {msg.type}")

    def on_build_embeddings(self, msg: Message):
        run = msg.payload.get("cfg_run", {}) or {}
        llm = msg.payload.get("cfg_llm", {}) or {}

        device = "cuda"
        a_args = SimpleNamespace(
            device=device,
            seed=_to_int(run.get("seed", 7), 7),
            epochs=_to_int(run.get("epochs_a", 8), 8),
            batch_size=_to_int(run.get("batch_size_a", 128), 128),
            lr=_to_float(run.get("lr_a", 1e-3), 1e-3),
            num_samples=_to_int(run.get("num_samples_a", 4000), 4000),
            d_node=_to_int(run.get("d_node", 128), 128),
            d_log=_to_int(run.get("d_log", 256), 256),
            d_met=_to_int(run.get("d_met", 128), 128),
            d_trc=_to_int(run.get("d_trc", 128), 128),
            use_openai_logs=int(_to_int(llm.get("enable", 0), 0)),
            openai_api_base=str(llm.get("api_base", "")),
            openai_api_key=str(llm.get("api_key", "")),
        )
        self.log(f"Stage-A starting... epochs={a_args.epochs} lr={a_args.lr}")

        fm.train(a_args)

        # 规范化输出位置
        out_dir = os.path.join(ROOT, "outputs", "perception")
        os.makedirs(out_dir, exist_ok=True)
        dst = os.path.join(out_dir, "mm_node_embeddings.pt")
        # 训练函数可能写在当前工作目录
        for cand in [dst, os.path.join(os.getcwd(),"mm_node_embeddings.pt")]:
            if os.path.exists(cand):
                if cand != dst:
                    shutil.copy2(cand, dst)
                self.log(f"saved embeddings -> {dst}")
                return self.reply_ok({"embeddings_path": dst})

        # 兜底全局搜索
        from glob import glob
        candidates = glob(os.path.join(ROOT, "**", "mm_node_embeddings.pt"), recursive=True)
        if candidates:
            newest = max(candidates, key=os.path.getmtime)
            shutil.copy2(newest, dst)
            self.log(f"saved embeddings -> {dst}")
            return self.reply_ok({"embeddings_path": dst})

        return self.reply_err("mm_node_embeddings.pt not found")
