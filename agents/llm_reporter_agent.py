# -*- coding: utf-8 -*-
# agents/llm_reporter_agent.py
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agents.base import BaseAgent, Message

class LLMReporterAgent(BaseAgent):
    """
    职责：调用 stage_c_reporter/predictor_agent.py 生成 LLM 报告
    输入 payload（可选，若缺省使用默认 outputs 路径）:
      - cfg_llm_path: str (默认 configs/llm_openai_compat.yaml)
      - reason_json : str (默认 outputs/reasoning/single_case.json)
      - percep_pt   : str (默认 outputs/perception/mm_node_embeddings.pt)
      - out_md/out_json: 输出文件路径
    返回:
      { ok: True, data: {"md": ".../outputs/reports/llm_report.md", "json": "..."} }
    """
    def handle(self, msg: Message):
        return self.reply_err(f"unknown message type: {msg.type}")

    def on_generate_report(self, msg: Message):
        payload = msg.payload or {}
        cfg_llm  = payload.get("cfg_llm_path", os.path.join(ROOT, "configs", "llm_openai_compat.yaml"))
        reason   = payload.get("reason_json",  os.path.join(ROOT, "outputs", "reasoning", "single_case.json"))
        percep   = payload.get("percep_pt",    os.path.join(ROOT, "outputs", "perception", "mm_node_embeddings.pt"))
        out_md   = payload.get("out_md",       os.path.join(ROOT, "outputs", "reports", "llm_report.md"))
        out_json = payload.get("out_json",     os.path.join(ROOT, "outputs", "reports", "llm_report_raw.json"))

        # 为了尽量少改 reporter 脚本，直接以“模块 + 伪 argv”方式调用其 main()
        import importlib
        module = importlib.import_module("stage_c_reporter.predictor_agent")
        argv_bak = sys.argv[:]
        try:
            sys.argv = [
                "predictor_agent.py",
                f"--cfg_llm={cfg_llm}",
                f"--reason_json={reason}",
                f"--percep_pt={percep}",
                f"--out_md={out_md}",
                f"--out_json={out_json}",
            ]
            module.main()
        finally:
            sys.argv = argv_bak

        self.log(f"LLM report generated -> {out_md}")
        return self.reply_ok({"md": out_md, "json": out_json})
