# -*- coding: utf-8 -*-
# agents/verifier_agent.py

import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from typing import Any, Dict
from agents.base import BaseAgent, Message
from stage_c_reporter.consistency import run_consistency

class VerifierAgent(BaseAgent):
    name = "verifier"
    def handle(self, msg: Message):
        if msg.type != "VERIFY":
            return self.reply_err(f"unknown message type: {msg.type}")
        ev = msg.payload.get("evidence", {})
        lp = msg.payload.get("llm_pred", {})
        res = run_consistency(ev, lp)
        return self.reply_ok(res)

if __name__ == "__main__":
    # 可选：本地快速自测
    from agents.bus import EventBus
    bus = EventBus()
    agent = VerifierAgent("verifier", bus=bus)
    dummy = Message(sender="tester", recipient="verifier", type="VERIFY",
                    payload={"evidence": {"constraints":{"root_candidates":["A"],"allowed_edges":[["A","B"]]}},
                             "llm_pred":{"pred_root":["A"],"pred_chain":[["A","B"]]}})
    print(agent.on_message(dummy))
