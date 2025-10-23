# -*- coding: utf-8 -*-
# agents/quick_test_bus.py
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agents.base import BaseAgent, Message
from agents.bus import EventBus

class EchoAgent(BaseAgent):
    def on_ping(self, msg: Message):
        self.log(f"got ping: {msg.payload}")
        return self.reply_ok({"pong": True, "echo": msg.payload})

if __name__ == "__main__":
    bus = EventBus()
    echo = EchoAgent("echo", bus=bus)
    resp = echo.ask("echo", "ping", {"x": 1})
    print(resp)
