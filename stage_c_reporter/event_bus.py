# -*- coding: utf-8 -*-
"""
stage_c_reporter/event_bus.py

Lightweight asynchronous EventBus for Stage A→B→C coordination.
Each event includes metadata (timestamp, sender, recipient, type, payload).
The bus writes events both to console (structured log) and to a JSONL file
at `outputs/event_log.jsonl`.

Usage Example:
    from stage_c_reporter.event_bus import EventBus

    bus = EventBus()
    bus.emit("StageA", "StageB", "DATA_READY", {"num_nodes": 512})
    bus.emit("StageB", "StageC", "REASONING_COMPLETE", {"root": 21})
    bus.emit("StageC", None, "REPORT_GENERATED", {"status": "ok"})
"""

import os
import sys
import json
import time
import threading
from typing import Any, Dict, Optional

# Allow both package and script execution
THIS_DIR = os.path.dirname(os.path.abspath(__file__))   # .../stage_c_reporter
ROOT = os.path.dirname(THIS_DIR)                        # .../agentic_rca
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Default output file under project root
DEFAULT_EVENT_LOG = os.path.join(ROOT, "outputs", "event_log.jsonl")


# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------
def _timestamp() -> str:
    """Return a human-readable UTC timestamp."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------------
# Core EventBus class
# ---------------------------------------------------------------------
class EventBus:
    """
    Minimal event bus for cross-agent communication and audit logging.

    Each event record has the format:
    {
        "timestamp": "2025-10-23T13:05:41Z",
        "sender": "StageA",
        "recipient": "StageB",
        "type": "DATA_READY",
        "payload": {...}
    }
    """

    def __init__(self, log_path: Optional[str] = None, console: bool = True):
        self.log_path = log_path or DEFAULT_EVENT_LOG
        self.console = console
        self.lock = threading.Lock()
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        # Initialize file if not present
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", encoding="utf-8") as f:
                f.write("")  # empty JSONL

    # -----------------------------------------------------------------
    def emit(
        self,
        sender: str,
        recipient: Optional[str],
        event_type: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Emit an event to the bus, logging it to console and JSONL file.

        Args:
            sender: Name of emitting agent (e.g., "StageA").
            recipient: Target agent or None if broadcast.
            event_type: One of {"DATA_READY", "REASONING_COMPLETE", "REPORT_GENERATED"}.
            payload: Optional dictionary with event-specific data.
        """
        event = {
            "timestamp": _timestamp(),
            "sender": sender,
            "recipient": recipient,
            "type": event_type,
            "payload": payload or {},
        }

        # Write to file atomically
        with self.lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")

        # Optional console log
        if self.console:
            pretty = json.dumps(event, ensure_ascii=False, indent=2)
            print(f"[EventBus] {event_type} — {sender} → {recipient or 'ALL'}")
            print(pretty)
            print("-" * 80)

        return event

    # -----------------------------------------------------------------
    def last_events(self, n: int = 10):
        """Return the last `n` events from the log file."""
        if not os.path.exists(self.log_path):
            return []
        with open(self.log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()[-n:]
        return [json.loads(l) for l in lines if l.strip()]


# ---------------------------------------------------------------------
# Example usage (manual testing)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    bus = EventBus()
    bus.emit("StageA", "StageB", "DATA_READY", {"num_nodes": 400})
    time.sleep(0.5)
    bus.emit("StageB", "StageC", "REASONING_COMPLETE", {"root": 42})
    time.sleep(0.5)
    bus.emit("StageC", None, "REPORT_GENERATED", {"status": "ok"})

    print("\nRecent events:")
    for e in bus.last_events(3):
        print(f"  {e['timestamp']} | {e['type']} | {e['sender']} → {e['recipient']}")
