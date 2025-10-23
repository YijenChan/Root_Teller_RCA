# -*- coding: utf-8 -*-
# tools/collectors/fs_traces.py
import os, json, glob
from typing import List, Dict, Any

def collect_traces(root_dir: str) -> List[Dict[str, Any]]:
    """
    读取 trace_*.json 或 trace_*.jsonl
    """
    items = []
    for p in glob.glob(os.path.join(root_dir, "trace_*.json")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                items.append(data)
            elif isinstance(data, list):
                items.extend(data)
        except Exception:
            pass
    for p in glob.glob(os.path.join(root_dir, "trace_*.jsonl")):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except Exception:
                    pass
    return items
