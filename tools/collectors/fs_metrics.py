# -*- coding: utf-8 -*-
# tools/collectors/fs_metrics.py
import os, json, glob
from typing import List, Dict, Any

def collect_metrics(root_dir: str) -> List[Dict[str, Any]]:
    """
    读取 metrics_*.json 汇总
    """
    items = []
    for p in glob.glob(os.path.join(root_dir, "metrics_*.json")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                items.append(data)
            elif isinstance(data, list):
                items.extend(data)
        except Exception:
            pass
    return items
