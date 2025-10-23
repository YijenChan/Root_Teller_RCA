# -*- coding: utf-8 -*-
# tools/collectors/fs_logs.py
import os, json, glob
from typing import List, Dict, Any

def collect_logs(root_dir: str) -> List[Dict[str, Any]]:
    """
    从 root_dir 下读取 *.log 或 *.jsonl，返回列表（每行或每条即一条记录）
    """
    items = []
    for p in glob.glob(os.path.join(root_dir, "*.jsonl")):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except Exception:
                    pass
    for p in glob.glob(os.path.join(root_dir, "*.log")):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                items.append({"raw": line.rstrip("\n")})
    return items
