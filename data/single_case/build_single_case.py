# -*- coding: utf-8 -*-
"""
data/single_case/build_single_case.py

快速生成一个可调种子的“单个样例”，与 stage_a_perception.fusion_mid_raw.make_one_sample
返回结构保持一致；可选择落盘到 outputs/single_case_demo/。
"""
from __future__ import annotations
import os, sys, json, numpy as np

# --- 让脚本无论从哪里启动都能找到项目根 ---
THIS = os.path.abspath(__file__)
SINGLE_CASE_DIR = os.path.dirname(THIS)       # .../data/single_case
DATA_DIR       = os.path.dirname(SINGLE_CASE_DIR)  # .../data
ROOT           = os.path.dirname(DATA_DIR)         # .../agentic_rca
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 复用已有的样例生成函数
try:
    from stage_a_perception.fusion_mid_raw import make_one_sample
except Exception as e:
    raise ImportError(
        "无法导入 stage_a_perception.fusion_mid_raw.make_one_sample。\n"
        "建议在项目根运行：python -m data.single_case.build_single_case"
    ) from e

def build(seed: int = 7):
    """返回与 fusion_mid_raw.make_one_sample 相同结构的 dict"""
    return make_one_sample(seed=seed)

def save(sample, out_dir: str):
    """将单样例落盘，返回各文件路径"""
    os.makedirs(out_dir, exist_ok=True)
    paths = {
        "log_text": os.path.join(out_dir, "log.txt"),
        "metrics": os.path.join(out_dir, "metrics.npy"),
        "traces":  os.path.join(out_dir, "traces.npy"),
        "meta":    os.path.join(out_dir, "meta.json"),
    }
    with open(paths["log_text"], "w", encoding="utf-8") as f:
        f.write(sample["log_text"])
    np.save(paths["metrics"], np.asarray(sample["metrics"], dtype=np.float32))
    np.save(paths["traces"],  np.asarray(sample["trace_spans"], dtype=np.float32))
    meta = {"y": int(sample["y"]), "q": [float(x) for x in np.asarray(sample["q"]).tolist()]}
    with open(paths["meta"], "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return paths

def load(in_dir: str):
    """从目录加载单样例为内存结构"""
    with open(os.path.join(in_dir, "log.txt"), "r", encoding="utf-8") as f:
        log_text = f.read()
    metrics = np.load(os.path.join(in_dir, "metrics.npy"))
    traces  = np.load(os.path.join(in_dir, "traces.npy"))
    with open(os.path.join(in_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    return {
        "log_text": log_text,
        "metrics": metrics.astype(np.float32),
        "trace_spans": traces.astype(np.float32),
        "y": int(meta.get("y", 0)),
        "q": np.array(meta.get("q", [1,1,1]), dtype=np.float32),
    }

if __name__ == "__main__":
    out = os.path.join(ROOT, "outputs", "single_case_demo")
    paths = save(build(seed=7), out)
    print("Saved single case to:")
    for k, v in paths.items():
        print(f" - {k}: {v}")
    sample = load(out)
    print("Loaded keys:", list(sample.keys()))
