# utils/io.py
import os, json

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path

def load_json(path: str, default=None):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj, path: str, indent: int = 2):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
