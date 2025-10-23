# utils/seed.py
import torch, random, numpy as np

def set_seed(s: int = 7):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
