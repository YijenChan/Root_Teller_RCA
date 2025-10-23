# -*- coding: utf-8 -*-
"""
fusion_mid_raw.py
Lightweight three-modality perception and mid-level fusion.
Encodes logs (text) / metrics (time series) / traces (span sequences)
and fuses them using adaptive content–quality gating.
Outputs node embeddings h aligned with downstream R-GAT input.
"""

import argparse, os, random, math, json, hashlib
from typing import List, Dict, Any
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------ Reproducibility ------------------
def set_seed(seed=7):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    except Exception:
        pass


# =========================================================
# I. Minimal Encoders
# =========================================================
class HashingTextEncoder(nn.Module):
    """Simple hash-based text encoder."""
    def __init__(self, num_buckets=4096, d_emb=256, pad_idx: int = 0):
        super().__init__()
        self.num_buckets = num_buckets
        self.d_emb = d_emb
        self.pad_idx = pad_idx
        self.emb = nn.Embedding(num_buckets, d_emb, padding_idx=pad_idx)
        nn.init.normal_(self.emb.weight, std=0.02)
        if 0 <= pad_idx < num_buckets:
            with torch.no_grad():
                self.emb.weight[pad_idx].zero_()

    def _hash(self, token: str) -> int:
        h = hashlib.md5(token.encode("utf-8")).hexdigest()
        return int(h, 16) % self.num_buckets

    def encode(self, texts: List[str]) -> torch.Tensor:
        device = self.emb.weight.device
        idx_lists = []
        for t in texts:
            toks = t.lower().strip().split()
            if not toks:
                toks = ["<pad>"]
            idxs = [self._hash(tok) for tok in toks]
            idx_lists.append(torch.tensor(idxs, dtype=torch.long))
        L = max(len(x) for x in idx_lists)
        batch = torch.stack([F.pad(x, (0, L - len(x)), value=self.pad_idx) for x in idx_lists])
        mask = (batch != self.pad_idx).float()
        emb = self.emb(batch.to(device))
        vec = (emb * mask.unsqueeze(-1)).sum(1) / (mask.sum(1, keepdim=True) + 1e-6)
        return F.normalize(vec, p=2, dim=1)


class MetricsEncoder(nn.Module):
    """1D-CNN encoder for time-series metrics."""
    def __init__(self, in_channels: int, hidden: int = 64, d_out: int = 128):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden, 7, padding=3)
        self.conv2 = nn.Conv1d(hidden, hidden, 5, padding=2)
        self.conv3 = nn.Conv1d(hidden, hidden, 3, padding=1)
        self.proj = nn.Linear(hidden, d_out)
        for m in [self.conv1, self.conv2, self.conv3]:
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):  # [B,T,C]
        x = x.transpose(1, 2)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = F.gelu(self.conv3(x))
        x = x.mean(2)
        x = self.proj(x)
        return F.normalize(x, p=2, dim=1)


class TraceEncoder(nn.Module):
    """GRU + Attention encoder for variable-length traces."""
    def __init__(self, num_services=64, num_ops=128, d_emb=64, d_num=16, d_hid=128, d_out=128):
        super().__init__()
        self.emb_svc = nn.Embedding(num_services, d_emb)
        self.emb_op = nn.Embedding(num_ops, d_emb)
        self.num_mlp = nn.Sequential(
            nn.Linear(2, d_num), nn.GELU(), nn.Linear(d_num, d_num)
        )
        self.gru = nn.GRU(d_emb * 2 + d_num, d_hid, batch_first=True, bidirectional=True)
        self.att_u = nn.Linear(d_hid * 2, 1)
        self.proj = nn.Linear(d_hid * 2, d_out)

    def forward(self, spans, lengths):
        B, L, _ = spans.shape
        svc = spans[:, :, 0].long().clamp_min(0)
        op = spans[:, :, 1].long().clamp_min(0)
        dur, err = spans[:, :, 2:3].float(), spans[:, :, 3:4].float()
        x = torch.cat(
            [self.emb_svc(svc), self.emb_op(op), self.num_mlp(torch.cat([dur, err], -1))],
            dim=-1,
        )
        mask = torch.arange(L, device=spans.device)[None, :] < lengths[:, None]
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=L)
        att = self.att_u(out).squeeze(-1)
        att = att.masked_fill(~mask, float("-inf"))
        att = torch.softmax(att, 1)
        rep = (out * att.unsqueeze(-1)).sum(1)
        rep = self.proj(rep)
        return F.normalize(rep, p=2, dim=1)


# =========================================================
# II. Adaptive Mid-Level Fusion (content + quality gating)
# =========================================================
class GatedMidFusionFromEmb(nn.Module):
    """Adaptive mid-level fusion with content–quality gating."""
    def __init__(self, d_log, d_met, d_trc, d_node, lam=0.5, dropout=0.0):
        super().__init__()
        self.lam = lam
        self.P_log, self.P_met, self.P_trc = nn.Linear(d_log, d_node), nn.Linear(d_met, d_node), nn.Linear(d_trc, d_node)
        self.u = nn.Parameter(torch.zeros(3))
        self.do = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        for m in [self.P_log, self.P_met, self.P_trc]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    @torch.no_grad()
    def content_score(self, e_log, e_met, e_trc):
        def _score(e): return e.var(dim=1, unbiased=False)
        c = torch.stack([
            _score(e_log).mean(1),
            _score(e_met).mean(1),
            _score(e_trc).mean(1)
        ], 1)
        return (c - c.min()) / (c.max() - c.min() + 1e-6)

    def forward(self, e_log, e_met, e_trc, quality):
        c = self.content_score(e_log, e_met, e_trc)
        s = self.lam * quality + (1 - self.lam) * c
        z_log, z_met, z_trc = self.P_log(e_log), self.P_met(e_met), self.P_trc(e_trc)
        score = torch.stack([
            (z_log * self.u[0]).sum(1) + s[:, 0],
            (z_met * self.u[1]).sum(1) + s[:, 1],
            (z_trc * self.u[2]).sum(1) + s[:, 2]
        ], 1)
        alpha = torch.softmax(score, 1)
        h = alpha[:, 0:1] * z_log + alpha[:, 1:2] * z_met + alpha[:, 2:3] * z_trc
        return self.do(h), {"alpha": alpha, "content": c}


# =========================================================
# III. Synthetic Dataset for Quick Testing
# =========================================================
def make_one_sample(num_classes=4, T=96, C=4, max_len=20, seed=None):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, num_classes)
    kw = rng.choice(["db", "cache", "queue", "auth"], size=3, replace=False)
    noise = rng.choice(["foo", "bar", "baz", "warn"], size=3)
    log_text = " ".join(list(kw) + list(noise))
    t = np.linspace(0, 2 * np.pi, T)
    base = np.stack([np.sin(0.1 * (y + 1) * (i + 1) * t) for i in range(C)], 1)
    metrics = base + rng.normal(0, 0.2, size=(T, C))
    L = int(rng.integers(8, max_len))
    spans = np.stack([
        rng.integers(0, 64, L),
        rng.integers(0, 128, L),
        np.clip(rng.normal(1.0, 0.5, L), 0.05, 5.0),
        rng.integers(0, 2, L)
    ], 1)
    q = np.array([0.8, 0.9, 0.85], np.float32)
    return {"log_text": log_text, "metrics": metrics.astype(np.float32),
            "trace_spans": spans.astype(np.float32), "y": int(y), "q": q}


class RawMultiModalDataset(torch.utils.data.Dataset):
    def __init__(self, N=6000, seed=7):
        rng = np.random.default_rng(seed)
        self.samples = [make_one_sample(seed=int(rng.integers(0, 1 << 31))) for _ in range(N)]

    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


def collate_raw(batch):
    logs = [b["log_text"] for b in batch]
    y = torch.tensor([b["y"] for b in batch])
    Q = torch.tensor(np.stack([b["q"] for b in batch]), dtype=torch.float32)
    M = torch.tensor(np.stack([b["metrics"] for b in batch]), dtype=torch.float32)
    Lmax = max(b["trace_spans"].shape[0] for b in batch)
    spans, lengths = [], []
    for b in batch:
        s = b["trace_spans"]
        pad = np.zeros((Lmax - s.shape[0], 4), np.float32)
        spans.append(torch.tensor(np.concatenate([s, pad]), dtype=torch.float32))
        lengths.append(s.shape[0])
    spans = torch.stack(spans)
    lengths = torch.tensor(lengths)
    return logs, M, spans, lengths, Q, y


# =========================================================
# IV. Model Assembly and Training
# =========================================================
class FusionFromRaw(nn.Module):
    def __init__(self, d_node=128, d_log=256, d_met=128, d_trc=128, metrics_inC=4):
        super().__init__()
        self.hash_enc = HashingTextEncoder(d_emb=d_log)
        self.met_enc = MetricsEncoder(in_channels=metrics_inC, d_out=d_met)
        self.trc_enc = TraceEncoder(d_out=d_trc)
        self.fuse = GatedMidFusionFromEmb(d_log, d_met, d_trc, d_node)
        self.head = nn.Linear(d_node, 4)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def encode_logs(self, texts): return self.hash_enc.encode(texts)

    def forward(self, logs_text, metrics, spans, lengths, quality):
        e_log = self.encode_logs(logs_text)
        e_met = self.met_enc(metrics)
        e_trc = self.trc_enc(spans, torch.clamp(lengths, 1))
        h, aux = self.fuse(e_log, e_met, e_trc, quality)
        logits = self.head(h)
        return logits, h, aux


def accuracy(logits, y): return (logits.argmax(1) == y).float().mean().item()


def train(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = RawMultiModalDataset(N=args.num_samples, seed=args.seed)
    n = len(ds)
    n_tr = int(n * 0.75)
    tr_loader = torch.utils.data.DataLoader(ds[:n_tr], batch_size=args.batch_size, shuffle=True, collate_fn=collate_raw)
    va_loader = torch.utils.data.DataLoader(ds[n_tr:], batch_size=args.batch_size, collate_fn=collate_raw)
    model = FusionFromRaw(d_node=args.d_node, d_log=args.d_log, d_met=args.d_met, d_trc=args.d_trc).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()

    best = 0.0
    for ep in range(1, args.epochs + 1):
        model.train()
        for logs, M, spans, lengths, Q, y in tr_loader:
            M, spans, lengths, Q, y = M.to(device), spans.to(device), lengths.to(device), Q.to(device), y.to(device)
            logits, _, _ = model(logs, M, spans, lengths, Q)
            loss = ce(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
        accs = []
        model.eval()
        with torch.no_grad():
            for logs, M, spans, lengths, Q, y in va_loader:
                M, spans, lengths, Q, y = M.to(device), spans.to(device), lengths.to(device), Q.to(device), y.to(device)
                logits, _, _ = model(logs, M, spans, lengths, Q)
                accs.append(accuracy(logits, y))
        mean_acc = np.mean(accs)
        best = max(best, mean_acc)
        print(f"[Epoch {ep}] val_acc={mean_acc:.3f}")
    print(f"[BEST] val_acc={best:.3f}")

    # Export embeddings + meta
    model.eval()
    logs, M, spans, lengths, Q, y = next(iter(va_loader))
    M, spans, lengths, Q = M.to(device), spans.to(device), lengths.to(device), Q.to(device)
    with torch.no_grad():
        _, H, aux = model(logs, M, spans, lengths, Q)
    out_dir = Path("outputs/perception"); out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"h": H.cpu(), "alpha": aux["alpha"].cpu(), "y": y}, out_dir / "mm_node_embeddings.pt")
    meta = {
        "num_samples": int(y.size(0)),
        "quality_mean": Q.mean().item(),
        "content_mean": aux["content"].mean().item(),
        "timestamp": __import__("time").strftime("%Y-%m-%d %H:%M:%S")
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[Perception] saved mm_node_embeddings.pt and meta.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_samples", type=int, default=6000)
    p.add_argument("--d_node", type=int, default=128)
    p.add_argument("--d_log", type=int, default=256)
    p.add_argument("--d_met", type=int, default=128)
    p.add_argument("--d_trc", type=int, default=128)
    args = p.parse_args()
    train(args)
