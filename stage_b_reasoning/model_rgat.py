# -*- coding: utf-8 -*-
"""
model_rgat.py
Enhanced R-GAT(+time) with direction/time bias and source-likeness utilities.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------------
# Time Encoding (sin–cos)
# -------------------------------------------------------------
def time_encode(t: torch.Tensor, dim: int = 8):
    t = t.view(-1, 1)
    k = torch.arange(1, dim // 2 + 1, device=t.device, dtype=t.dtype).view(1, -1)
    ang = 2.0 * math.pi * t * k
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)  # [E, dim]


# -------------------------------------------------------------
# Relational GAT layer with time & edge bias
# -------------------------------------------------------------
class RelGATTimeLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=4, time_dim=8, edge_dim=0, dropout=0.1):
        super().__init__()
        assert out_dim % num_heads == 0
        self.h = num_heads
        self.d = out_dim // num_heads
        self.time_dim = time_dim
        self.edge_dim = edge_dim
        self.dropout = nn.Dropout(dropout)

        # projection
        self.Wq = nn.Linear(in_dim, out_dim, bias=False)
        self.Wk = nn.Linear(in_dim, out_dim, bias=False)
        self.Wv = nn.Linear(in_dim, out_dim, bias=False)

        # edge features (type + time + direction) → head bias
        self.edge_in = edge_dim + time_dim + 3  # add 3 for direction one-hot
        self.edge_mlp = nn.Sequential(
            nn.Linear(self.edge_in, self.h),
            nn.Tanh()
        )

        self.a = nn.Parameter(torch.empty(self.h, 2 * self.d))
        nn.init.xavier_uniform_(self.a)

        self.out_proj = nn.Linear(out_dim, out_dim)
        self.res_proj = nn.Linear(in_dim, out_dim)
        self.ln = nn.LayerNorm(out_dim)
        self.act = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x, edge_index, edge_time, edge_attr=None, edge_dir=None):
        """
        x: [N, in]
        edge_index: [E, 2] (u -> v)
        edge_time: [E]
        edge_attr: [E, D_e] optional
        edge_dir:  [E, 3] one-hot for {fwd,bwd,self}
        """
        N = x.size(0)
        E = edge_index.size(0)
        device = x.device

        Q = self.Wq(x).view(N, self.h, self.d)
        K = self.Wk(x).view(N, self.h, self.d)
        V = self.Wv(x).view(N, self.h, self.d)

        src, dst = edge_index[:, 0].long(), edge_index[:, 1].long()
        Qdst, Ksrc, Vsrc = Q[dst], K[src], V[src]

        # time encoding + edge attribute + direction one-hot
        t_enc = time_encode(edge_time.to(x.dtype), dim=self.time_dim)
        parts = [t_enc]
        if edge_attr is not None:
            parts.append(edge_attr.to(x.dtype))
        if edge_dir is not None:
            parts.append(edge_dir.to(x.dtype))
        e_feat = torch.cat(parts, dim=1)
        e_bias = self.edge_mlp(e_feat)  # [E, h]

        # attention logits per head
        a = self.a.view(self.h, 2 * self.d)
        cat = torch.cat([Qdst, Ksrc], dim=2)
        logits = (cat * a).sum(dim=2) + e_bias
        logits = self.leaky_relu(logits)

        # stable softmax per dst node
        alphas = torch.zeros(E, self.h, device=device)
        for h in range(self.h):
            l = logits[:, h]
            l = l - l.max()
            exp_l = torch.exp(l).clamp_max(1e6)
            denom = torch.zeros(N, device=device)
            denom.index_add_(0, dst, exp_l)
            alphas[:, h] = exp_l / (denom[dst] + 1e-12)

        # message aggregation
        out = torch.zeros(N, self.h, self.d, device=device)
        for h in range(self.h):
            m = Vsrc[:, h, :] * alphas[:, h].unsqueeze(-1)
            out[:, h, :].index_add_(0, dst, m)
        out = out.reshape(N, self.h * self.d)

        out = self.dropout(self.out_proj(out))
        y = self.ln(out + self.res_proj(x))
        return self.act(y), alphas.detach(), edge_index, edge_time


# -------------------------------------------------------------
# RGAT network with two layers
# -------------------------------------------------------------
class RGATTimeNet(nn.Module):
    def __init__(self, in_dim, hidden=128, num_classes=3, heads=4,
                 time_dim=8, edge_dim=0, dropout=0.1):
        super().__init__()
        self.l1 = RelGATTimeLayer(in_dim, hidden, num_heads=heads,
                                  time_dim=time_dim, edge_dim=edge_dim,
                                  dropout=dropout)
        self.l2 = RelGATTimeLayer(hidden, hidden, num_heads=heads,
                                  time_dim=time_dim, edge_dim=edge_dim,
                                  dropout=dropout)
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, x, edge_index, edge_time, edge_attr=None, edge_dir=None):
        h, att1, ei1, et1 = self.l1(x, edge_index, edge_time, edge_attr, edge_dir)
        h, att2, ei2, et2 = self.l2(h, edge_index, edge_time, edge_attr, edge_dir)
        logits = self.classifier(h)
        return logits, att2, ei2, et2

    # ---------------------------------------------------------
    # Utilities for reasoning analysis
    # ---------------------------------------------------------
    @staticmethod
    def compute_source_likeness(att, edge_index, N):
        """Compute source-likeness φ_i = Σ_out α."""
        src = edge_index[:, 0].long()
        phi = torch.zeros(N, device=att.device)
        phi.index_add_(0, src, att.mean(dim=1))
        return phi

    @staticmethod
    def compute_temporal_lead(edge_time, edge_index, N):
        """Compute temporal lead τ_i (avg outgoing Δt)."""
        src, dst = edge_index[:, 0], edge_index[:, 1]
        dt = torch.zeros(N, device=edge_time.device)
        cnt = torch.zeros(N, device=edge_time.device)
        diff = edge_time[src] - edge_time[dst]
        dt.index_add_(0, src, diff)
        cnt.index_add_(0, src, torch.ones_like(diff))
        tau = dt / (cnt + 1e-6)
        return tau
