# data_rca.py
import math, random
from dataclasses import dataclass
from typing import Dict, List, Tuple
import torch

@dataclass
class RCAGraph:
    x: torch.Tensor
    y: torch.Tensor
    adj_hat: torch.Tensor
    edge_index_dir: torch.Tensor
    edge_time: torch.Tensor
    root: int
    chain_nodes: List[int]
    masks: Dict[str, torch.Tensor]

def _normalized_adj(num_nodes: int, undirected_edges: List[Tuple[int,int]]):
    if undirected_edges:
        idx_i, idx_j = zip(*undirected_edges)
        idx_i, idx_j = list(idx_i), list(idx_j)
    else:
        idx_i, idx_j = [], []
    i = torch.tensor(idx_i + list(range(num_nodes)), dtype=torch.long)
    j = torch.tensor(idx_j + list(range(num_nodes)), dtype=torch.long)
    v = torch.ones(i.numel(), dtype=torch.float32)
    A = torch.sparse_coo_tensor(torch.stack([i, j]), v, (num_nodes, num_nodes)).coalesce()
    deg = torch.sparse.sum(A, dim=1).to_dense()
    deg_inv_sqrt = (deg + 1e-8).pow(-0.5)
    row, col = A.indices()
    val = deg_inv_sqrt[row] * A.values() * deg_inv_sqrt[col]
    return torch.sparse_coo_tensor(torch.stack([row, col]), val, (num_nodes, num_nodes)).coalesce()

def _split_masks(N: int, y: torch.Tensor, seed=42, train=0.6, val=0.2):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(N, generator=g)
    n_tr = int(N * train); n_va = int(N * val)
    tr = torch.zeros(N, dtype=torch.bool); tr[idx[:n_tr]] = True
    va = torch.zeros(N, dtype=torch.bool); va[idx[n_tr:n_tr+n_va]] = True
    te = ~(tr | va)
    return {"train": tr, "val": va, "test": te}

def make_synthetic_rca(num_nodes=300, base_feat_dim=16, num_types=3,
                       avg_deg=3.0, chain_len=12, seed=123) -> RCAGraph:
    random.seed(seed); torch.manual_seed(seed)
    N = num_nodes
    types = torch.randint(low=0, high=num_types, size=(N,))
    type_onehot = torch.nn.functional.one_hot(types, num_types).float()
    base_feat = torch.randn(N, base_feat_dim)
    X = torch.cat([base_feat, type_onehot], dim=1)

    # 背景随机有向图
    E_bg = int(N * avg_deg)
    dir_edges = set()
    while len(dir_edges) < E_bg:
        u = random.randrange(N); v = random.randrange(N)
        if u != v:
            dir_edges.add((u, v))
    dir_edges = list(dir_edges)

    # 注入攻击链
    root = random.randrange(N)
    chain = [root]; used = {root}; cur = root
    for _ in range(chain_len - 1):
        for _try in range(50):
            v = random.randrange(N)
            if v != cur and v not in used:
                dir_edges.append((cur, v))
                chain.append(v); used.add(v); cur = v
                break
        else:
            break

    edge_index = torch.tensor(dir_edges, dtype=torch.long)  # [E,2]
    edge_time = torch.rand(edge_index.size(0))

    # —— 严格按链顺序设置时间递增（0.6→0.99）——
    if len(chain) > 1:
        order = torch.linspace(0.60, 0.99, steps=len(chain)-1)
        # 建立 (u,v)->idx 的查找
        e2idx = {(int(u), int(v)): i for i, (u, v) in enumerate(edge_index.tolist())}
        for k in range(len(chain)-1):
            u, v = chain[k], chain[k+1]
            idx = e2idx.get((u, v), None)
            if idx is not None:
                edge_time[idx] = order[k]

    # 标签
    y = torch.zeros(N, dtype=torch.long)
    if len(chain) > 1: y[chain[1:]] = 1
    y[root] = 2

    # 沿链注入特征签名
    sig = torch.randn(1, base_feat_dim) * 1.5
    for step, nid in enumerate(chain):
        X[nid, :base_feat_dim] += math.exp(-0.25 * step) * sig.squeeze(0)

    # GCN 用的无向 A_hat
    undirected = {(u, v) for (u, v) in dir_edges} | {(v, u) for (u, v) in dir_edges}
    A_hat = _normalized_adj(N, list(undirected))
    masks = _split_masks(N, y, seed=seed)

    return RCAGraph(
        x=X, y=y, adj_hat=A_hat,
        edge_index_dir=edge_index, edge_time=edge_time,
        root=root, chain_nodes=chain, masks=masks
    )
