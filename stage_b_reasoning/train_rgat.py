# train_rgat.py
import torch
import torch.nn as nn
import torch.optim as optim

# --- Import guards: package vs single-file execution ---
try:
    # package usage: from stage_b_reasoning import train_rgat as rgat
    from .data_rca import make_synthetic_rca
    from .model_rgat import RGATTimeNet
except ImportError:
    # standalone: python stage_b_reasoning/train_rgat.py
    from data_rca import make_synthetic_rca
    from model_rgat import RGATTimeNet


# ---------------------------------------------------------------------
# Class reweighting (boost "chain" and "root" classes)
# ---------------------------------------------------------------------
def class_weights(y: torch.Tensor, boost_root=6.0, boost_chain=2.0):
    counts = torch.bincount(y, minlength=3).float()
    w = (counts.sum() / (counts + 1e-8))
    w = w / w.mean()
    w[1] *= boost_chain  # chain
    w[2] *= boost_root   # root
    return w


# ---------------------------------------------------------------------
# Build augmented edges for training (fwd + bwd + self)
# Returns: edge_index_aug, edge_time_aug, edge_dir_onehot
# ---------------------------------------------------------------------
def augment_edges_for_rgat(edge_index, edge_time, num_nodes):
    # forward
    ei_f, et_f = edge_index, edge_time
    ea_f = torch.tensor([[1., 0., 0.]], dtype=torch.float32).repeat(ei_f.size(0), 1)
    # backward
    ei_b = torch.stack([edge_index[:, 1], edge_index[:, 0]], dim=1)
    et_b = edge_time * 0.98
    ea_b = torch.tensor([[0., 1., 0.]], dtype=torch.float32).repeat(ei_b.size(0), 1)
    # self-loop
    idx = torch.arange(num_nodes, dtype=torch.long)
    ei_s = torch.stack([idx, idx], dim=1)
    et_s = torch.ones(num_nodes, dtype=torch.float32)
    ea_s = torch.tensor([[0., 0., 1.]], dtype=torch.float32).repeat(num_nodes, 1)

    edge_index_aug = torch.cat([ei_f, ei_b, ei_s], dim=0)
    edge_time_aug = torch.cat([et_f, et_b, et_s], dim=0)
    edge_dir = torch.cat([ea_f, ea_b, ea_s], dim=0)  # one-hot for {fwd,bwd,self}
    return edge_index_aug, edge_time_aug, edge_dir


# ---------------------------------------------------------------------
# Chain reconstruction (greedy along time-forward edges)
# ---------------------------------------------------------------------
def reconstruct_chain_greedy(pred_root, edge_index_dir, edge_time, prob_infected, max_len=16):
    N = prob_infected.numel()
    out_map = [[] for _ in range(N)]
    for e, (u, v) in enumerate(edge_index_dir.tolist()):
        out_map[u].append((v, edge_time[e].item(), e))
    chain = [pred_root]
    visited = {pred_root}
    cur, cur_t = pred_root, -1.0
    for _ in range(max_len - 1):
        cands = [(v, t) for (v, t, _) in out_map[cur] if t > cur_t and v not in visited]
        if not cands:
            break
        # pick by highest infection probability
        v_best, t_best = max(cands, key=lambda vt: prob_infected[vt[0]].item())
        chain.append(v_best)
        visited.add(v_best)
        cur, cur_t = v_best, t_best
    return chain


def jaccard(a, b):
    sa, sb = set(a), set(b)
    return len(sa & sb) / (len(sa | sb) + 1e-8)


# ---------------------------------------------------------------------
# Evaluation with evidence features (φ_i, τ_i)
# ---------------------------------------------------------------------
def evaluate(model, g, device, topk=5):
    model.eval()
    with torch.no_grad():
        E = g.edge_index_dir.to(device)               # use forward edges for evaluation
        T = g.edge_time.to(device)
        DIR = torch.tensor([[1., 0., 0.]], dtype=torch.float32, device=device).repeat(E.size(0), 1)

        logits, att, ei_used, et_used = model(g.x.to(device), E, T, edge_attr=None, edge_dir=DIR)
        probs = torch.softmax(logits, dim=1).cpu()
        y_pred = probs.argmax(dim=1)

        mask = g.masks["test"]
        acc = (y_pred[mask] == g.y[mask]).float().mean().item()

        # root ranking
        root_scores = probs[:, 2]
        pred_root = int(torch.argmax(root_scores).item())
        topk_vals, topk_idx = torch.topk(root_scores, k=min(topk, root_scores.numel()))
        root_hit1 = float(pred_root == g.root)
        root_hitk = float((topk_idx == g.root).any().item())

        # greedy chain
        chain_pred = reconstruct_chain_greedy(
            pred_root, g.edge_index_dir, g.edge_time, probs[:, 1], max_len=len(g.chain_nodes) + 2
        )
        jac = jaccard(chain_pred, g.chain_nodes)

        # ----- evidence features -----
        # attention from the last RGAT layer
        N = g.x.size(0)
        phi = RGATTimeNet.compute_source_likeness(att, ei_used, N).cpu()           # source-likeness
        tau = RGATTimeNet.compute_temporal_lead(et_used, ei_used, N).cpu()         # temporal lead
        a_i = root_scores.cpu()

        evidence_pack = {
            "nodes": [
                {"id": int(i), "score_root": float(a_i[i]), "phi": float(phi[i]), "tau": float(tau[i])}
                for i in range(N)
            ],
            "topk_roots": [int(i) for i in topk_idx.cpu().tolist()],
            "topk_scores": [float(v) for v in topk_vals.cpu().tolist()],
            "pred_root": pred_root,
            "pred_chain": [int(v) for v in chain_pred],
            "true_root": int(g.root),
            "true_chain": [int(v) for v in g.chain_nodes],
        }

    return {
        "test_acc": acc,
        "root_hit@1": root_hit1,
        "root_hit@5": root_hitk,
        "chain_jaccard": jac,
        "pred_root": pred_root,
        "topk_roots": topk_idx.tolist(),
        "topk_scores": topk_vals.tolist(),
        "chain_pred": chain_pred,
        "evidence_pack": evidence_pack,
    }


# ---------------------------------------------------------------------
# Training entry
# ---------------------------------------------------------------------
def train(epochs=200, lr=1e-3, hidden=128, heads=4, dropout=0.1, time_dim=8,
          num_nodes=400, base_feat_dim=16, chain_len=10, seed=7, device=None):
    torch.manual_seed(seed)
    device = "cuda"  # force CUDA to match pipeline expectation
    assert torch.cuda.is_available(), "CUDA is required but not available."

    g = make_synthetic_rca(num_nodes=num_nodes, base_feat_dim=base_feat_dim,
                           chain_len=chain_len, seed=seed)
    in_dim = g.x.size(1)

    # edge_dim=3 for {fwd, bwd, self}
    model = RGATTimeNet(in_dim, hidden=hidden, num_classes=3, heads=heads,
                        time_dim=time_dim, edge_dim=3, dropout=dropout).to(device)

    ce = nn.CrossEntropyLoss(weight=class_weights(g.y).to(device))
    opt = optim.AdamW(model.parameters(), lr=lr)

    # ensure the unique "root" node is seen in training
    tr = g.masks["train"].clone()
    va = g.masks["val"].clone()
    te = g.masks["test"].clone()
    tr[g.root] = True
    va[g.root] = False
    te[g.root] = False

    # train with augmented edges (fwd+bwd+self), pass direction one-hot
    E_aug, T_aug, DIR_aug = augment_edges_for_rgat(g.edge_index_dir, g.edge_time, g.x.size(0))
    X = g.x.to(device)
    E_aug, T_aug, DIR_aug = E_aug.to(device), T_aug.to(device), DIR_aug.to(device)
    y = g.y.to(device)
    tr, va = tr.to(device), va.to(device)

    best_val, best = -1.0, None
    for ep in range(1, epochs + 1):
        model.train()
        logits, _, _, _ = model(X, E_aug, T_aug, edge_attr=None, edge_dir=DIR_aug)
        loss = ce(logits[tr], y[tr])

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if ep % 20 == 0 or ep == 1:
            with torch.no_grad():
                val_acc = (logits[va].argmax(dim=1) == y[va]).float().mean().item()
            if val_acc > best_val:
                best_val = val_acc
                best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"epoch {ep:03d} | train_loss {loss.item():.4f} | val_acc {val_acc:.3f}")

    if best is not None:
        model.load_state_dict(best)

    res = evaluate(model, g, device, topk=5)
    print(f"[TEST] acc={res['test_acc']:.3f} | root_hit@1={res['root_hit@1']:.3f} | "
          f"root_hit@5={res['root_hit@5']:.3f} | chain_jaccard={res['chain_jaccard']:.3f}")
    print(f"true_root={g.root} | pred_root={res['pred_root']} | top5_roots={res['topk_roots']}")
    print(f"true_chain={g.chain_nodes}")
    print(f"pred_chain={res['chain_pred']}")

    torch.save(model.state_dict(), "rgat_time_min.pt")
    print("Saved model to rgat_time_min.pt")

    return res  # allow caller (pipeline) to write evidence_pack JSON


if __name__ == "__main__":
    train()
