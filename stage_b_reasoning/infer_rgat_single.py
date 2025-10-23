# stage_b_reasoning/infer_rgat_single.py
# 载入已训练的 R-GAT 权重，对合成图做一次独立推理评估

import os, sys, argparse, json, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from stage_b_reasoning.model_rgat import RGATTimeNet
from stage_b_reasoning.data_rca import make_synthetic_rca
from stage_b_reasoning import train_rgat as rgat

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "outputs", "reasoning", "rgat_time_min.pt"))
    ap.add_argument("--device", default="auto", choices=["auto","cuda","cpu"])
    ap.add_argument("--num_nodes", type=int, default=400)
    ap.add_argument("--base_feat_dim", type=int, default=16)
    ap.add_argument("--chain_len", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--time_dim", type=int, default=8)
    ap.add_argument("--save_json", default=os.path.join(ROOT, "outputs", "reasoning", "infer_single.json"))
    args = ap.parse_args()

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else (args.device if args.device!="auto" else "cpu")

    # 构建与训练时一致的合成图
    g = make_synthetic_rca(
        num_nodes=args.num_nodes,
        base_feat_dim=args.base_feat_dim,
        chain_len=args.chain_len,
        seed=args.seed
    )

    in_dim = g.x.size(1)
    model = RGATTimeNet(
        in_dim, hidden=args.hidden, num_classes=3,
        heads=args.heads, time_dim=args.time_dim, edge_dim=3, dropout=args.dropout
    ).to(device)

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"model not found: {args.model}")

    # 原来：
    # state = torch.load(args.model, map_location=device)

    # 建议改为：
    try:
        state = torch.load(args.model, map_location=device, weights_only=True)
    except TypeError:
        # 旧版 PyTorch 不支持 weights_only 参数时的回退
        state = torch.load(args.model, map_location=device)

    model.load_state_dict(state)

    res = rgat.evaluate(model, g, device=device, topk=5)

    os.makedirs(os.path.dirname(args.save_json), exist_ok=True)
    with open(args.save_json, "w", encoding="utf-8") as f:
        json.dump({
            "cfg": vars(args),
            "result": res,
            "true_root": g.root,
            "true_chain": g.chain_nodes
        }, f, ensure_ascii=False, indent=2)

    print(f"Saved: {args.save_json}")
    print(f"[TEST] acc={res['test_acc']:.3f} | root_hit@1={res['root_hit@1']:.3f} | root_hit@5={res['root_hit@5']:.3f} | chain_jaccard={res['chain_jaccard']:.3f}")

if __name__ == "__main__":
    main()
