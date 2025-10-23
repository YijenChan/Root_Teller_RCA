# -*- coding: utf-8 -*-
# agents/graph_reasoner_agent.py
import os, sys, json, shutil, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agents.base import BaseAgent, Message
from stage_b_reasoning import train_rgat as rgat
from stage_b_reasoning.model_rgat import RGATTimeNet
from stage_b_reasoning.data_rca import make_synthetic_rca

def _to_float(x, d):
    try: return float(x)
    except: return float(d)

def _to_int(x, d):
    try: return int(float(x))
    except: return int(d)

class GraphReasonerAgent(BaseAgent):
    """
    职责：训练 R-GAT + 评测；落盘 model 与 metrics
    输入 payload:
      - cfg_run: dict（含 epochs_b/num_nodes/base_feat_dim/chain_len 等）
    返回:
      { ok: True, data: {"model_path":..., "metrics_path":..., "report_path":...} }
    """
    def handle(self, msg: Message):
        return self.reply_err(f"unknown message type: {msg.type}")

    def on_train_and_eval(self, msg: Message):
        run = msg.payload.get("cfg_run", {}) or {}
        device = "cuda"
        cfg = {
            "epochs":       _to_int(run.get("epochs_b", 120), 120),
            "lr":           _to_float(run.get("lr_b", 1e-3), 1e-3),
            "hidden":       _to_int(run.get("hidden_b", 128), 128),
            "heads":        _to_int(run.get("heads_b", 4), 4),
            "dropout":      _to_float(run.get("dropout_b", 0.1), 0.1),
            "time_dim":     _to_int(run.get("time_dim_b", 8), 8),
            "num_nodes":    _to_int(run.get("num_nodes_b", 400), 400),
            "base_feat_dim":_to_int(run.get("base_feat_dim_b", 16), 16),
            "chain_len":    _to_int(run.get("chain_len_b", 10), 10),
            "seed":         _to_int(run.get("seed", 7), 7),
            "device":       device,
        }
        self.log(f"Stage-B training... epochs={cfg['epochs']} lr={cfg['lr']}")
        rgat.train(**cfg)

        out_reason = os.path.join(ROOT, "outputs", "reasoning")
        out_reports = os.path.join(ROOT, "outputs", "reports")
        os.makedirs(out_reason, exist_ok=True)
        os.makedirs(out_reports, exist_ok=True)

        src_model = os.path.join(os.getcwd(), "rgat_time_min.pt")
        dst_model = os.path.join(out_reason, "rgat_time_min.pt")
        if os.path.exists(src_model):
            shutil.copy2(src_model, dst_model)
        else:
            dst_model = src_model  # 已经写在目标目录

        # 构建同分布图，重建模型并评测
        g = make_synthetic_rca(
            num_nodes=cfg["num_nodes"],
            base_feat_dim=cfg["base_feat_dim"],
            chain_len=cfg["chain_len"],
            seed=cfg["seed"]
        )
        in_dim = g.x.size(1)
        model = RGATTimeNet(
            in_dim, hidden=cfg["hidden"], num_classes=3,
            heads=cfg["heads"], time_dim=cfg["time_dim"], edge_dim=3, dropout=cfg["dropout"]
        ).to(device)

        try:
            state = torch.load(dst_model, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(dst_model, map_location=device)
        model.load_state_dict(state)

        res = rgat.evaluate(model, g, device=device, topk=5)

        metrics_path = os.path.join(out_reason, "single_case.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({"cfg": cfg, "result": res, "true_root": g.root, "true_chain": g.chain_nodes},
                      f, ensure_ascii=False, indent=2)

        # 简要文本摘要（与 pipelines/run_single_case 保持一致）
        report_txt = os.path.join(out_reports, "single_case_report.txt")
        with open(report_txt, "w", encoding="utf-8") as f:
            f.write(
                "R-GAT(+time) Single-Case Summary\n"
                "=================================\n"
                f"test_acc       : {res['test_acc']:.3f}\n"
                f"root_hit@1     : {res['root_hit@1']:.3f}\n"
                f"root_hit@5     : {res['root_hit@5']:.3f}\n"
                f"chain_jaccard  : {res['chain_jaccard']:.3f}\n"
                f"true_root      : {g.root}\n"
                f"pred_root      : {res['pred_root']}\n"
                f"true_chain     : {g.chain_nodes}\n"
                f"pred_chain     : {res['chain_pred']}\n"
                f"top5_roots     : {res['topk_roots']}\n"
            )

        self.log(f"saved model -> {dst_model}")
        self.log(f"saved metrics -> {metrics_path}")
        self.log(f"saved text report -> {report_txt}")
        return self.reply_ok({"model_path": dst_model, "metrics_path": metrics_path, "report_path": report_txt})
