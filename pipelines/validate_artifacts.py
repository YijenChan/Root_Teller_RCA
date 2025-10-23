# pipelines/validate_artifacts.py
# 检查 outputs 中关键产物是否就绪

import os, sys, argparse, json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perception", default=os.path.join(ROOT, "outputs", "perception", "mm_node_embeddings.pt"))
    ap.add_argument("--rgat_model", default=os.path.join(ROOT, "outputs", "reasoning", "rgat_time_min.pt"))
    ap.add_argument("--rgat_metrics", default=os.path.join(ROOT, "outputs", "reasoning", "single_case.json"))
    ap.add_argument("--report", default=os.path.join(ROOT, "outputs", "reports", "single_case_report.txt"))
    args = ap.parse_args()

    checks = {
        "perception/mm_node_embeddings.pt": os.path.exists(args.perception),
        "reasoning/rgat_time_min.pt": os.path.exists(args.rgat_model),
        "reasoning/single_case.json": os.path.exists(args.rgat_metrics),
        "reports/single_case_report.txt": os.path.exists(args.report)
    }

    print("Artifact checks:")
    for k, ok in checks.items():
        print(f" - {k:<35} : {'OK' if ok else 'MISSING'}")

    out_json = os.path.join(ROOT, "outputs", "validate_artifacts.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(checks, f, indent=2)
    print(f"Saved: {out_json}")

if __name__ == "__main__":
    main()
