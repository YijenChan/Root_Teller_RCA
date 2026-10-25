from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from root_teller.paths import workspace_root
from .classical import SOURCE_ROOT, causal_inputs, causal_rank, healthy_captures, nezha_rank, read_capture, slice_capture
from .data import load_specs, score_ranking


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["nezha", "multisource_rcd", "torai"])
    parser.add_argument("--dataset", required=True, choices=["re2_ob", "re2_tt", "eadro_sn"])
    parser.add_argument("--workspace", type=Path, default=workspace_root())
    parser.add_argument("--manifests", type=Path)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--mode", choices=["prepare", "run"], default="prepare")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new output directory")
    workspace = args.workspace.resolve()
    manifests = args.manifests or Path(__file__).resolve().parents[3] / "evaluation/rq1/manifests"
    specs = load_specs(workspace, manifests, args.dataset, args.fold)
    targets = sorted((s for s in specs if s.role == args.split), key=lambda s: s.opaque_id)
    if not targets:
        raise ValueError("Requested split is empty")
    configuration = {"method": args.method, "dataset": args.dataset, "fold": args.fold, "split": args.split,
        "mode": args.mode, "expected_cases": len(targets), "input_modalities": ["metric", "log"] if args.method == "multisource_rcd" else
            ["log", "trace"] if args.method == "nezha" else ["metric", "log", "trace"],
        "manifest_sha256": hashlib.sha256((manifests / (args.dataset + "_nested_folds.json")).read_bytes()).hexdigest(),
        "upstream_source_sha256": {str(p.relative_to(SOURCE_ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (SOURCE_ROOT / args.method).rglob("*.py")},
        "reference": "released independent SN no-fault captures" if args.dataset == "eadro_sn" else "first half of predefined observation range",
        "uses_injection_time_at_inference": False, "uses_labels_at_inference": False}
    dump(args.output / "config.json", configuration)
    reference = healthy_captures(workspace) if args.dataset == "eadro_sn" else None
    rows = []
    for spec in targets:
        capture = read_capture(spec.directory, args.dataset, spec.start, spec.end)
        if reference is None:
            middle = (capture.start + capture.end) / 2
            current_reference = [slice_capture(capture, capture.start, middle)]
            production = slice_capture(capture, middle, capture.end)
        else:
            current_reference, production = reference, capture
        if args.mode == "prepare":
            data, boundary = causal_inputs(current_reference, production)
            rows.append({"opaque_id": spec.opaque_id, "reference_samples": int(boundary),
                         "production_samples": len(data["metric"]) - int(boundary)})
            continue
        started = time.perf_counter()
        try:
            ranking, diagnostics = (nezha_rank(current_reference, production) if args.method == "nezha" else
                                    causal_rank(args.method, current_reference, production))
            result = {"status": "completed", "ranking": ranking, "diagnostics": diagnostics}
        except Exception as error:
            result = {"status": "failed", "ranking": [], "error_type": type(error).__name__, "error": str(error)[:200]}
        result.update(opaque_id=spec.opaque_id, private_incident_id=spec.incident_id, private_target=spec.target,
                      elapsed_seconds=time.perf_counter() - started, evaluation=score_ranking(result["ranking"], spec.target))
        dump(args.output / "cases" / (spec.opaque_id + ".json"), result)
        rows.append(result)
        print(json.dumps({"opaque_id": spec.opaque_id, "status": result["status"]}), flush=True)
    metrics = None if args.mode == "prepare" else {
        key: sum(r["evaluation"][key] for r in rows) / len(targets) for key in ["A@1", "A@3", "Avg@5"]}
    dump(args.output / "summary.json", {"status": "prepared_inputs" if args.mode == "prepare" else
        "completed" if all(r["status"] == "completed" for r in rows) else "completed_with_failures",
        "metrics": metrics, "configuration": configuration, "cases": rows})


if __name__ == "__main__":
    main()
