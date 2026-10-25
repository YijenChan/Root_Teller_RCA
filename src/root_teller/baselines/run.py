from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from .data import load_specs, diagnostic_document, score_ranking
from .llm import LLM
from .methods import Copilot, RCAgent, LogEmbedder
from .native import MODEL_DEFAULTS, ThinkFL, RCLAgent, load_document


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Auditable paper-based RQ1 adaptations; not official author code")
    parser.add_argument("--method", required=True, choices=list(MODEL_DEFAULTS))
    from root_teller.paths import workspace_root
    parser.add_argument("--workspace", type=Path, default=workspace_root())
    parser.add_argument("--manifests", type=Path)
    parser.add_argument("--dataset", required=True, choices=["re2_ob", "re2_tt", "eadro_sn"])
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--mode", choices=["prepare", "live", "replay"], default="prepare")
    parser.add_argument("--model")
    parser.add_argument("--max-rounds", type=int, default=40)
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--rclagent-source", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-calls", type=int, default=200)
    parser.add_argument("--max-output-tokens", type=int, default=1800)
    parser.add_argument("--max-total-tokens", type=int, default=100000)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--embedding-model", default="thenlper/gte-large")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.model = args.model or MODEL_DEFAULTS[args.method]
    native = args.method in {"thinkfl", "rclagent"}
    if args.max_rounds < 1 or args.max_parallel < 1:
        parser.error("Investigation budgets must be positive")
    workspace = args.workspace.resolve()
    root = args.output / "runtime"
    # Run-local non-C: caches, with existing model paths passed explicitly when used.
    for name, suffix in {"TEMP": "tmp", "TMP": "tmp", "PIP_CACHE_DIR": "cache/pip",
                         "HF_HOME": "cache/huggingface", "TORCH_HOME": "cache/torch",
                         "TRANSFORMERS_CACHE": "cache/transformers"}.items():
        path = root / suffix
        path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(name, str(path))
    manifests = args.manifests or Path(__file__).resolve().parents[3] / "evaluation" / "rq1" / "manifests"
    specs = load_specs(workspace, manifests, args.dataset, args.fold)
    targets = sorted((x for x in specs if x.role == args.split), key=lambda x: x.opaque_id)
    if args.limit is not None:
        targets = targets[:args.limit]
    # Validation pilots use TRAIN only; formal outer-test uses train+validation.
    history_specs = [] if native else [x for x in specs if x.role == "train" or (args.split == "test" and x.role == "validation")]
    args.output.mkdir(parents=True, exist_ok=True)
    configuration = {"method": args.method, "dataset": args.dataset, "fold": args.fold,
                     "split": args.split, "seed": args.seed, "mode": args.mode, "model": args.model,
                     "embedding_model": args.embedding_model if args.method == "rcagent" else "gensim.FastText train-fold only" if args.method == "rcacopilot" else None,
                     "samples": args.samples if args.method == "rcagent" else None,
                     "max_calls": args.max_calls, "max_output_tokens": args.max_output_tokens,
                     "max_total_tokens": args.max_total_tokens,
                     "reasoning_effort": "minimal" if args.model.startswith("gpt-5") and "chat" not in args.model else None,
                     "temperature_control": "unsupported and omitted" if args.model.startswith("gpt-5") and "chat" not in args.model else "requested per trajectory",
                     "expected_cases": len(targets), "full_split_cases": sum(x.role == args.split for x in specs),
                     "history_cases": len(history_specs), "limited": args.limit is not None,
                     "implementation": "paper-reimplementation with benchmark/model adaptations",
                     "max_rounds": args.max_rounds, "max_parallel": args.max_parallel,
                     "rclagent_source": str(args.rclagent_source) if args.rclagent_source else None,
                     "python": sys.executable,
                     "manifest_sha256": hashlib.sha256((manifests / f"{args.dataset}_nested_folds.json").read_bytes()).hexdigest(),
                     "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}}
    previous = None
    if args.resume and (args.output / "config.json").exists():
        previous = json.loads((args.output / "config.json").read_text(encoding="utf-8"))
        for key in ["method", "dataset", "fold", "split", "seed", "model", "embedding_model", "samples", "max_rounds", "max_parallel", "rclagent_source", "manifest_sha256", "source_sha256"]:
            if previous.get(key) != configuration.get(key):
                raise ValueError(f"Resume configuration changed: {key}; use a new output directory")
    dump(args.output / "config.json", configuration)
    target_documents = {}
    for i, spec in enumerate(targets, 1):
        target_documents[spec.opaque_id] = (load_document if native else diagnostic_document)(spec, workspace, root / "cache")
        print(f"Prepared target {i}/{len(targets)} {spec.opaque_id}", flush=True)
    if args.mode == "prepare":
        dump(args.output / "summary.json", {"status": "completed_pipeline_only", "stage": "telemetry preparation",
                                             "metrics": None, "configuration": configuration})
        return
    history = []
    for i, spec in enumerate(history_specs, 1):
        document = diagnostic_document(spec, workspace, root / "cache")
        history.append((document, spec.target))
        if i % 5 == 0 or i == len(history_specs):
            print(f"Prepared development history {i}/{len(history_specs)}", flush=True)
    configuration["history_library_sha256"] = hashlib.sha256(
        json.dumps([(d["opaque_id"], hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest(), label)
                    for d, label in history], sort_keys=True).encode()).hexdigest()
    if args.resume and previous is not None and previous.get("history_library_sha256") != configuration["history_library_sha256"]:
        raise ValueError("Resume historical data library changed; use a new output directory")
    dump(args.output / "config.json", configuration)
    llm = LLM(workspace, root / "cache" / args.method, args.model, args.mode, args.max_calls, seed=args.seed,
              max_output_tokens=args.max_output_tokens, max_total_tokens=args.max_total_tokens)
    if args.method == "thinkfl":
        algorithm = ThinkFL(llm, args.max_rounds)
    elif args.method == "rclagent":
        algorithm = RCLAgent(llm, args.rclagent_source, args.max_parallel)
    elif args.method == "rcacopilot":
        algorithm = Copilot(history, llm, args.seed)
    else:
        algorithm = RCAgent(llm, LogEmbedder(args.embedding_model), history, seed=args.seed, samples=args.samples)
    predictions = []
    for i, spec in enumerate(targets, 1):
        destination = args.output / "cases" / f"{spec.opaque_id}.json"
        prior_case = json.loads(destination.read_text(encoding="utf-8")) if destination.exists() else None
        if args.resume and prior_case and prior_case["status"] == "completed":
            current_document_hash = hashlib.sha256(json.dumps(target_documents[spec.opaque_id], sort_keys=True).encode()).hexdigest()
            if prior_case.get("document_sha256") != current_document_hash:
                raise ValueError("Resume target diagnostic input changed; use a new output directory")
            result = prior_case
        else:
            begin = time.perf_counter()
            before, cached_before, tokens_before = llm.calls, llm.cache_hits, llm.tokens
            request_offset = len(llm.requests)
            try:
                ranking, diagnostics = algorithm.predict(target_documents[spec.opaque_id])
                result = {"status": "completed", "ranking": ranking, "diagnostics": diagnostics}
            except Exception as exc:
                # Failures receive no anomaly-based or alphabetical fallback ranking.
                result = {"status": "failed", "ranking": [], "error_type": type(exc).__name__,
                          "error": str(exc)[:200] if type(exc).__name__ in {"InferenceError", "ValueError"} else "See error type"}
                if args.method == "rcagent":
                    result["diagnostics"] = {"partial_trace": algorithm.last_trace,
                                             "snapshot_store": algorithm.snapshots}
            result.update({"opaque_id": spec.opaque_id, "dataset": spec.dataset, "fold": spec.fold,
                           "document_sha256": hashlib.sha256(json.dumps(target_documents[spec.opaque_id], sort_keys=True).encode()).hexdigest(),
                           "private_incident_id": spec.incident_id, "private_target": spec.target,
                           "elapsed_seconds": time.perf_counter()-begin, "physical_calls": llm.calls-before,
                           "cache_hits": llm.cache_hits-cached_before, "physical_tokens": llm.tokens-tokens_before,
                           "api_requests": llm.requests[request_offset:], "evaluation": score_ranking(result["ranking"], spec.target)})
            if prior_case is not None:
                old = args.output / "attempts" / f"{spec.opaque_id}-{time.time_ns()}.json"
                dump(old, prior_case)
            dump(destination, result)
        predictions.append(result)
        print(f"Case {i}/{len(targets)} {spec.opaque_id} status={result['status']} rank={result['evaluation']['rank']}", flush=True)
        metrics = {name: sum(r["evaluation"][name] for r in predictions)/len(targets)
                   for name in ["A@1", "A@3", "Avg@5"]}
        interface_blocked = result["status"] == "failed" and result.get("error_type") == "InferenceError"
        dump(args.output / "summary.json", {"status": "blocked_interface" if interface_blocked else "running" if len(predictions) < len(targets) else
                                             "completed" if all(r["status"] == "completed" for r in predictions) else "completed_with_failures",
              "scope": "limited pilot, not paper results" if args.limit is not None or args.split != "test" else "one registered outer fold",
              "metrics": metrics, "evaluated_cases": len(predictions), "expected_cases": len(targets),
              "failed_cases": sum(r["status"] != "completed" for r in predictions),
              "physical_calls": llm.calls, "physical_tokens": llm.tokens, "cache_hits": llm.cache_hits,
              "configuration": configuration})
        if interface_blocked:
            print("Stopped on model-interface failure; unprocessed cases are not fabricated predictions", flush=True)
            break


if __name__ == "__main__":
    main()
