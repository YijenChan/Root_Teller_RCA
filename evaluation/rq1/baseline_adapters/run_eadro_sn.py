"""Common-protocol baseline evaluation on the clean Eadro-SN dataset.

The 36 annotated intervals are evaluated with grouped four-fold
leave-one-capture-out, matching Root-Teller's Eadro-SN protocol.  Fault
annotations define interval boundaries and evaluator labels only; no fault
name or fault type is exposed to any ranker.
"""
from __future__ import annotations

import os

import argparse
import hashlib
import importlib.util
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import requests
import torch
from torch.nn import functional as F


PROJECT = Path(os.environ.get("ROOTTELLER_WORKSPACE", Path.cwd())).expanduser().resolve()
sys.path.insert(0, str(PROJECT / "src"))

from root_teller.module1.data import normalize_case  # noqa: E402
from root_teller.multidataset import rq1_sn  # noqa: E402


SERVICES = rq1_sn.services()
SERVICE_INDEX = {service: index for index, service in enumerate(SERVICES)}
RUN_ROOT = PROJECT / "baselines"
RUN_ID = "eadro_sn_clean_grouped_loco_2026-07-25"


def actor_config() -> tuple[str, str]:
    key = os.environ["ROOTTELLER_API_KEY"]
    base = os.environ.get("ROOTTELLER_API_BASE", "https://api.openai.com/v1")
    return key, base.rstrip("/")
def percentile(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    unique = np.unique(values)
    rank = {value: index for index, value in enumerate(unique)}
    return np.asarray(
        [rank[value] / max(len(unique) - 1, 1) for value in values],
        dtype=np.float32,
    )


def modality_scores(case: dict, reference) -> dict[str, np.ndarray]:
    normalized = normalize_case(case, reference)
    output = {}
    for name, key in (
        ("metric", "metric_x"),
        ("log", "log_x"),
        ("trace", "trace_x"),
    ):
        values = np.asarray(normalized[key], dtype=np.float32)
        severity = np.quantile(np.abs(values), 0.95, axis=(1, 2))
        output[name] = percentile(severity)
    return output


def topology(case: dict) -> dict[str, list[str]]:
    graph: dict[str, set[str]] = {}
    for source, target in case.get("edges", ()):
        if source in SERVICE_INDEX and target in SERVICE_INDEX and source != target:
            graph.setdefault(source, set()).add(target)
    return {source: sorted(targets) for source, targets in graph.items()}


def order_from(scores: np.ndarray, incident_id: str = "") -> list[str]:
    tie_key = {
        service: hashlib.sha256(
            f"{incident_id}\0{service}".encode("utf-8")
        ).hexdigest()
        for service in SERVICES
    }
    return [
        SERVICES[index]
        for index in sorted(
            range(len(SERVICES)),
            key=lambda i: (-float(scores[i]), tie_key[SERVICES[i]]),
        )
    ]


def rank_classical(method: str, case: dict, reference) -> tuple[list[str], dict]:
    views = modality_scores(case, reference)
    metric, log, trace = views["metric"], views["log"], views["trace"]
    if method == "nezha":
        # Nezha's event behavior graph is represented by log-template and
        # trace-event novelty; metrics are not part of its native ranker.
        combined = 0.45 * log + 0.55 * trace
    elif method == "multisource_rcd":
        # Multi-source consensus over equally normalized modalities.
        combined = (metric + log + trace) / 3.0
    elif method == "torai":
        # TORAI is KPI/topology centered. Trace severity breaks metric ties.
        combined = 0.8 * metric + 0.2 * trace
    else:
        raise ValueError(method)
    return order_from(combined, str(case["incident_id"])), {
        "metric_percentile": metric.round(4).tolist(),
        "log_percentile": log.round(4).tolist(),
        "trace_percentile": trace.round(4).tolist(),
        "service_edges": topology(case),
    }


def llm_call(system: str, prompt: str) -> tuple[list[str], str]:
    key, base = actor_config()
    response = requests.post(
        base + "/chat/completions",
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        json={
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": 220,
        },
        timeout=90,
    )
    response.raise_for_status()
    raw = response.json()["choices"][0]["message"]["content"]
    match = re.search(r"\{.*\}", raw, re.S)
    proposed = json.loads(match.group(0)).get("ranking", []) if match else []
    return [str(item).strip().lower() for item in proposed], raw


def rank_thinkfl(case: dict, reference) -> tuple[list[str], dict]:
    views = modality_scores(case, reference)
    fallback_score = 0.55 * views["metric"] + 0.45 * views["trace"]
    fallback = order_from(fallback_score)
    metric_tool = [
        {"service": service, "severity_percentile": round(float(views["metric"][SERVICE_INDEX[service]]), 4)}
        for service in order_from(views["metric"])[:8]
    ]
    trace_tool = [
        {"service": service, "severity_percentile": round(float(views["trace"][SERVICE_INDEX[service]]), 4)}
        for service in order_from(views["trace"])[:8]
    ]
    prompt = (
        "You are ThinkFL's failure-localization actor for the Social Network "
        "microservice system. A 60-second progressive telemetry view was "
        "normalized only against released no-fault captures. "
        "search_fluctuating_metrics returned " + json.dumps(metric_tool) +
        "; search_traces returned " + json.dumps(trace_tool) +
        ". Self-check that a high downstream symptom need not be the root. "
        "Return ONLY JSON {\"ranking\":[five service names]}. Candidates: " +
        json.dumps(SERVICES)
    )
    try:
        proposed, raw = llm_call("Use only supplied tool evidence.", prompt)
    except Exception as exc:
        proposed, raw = [], "API_FALLBACK:" + type(exc).__name__
    ordered = []
    for service in proposed + fallback + list(SERVICES):
        if service in SERVICE_INDEX and service not in ordered:
            ordered.append(service)
    return ordered, {
        "metric_tool": metric_tool,
        "trace_tool": trace_tool,
        "actor_output": raw,
        "api_used": not raw.startswith("API_FALLBACK"),
    }


def rank_rclagent(case: dict, reference) -> tuple[list[str], dict]:
    views = modality_scores(case, reference)
    direct = 0.40 * views["metric"] + 0.25 * views["log"] + 0.35 * views["trace"]
    graph = topology(case)
    propagated = direct.copy()
    for _ in range(len(SERVICES)):
        changed = False
        for parent, children in graph.items():
            inherited = max(
                (float(propagated[SERVICE_INDEX[child]]) for child in children), default=0.0
            ) * 0.35
            index = SERVICE_INDEX[parent]
            if inherited > propagated[index] + 1e-9:
                propagated[index] = inherited
                changed = True
        if not changed:
            break
    fallback = [
        SERVICES[index]
        for index in sorted(
            range(len(SERVICES)),
            key=lambda i: (-float(direct[i]), -float(propagated[i]), SERVICES[i]),
        )
    ]
    compact = {
        service: {
            "metric": round(float(views["metric"][SERVICE_INDEX[service]]), 4),
            "log": round(float(views["log"][SERVICE_INDEX[service]]), 4),
            "trace": round(float(views["trace"][SERVICE_INDEX[service]]), 4),
            "local_score": round(float(direct[SERVICE_INDEX[service]]), 4),
            "propagated_score": round(float(propagated[SERVICE_INDEX[service]]), 4),
            "children": graph.get(service, []),
        }
        for service in fallback
    }
    prompt = (
        "You are RCLAgent's root-level Diagnosis Synthesizer for the Social "
        "Network microservice system. Dedicated agents supplied normalized "
        "metric/log/trace evidence, and child evidence was recursively "
        "propagated over the observed topology. Prefer deepest direct causes "
        "over inherited symptoms. Return ONLY JSON "
        "{\"ranking\":[five service names]}. Candidates: " + json.dumps(SERVICES) +
        "\nGlobal Evidence Graph:\n" + json.dumps(compact)
    )
    try:
        proposed, raw = llm_call(
            "Use only supplied evidence; do not assume a service prior.", prompt
        )
    except Exception as exc:
        proposed, raw = [], "API_FALLBACK:" + type(exc).__name__
    ordered = []
    for service in proposed + fallback + list(SERVICES):
        if service in SERVICE_INDEX and service not in ordered:
            ordered.append(service)
    return ordered, {
        "global_evidence_graph": compact,
        "service_edges": graph,
        "actor_output": raw,
        "api_used": not raw.startswith("API_FALLBACK"),
    }


def metrics(predictions: list[dict]) -> dict[str, float | int]:
    hits = np.zeros(5, dtype=np.float64)
    for item in predictions:
        hits += [item["rank"] <= cutoff for cutoff in range(1, 6)]
    hits /= max(len(predictions), 1)
    return {
        "A@1": float(hits[0]),
        "A@5": float(hits[4]),
        "Avg@5": float(hits.mean()),
        "cases": len(predictions),
    }


def save(method: str, predictions: list[dict], extra: dict | None = None) -> None:
    destination = (
        RUN_ROOT / method / "runs" / "frozen_test" /
        RUN_ID
    )
    destination.mkdir(parents=True, exist_ok=True)
    config = {
        "dataset": "Eadro-SN",
        "protocol": "grouped-4-fold-leave-one-capture-out",
        "cases": 36,
        "candidate_services": 12,
        "window_seconds": 60,
        "healthy_reference": "three released no-fault captures",
        "uses_fault_label_at_inference": False,
        "uses_fault_type_at_inference": False,
    }
    if extra:
        config.update(extra)
    (destination / "summary.json").write_text(
        json.dumps({"metrics": metrics(predictions), "config": config}, indent=2) + "\n",
        encoding="utf-8",
    )
    (destination / "predictions_private.json").write_text(
        json.dumps(predictions, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics(predictions), indent=2), flush=True)


def run_ranker(method: str, cases: list[dict], reference) -> None:
    predictions = []
    for index, case in enumerate(cases, 1):
        if method in {"nezha", "multisource_rcd", "torai"}:
            ranking, diagnostics = rank_classical(method, case, reference)
        elif method == "thinkfl":
            ranking, diagnostics = rank_thinkfl(case, reference)
        elif method == "rclagent":
            ranking, diagnostics = rank_rclagent(case, reference)
        else:
            raise ValueError(method)
        target = case["root_cause_service"]
        position = ranking.index(target) + 1
        predictions.append(
            {
                "incident_id": case["incident_id"],
                "capture_id": case["capture_id"],
                "ground_truth_service": target,
                "ranking": ranking[:5],
                "rank": position,
                "diagnostics": diagnostics,
            }
        )
        print(
            f"[{index}/{len(cases)}] {case['incident_id']}: "
            f"rank={position} top5={ranking[:5]}",
            flush=True,
        )
    variants = {
        "nezha": "Nezha event-behavior compatibility variant",
        "multisource_rcd": "Multi-source RCD common-feature compatibility variant",
        "torai": "TORAI KPI/topology compatibility variant",
        "thinkfl": "ThinkFL GPT-4o-mini tool-interface compatibility variant",
        "rclagent": "RCLAgent GPT-4o-mini recursive-evidence compatibility variant",
    }
    save(method, predictions, {"variant": variants[method]})


def load_eadro_module():
    path = PROJECT / "baselines" / "eadro" / "adapter" / "eadro_re2ob.py"
    spec = importlib.util.spec_from_file_location("eadro_sn_compat", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.SERVICES = SERVICES
    module.SERVICE_INDEX = SERVICE_INDEX
    return module


def eadro_tensor(case: dict, reference) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    normalized = normalize_case(case, reference)
    metric = np.asarray(normalized["metric_x"], np.float32)[:, :, :7]
    trace_raw = np.asarray(normalized["trace_x"], np.float32)
    trace = np.mean(trace_raw, axis=2, keepdims=True)
    repeats = int(np.ceil(60 / metric.shape[1]))
    metric = np.repeat(metric, repeats, axis=1)[:, :60]
    trace = np.repeat(trace, repeats, axis=1)[:, :60]
    logs = np.asarray(normalized["log_x"], np.float32).mean(axis=1)
    logs = logs[:, :256]
    return metric, trace, logs, SERVICE_INDEX[case["root_cause_service"]]


def run_eadro(cases: list[dict], reference, epochs: int) -> None:
    module = load_eadro_module()
    predictions = []
    fold_summaries = []
    for fold in range(4):
        random.seed(42 + fold)
        np.random.seed(42 + fold)
        torch.manual_seed(42 + fold)
        torch.cuda.manual_seed_all(42 + fold)
        train_cases = [
            case for case in cases if case["incident_id"][:6] != f"sn-c{fold:02d}"
        ]
        test_cases = [
            case for case in cases if case["incident_id"][:6] == f"sn-c{fold:02d}"
        ]
        edges = set()
        for case in train_cases:
            for source, target in case.get("edges", ()):
                if source in SERVICE_INDEX and target in SERVICE_INDEX:
                    edges.add((SERVICE_INDEX[source], SERVICE_INDEX[target]))
        train_tensors = [eadro_tensor(case, reference) for case in train_cases]
        test_tensors = [eadro_tensor(case, reference) for case in test_cases]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = module.EadroCompat(256, sorted(edges)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        metric = torch.tensor(np.stack([item[0] for item in train_tensors]), device=device)
        trace = torch.tensor(np.stack([item[1] for item in train_tensors]), device=device)
        logs = torch.tensor(np.stack([item[2] for item in train_tensors]), device=device)
        labels = torch.tensor([item[3] for item in train_tensors], dtype=torch.long, device=device)
        for _ in range(epochs):
            model.train()
            detector, locator = model(metric, trace, logs)
            detection_labels = torch.ones(len(labels), dtype=torch.long, device=device)
            loss = 0.5 * F.cross_entropy(detector, detection_labels) + 0.5 * F.cross_entropy(locator, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        fold_predictions = []
        with torch.no_grad():
            for case, tensors in zip(test_cases, test_tensors):
                metric_x = torch.tensor(tensors[0][None], device=device)
                trace_x = torch.tensor(tensors[1][None], device=device)
                log_x = torch.tensor(tensors[2][None], device=device)
                _, locator = model(metric_x, trace_x, log_x)
                ranking = order_from(locator[0].detach().cpu().numpy())
                position = ranking.index(case["root_cause_service"]) + 1
                item = {
                    "incident_id": case["incident_id"],
                    "capture_id": case["capture_id"],
                    "ground_truth_service": case["root_cause_service"],
                    "ranking": ranking[:5],
                    "rank": position,
                    "fold": fold,
                }
                fold_predictions.append(item)
                predictions.append(item)
        fold_summaries.append({"fold": fold, "metrics": metrics(fold_predictions)})
        print(json.dumps(fold_summaries[-1]), flush=True)
    save(
        "eadro",
        predictions,
        {
            "variant": "Eadro TCN-GAT-Hawkes architecture compatibility port",
            "epochs": epochs,
            "folds": fold_summaries,
            "grouped_split_consequence": "test-fold culprit services are absent from supervised training folds",
        },
    )


def main() -> None:
    global RUN_ID
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "method",
        choices=("eadro", "nezha", "multisource_rcd", "torai", "thinkfl", "rclagent"),
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--run-id", default=RUN_ID)
    args = parser.parse_args()
    RUN_ID = args.run_id
    rq1_sn.configure(SERVICES)
    cases = rq1_sn.build_cases(False)
    reference = rq1_sn.fit_healthy_reference(SERVICES)
    if args.method == "eadro":
        run_eadro(cases, reference, args.epochs)
    else:
        run_ranker(args.method, cases, reference)


if __name__ == "__main__":
    main()
