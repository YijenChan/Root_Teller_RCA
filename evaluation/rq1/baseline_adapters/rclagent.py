"""RCLAgent-style clean RE2-OB compatibility evaluation.

The official RCLAgent repository ships its coordinator and an RE2-OB
preprocessor, but its preprocessor selects anomalous traces around
``inject_time.txt``.  This adapter deliberately does not use that file or a
label at inference.  It builds the same three evidence views from a fixed
60-second reference prefix and the following observation period:

* dedicated-agent evidence: per-service metric, log, and trace observations;
* recursion-of-thought: child evidence is propagated over the observed trace
  parent-child graph, from leaves towards roots;
* global evidence graph: the compact propagated evidence is given to an
  OpenAI-compatible LLM for the final root-level ranking.

The final synthesizer is the user-authorized GPT-4o-mini endpoint; therefore
results must be reported as an API/data-interface compatibility variant, not
as the paper's Qwen/Claude configuration.
"""
from __future__ import annotations

import os

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

PROJECT = Path(os.environ.get("ROOTTELLER_WORKSPACE", Path.cwd())).expanduser().resolve()
RAW = PROJECT / "dataset" / "RCAEval RE" / "RE2" / "RE2-OB" / "RE2-OB"
MANIFEST = Path(os.environ.get("ROOTTELLER_ACTIVE_SPLIT_MANIFEST", PROJECT / "evaluation" / "rq1" / "manifests" / "active_split_manifest.csv"))
SERVICES = ("adservice", "cartservice", "checkoutservice", "currencyservice", "emailservice",
            "frontend", "paymentservice", "productcatalogservice", "recommendationservice",
            "redis", "shippingservice")


def service(value: object) -> str:
    value = str(value).lower().replace("_", "-")
    return {"frontendservice": "frontend", "frontend-external": "frontend", "redis-cart": "redis"}.get(value, value)


@dataclass(frozen=True)
class Row:
    incident_id: str
    root: str

    @property
    def directory(self) -> Path:
        return RAW / Path(self.incident_id)


def rows(split: str) -> list[Row]:
    table = pd.read_csv(MANIFEST)
    table = table[(table.dataset_system == "RCAEval RE2-OB") & (table.split == split)]
    if split == "test":
        table = table[table.eligible]
    return [Row(str(x.incident_id), service(x.root_cause_service)) for x in table.itertuples(index=False)]


def actor_config() -> tuple[str, str]:
    key = os.environ["ROOTTELLER_API_KEY"]
    base = os.environ.get("ROOTTELLER_API_BASE", "https://api.openai.com/v1")
    return key, base.rstrip("/")
def _metric_evidence(directory: Path, start: float) -> dict[str, list[dict]]:
    frame = pd.read_csv(directory / "simple_metrics.csv", low_memory=False)
    frame["time"] = pd.to_numeric(frame.time, errors="coerce")
    reference, observed = frame[frame.time < start + 60], frame[frame.time >= start + 60]
    output: dict[str, list[dict]] = defaultdict(list)
    for column in frame.columns:
        if column == "time":
            continue
        name = service(column.split("_", 1)[0])
        if name not in SERVICES:
            continue
        a = pd.to_numeric(reference[column], errors="coerce").dropna()
        b = pd.to_numeric(observed[column], errors="coerce").dropna()
        if len(a) < 4 or len(b) < 4:
            continue
        z = abs(float(b.mean() - a.mean())) / max(float(a.std(ddof=0)), 1e-6)
        if np.isfinite(z):
            output[name].append({"kpi": column.split("_", 1)[-1], "shift_z": round(z, 2)})
    for name in output:
        output[name] = sorted(output[name], key=lambda x: -x["shift_z"])[:3]
    return output


def _log_evidence(directory: Path, start: float) -> dict[str, dict]:
    logs = pd.read_csv(directory / "logs.csv", low_memory=False)
    ts = pd.to_numeric(logs.get("time"), errors="coerce")
    logs = logs.assign(_time=ts)
    output: dict[str, dict] = {}
    for name, group in logs.groupby(logs.get("container_name", "").map(service)):
        if name not in SERVICES:
            continue
        before, after = group[group._time < start + 60], group[group._time >= start + 60]
        def err_count(x: pd.DataFrame) -> int:
            level = x.get("level", pd.Series("", index=x.index)).astype(str).str.upper()
            message = x.get("message", pd.Series("", index=x.index)).astype(str).str.upper()
            return int((level.isin(["ERROR", "FATAL", "WARN"]) | message.str.contains("ERROR|EXCEPTION|FAIL", regex=True)).sum())
        b, a = err_count(before), err_count(after)
        samples = after.get("message", pd.Series([], dtype=str)).dropna().astype(str).head(2).tolist()
        output[name] = {"reference_error_logs": b, "observed_error_logs": a,
                        "error_increase": a - b, "sample_messages": samples}
    return output


def _trace_graph_evidence(directory: Path, start: float) -> tuple[dict[str, dict], dict[str, list[str]]]:
    traces = pd.read_csv(directory / "traces.csv", low_memory=False)
    traces["time"] = pd.to_numeric(traces.time, errors="coerce")
    observed = traces[traces.time >= start + 60].copy()
    observed["service"] = observed.serviceName.map(service)
    observed = observed[observed.service.isin(SERVICES)]
    status = pd.to_numeric(observed.get("statusCode"), errors="coerce").fillna(0)
    observed["_bad"] = (status >= 400).astype(int)
    output: dict[str, dict] = {}
    for name, group in observed.groupby("service"):
        output[name] = {"span_count": int(len(group)), "error_spans": int(group._bad.sum()),
                        "mean_duration_us": round(float(pd.to_numeric(group.duration, errors="coerce").mean() or 0), 1)}
    # Build service-level caller -> callee edges using observed span parents.
    by_span = observed.set_index("spanID")["service"].to_dict()
    edges: dict[str, set[str]] = defaultdict(set)
    for item in observed.itertuples(index=False):
        parent = by_span.get(getattr(item, "parentSpanID", None))
        child = getattr(item, "service")
        if parent and parent != child:
            edges[parent].add(child)
    return output, {k: sorted(v) for k, v in edges.items()}


def evidence(row: Row) -> tuple[dict, list[str]]:
    metrics = pd.read_csv(row.directory / "simple_metrics.csv", usecols=["time"])
    start = float(pd.to_numeric(metrics.time, errors="coerce").min())
    metric = _metric_evidence(row.directory, start)
    logs = _log_evidence(row.directory, start)
    trace, edges = _trace_graph_evidence(row.directory, start)
    local, score = {}, {}
    for name in SERVICES:
        m, l, t = metric.get(name, []), logs.get(name, {}), trace.get(name, {})
        own = (max([x["shift_z"] for x in m], default=0.0) + 2.0 * max(l.get("error_increase", 0), 0)
               + 0.5 * t.get("error_spans", 0))
        score[name] = own
        local[name] = {"metric_tool": m, "log_tool": l, "trace_tool": t, "local_score": round(own, 2)}
    # Bottom-up propagation mirrors RCLAgent consolidation: a caller inherits
    # evidence from a failing child, but with a discount so root candidates
    # remain deepest services with direct evidence.
    propagated = dict(score)
    for _ in range(len(SERVICES)):
        changed = False
        for parent, children in edges.items():
            candidate = max((propagated.get(child, 0.0) for child in children), default=0.0) * 0.35
            if candidate > propagated[parent] + 1e-9:
                propagated[parent], changed = candidate, True
        if not changed:
            break
    for name in SERVICES:
        local[name]["propagated_score"] = round(propagated[name], 2)
        local[name]["children"] = edges.get(name, [])
    deterministic = sorted(SERVICES, key=lambda n: (-score[n], -propagated[n], n))
    return {"reference_policy": "first fixed 60 seconds", "service_evidence": local,
            "service_edges": edges}, deterministic


def ask(evidence_graph: dict, fallback: list[str]) -> tuple[list[str], str]:
    key, base = actor_config()
    prompt = (
        "You are RCLAgent's root-level Diagnosis Synthesizer for Online Boutique. "
        "The supplied Global Evidence Graph was created by dedicated agents: each has local metric/log/trace evidence, "
        "and child evidence was recursively propagated along the service graph. Rank the DEEPEST root-cause services. "
        "Direct error logs and metric shifts matter more than propagated latency symptoms. Use only this evidence. "
        "Return ONLY JSON {\"ranking\":[five service names]}. Candidates: " + json.dumps(SERVICES) +
        "\nGlobal Evidence Graph:\n" + json.dumps(evidence_graph, ensure_ascii=False)
    )
    response = requests.post(base + "/chat/completions", headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        json={"model": "gpt-4o-mini", "messages": [{"role": "system", "content": "Use only supplied evidence; do not assume a service prior."}, {"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 220}, timeout=90)
    response.raise_for_status()
    raw = response.json()["choices"][0]["message"]["content"]
    match = re.search(r"\{.*\}", raw, re.S)
    proposed = json.loads(match.group(0)).get("ranking", []) if match else []
    return [service(x) for x in proposed], raw


def rank(row: Row) -> tuple[list[str], dict]:
    graph, fallback = evidence(row)
    try:
        proposed, raw = ask(graph, fallback)
    except Exception as exc:
        proposed, raw = [], "API_FALLBACK:" + type(exc).__name__
    ordered = []
    for name in proposed + fallback + list(SERVICES):
        if name in SERVICES and name not in ordered:
            ordered.append(name)
    return ordered, {"global_evidence_graph": graph, "actor_output": raw, "api_used": not raw.startswith("API_FALLBACK")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    predictions, hits = [], np.zeros(5)
    selected = rows(args.split)
    for index, row in enumerate(selected, 1):
        ranking, diagnostics = rank(row)
        position = ranking.index(row.root) + 1
        hits += [position <= k for k in range(1, 6)]
        predictions.append({"incident_id": row.incident_id, "ground_truth_service": row.root, "ranking": ranking[:5], "rank": position, "diagnostics": diagnostics})
        print(f"[{index}/{len(selected)}] {row.incident_id}: rank={position} top5={ranking[:5]}", flush=True)
    metrics = {"A@1": float(hits[0] / len(selected)), "A@5": float(hits[4] / len(selected)), "Avg@5": float(hits.mean() / len(selected)), "cases": len(selected)}
    config = {"variant": "RCLAgent-GPT-4o-mini recursive-evidence compatibility variant", "reference_policy": "first fixed 60 seconds", "uses_injection_time_at_inference": False, "uses_labels_at_inference": False, "actor": "gpt-4o-mini temperature=0", "official_structures_preserved": ["per-service dedicated evidence", "trace parent-child recursion", "global evidence graph", "root-level diagnosis synthesis"]}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps({"metrics": metrics, "config": config}, indent=2), encoding="utf-8")
    (args.output / "predictions_private.json").write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
