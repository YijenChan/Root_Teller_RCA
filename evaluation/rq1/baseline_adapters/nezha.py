"""Label-free RE2-OB compatibility variant of Nezha (FSE'23).

This is deliberately *not* a claim of native Nezha reproduction.  RE2-OB
does not expose the SpanID in its logs and does not provide an independent
construction-phase capture.  The adapter preserves the published method's
core comparison--frequent event patterns in a normal reference versus a
fault-suffering production stream--under two explicit compatibility rules:

1. the first half of every released incident is the reference and the second
   half is production data.  No injection time,
   fault type, or root-cause label participates in feature extraction;
2. each log is linked to the temporally nearest span from the same service,
   only when it is within ``association_seconds`` of that span's interval.

The output is a service ranking. Labels are read only in ``evaluate``.
"""
from __future__ import annotations

import os

import argparse
import hashlib
import json
import math
from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(os.environ.get("ROOTTELLER_WORKSPACE", Path.cwd())).expanduser().resolve()
ROOT = PROJECT / "baselines" / "nezha"
RAW = PROJECT / "dataset" / "RCAEval RE" / "RE2" / "RE2-OB" / "RE2-OB"
MANIFEST = Path(os.environ.get("ROOTTELLER_ACTIVE_SPLIT_MANIFEST", PROJECT / "evaluation" / "rq1" / "manifests" / "active_split_manifest.csv"))
SERVICES = (
    "adservice", "cartservice", "checkoutservice", "currencyservice", "emailservice",
    "frontend", "paymentservice", "productcatalogservice", "recommendationservice",
    "redis", "shippingservice",
)
SERVICE_SET = set(SERVICES)
BIN_SECONDS = 60


def canonical_service(value: object) -> str:
    name = str(value).strip().lower().replace("_", "-")
    aliases = {"frontendservice": "frontend", "frontend-external": "frontend", "redis-cart": "redis"}
    name = aliases.get(name, name)
    if name.endswith("-service") and name.replace("-", "") in SERVICE_SET:
        name = name.replace("-", "")
    return name


@dataclass(frozen=True)
class Record:
    incident_id: str
    split: str
    root: str
    eligible: bool

    @property
    def directory(self) -> Path:
        return RAW / Path(self.incident_id)


def records_for(split: str) -> list[Record]:
    frame = pd.read_csv(MANIFEST)
    frame = frame.loc[frame.dataset_system.eq("RCAEval RE2-OB")]
    rows = [Record(str(x.incident_id), str(x.split), canonical_service(x.root_cause_service), bool(x.eligible))
            for x in frame.itertuples(index=False)]
    rows = [x for x in rows if x.split == split]
    return [x for x in rows if split != "test" or x.eligible]


def bins(start: float, values: np.ndarray) -> np.ndarray:
    return np.maximum(0, np.floor((values - start) / BIN_SECONDS).astype(np.int64))


def add(counter: dict[int, Counter], window: int, service: str, pattern: str) -> None:
    if service in SERVICE_SET:
        counter[window][(service, pattern)] += 1


def merge_counts(output: dict[int, Counter], frame: pd.DataFrame) -> None:
    """Transfer vectorised event counts into the per-window Counter format."""
    if frame.empty:
        return
    grouped = frame.groupby(["window", "service", "pattern"], sort=False).size()
    for (window, service, pattern), count in grouped.items():
        add(output, int(window), str(service), str(pattern))
        output[int(window)][(str(service), str(pattern))] += int(count) - 1


def event_patterns(record: Record, association_seconds: float) -> tuple[dict[int, Counter], int]:
    trace_path, log_path, metric_path = (record.directory / "traces.csv", record.directory / "logs.csv", record.directory / "metrics.csv")
    metrics = pd.read_csv(metric_path, usecols=["time"], low_memory=False)
    metric_time = pd.to_numeric(metrics["time"], errors="coerce").dropna().to_numpy(float)
    if not len(metric_time):
        return defaultdict(Counter), 0
    start, maximum = float(metric_time.min()), float(metric_time.max())
    num_bins = max(1, int(math.floor((maximum - start) / BIN_SECONDS)) + 1)
    output: dict[int, Counter] = defaultdict(Counter)

    traces = pd.read_csv(trace_path, dtype=str, low_memory=False)
    required = {"traceID", "spanID", "parentSpanID", "serviceName", "operationName", "startTimeMillis", "duration"}
    if not required.issubset(traces.columns):
        raise ValueError(f"unexpected trace schema for {record.incident_id}")
    traces["service"] = traces.serviceName.map(canonical_service)
    traces["start"] = pd.to_numeric(traces.startTimeMillis, errors="coerce").fillna(0).astype(float) / 1000.0
    traces["duration"] = pd.to_numeric(traces.duration, errors="coerce").fillna(0).clip(lower=0).astype(float) / 1000.0
    traces["window"] = bins(start, traces["start"].to_numpy(float))
    traces = traces.loc[traces.service.isin(SERVICE_SET)].copy()
    traces["operation"] = traces.operationName.fillna("").astype(str).str.rsplit("/", n=1).str[-1]
    # Trace parent/child edges are native Nezha-like execution-order events.
    merge_counts(output, pd.DataFrame({"window": traces.window, "service": traces.service,
                                       "pattern": "operation:" + traces.service + "/" + traces.operation}))
    lookup = traces[["spanID", "service", "operation"]].rename(
        columns={"spanID": "parentSpanID", "service": "parent_service", "operation": "parent_operation"})
    calls = traces.merge(lookup, on="parentSpanID", how="left")
    calls = calls.loc[calls.parent_service.notna()]
    merge_counts(output, pd.DataFrame({"window": calls.window, "service": calls.service,
                                       "pattern": "call:" + calls.parent_service + "/" + calls.parent_operation + "->" + calls.service + "/" + calls.operation}))

    # Compatibility log-to-span association: same service, nearest temporal span.
    logs = pd.read_csv(log_path, dtype=str, low_memory=False)
    logs["service"] = logs.container_name.map(canonical_service)
    logs["time_s"] = pd.to_numeric(logs.timestamp, errors="coerce").fillna(0).astype(float) / 1e9
    logs = logs.loc[logs.service.isin(SERVICE_SET)].copy()
    # RE2-OB compatibility association: nearest start time among spans from the same service.
    linked_logs: list[pd.DataFrame] = []
    for service, service_logs in logs.groupby("service", sort=False):
        service_spans = traces.loc[traces.service.eq(service), ["start", "operation"]].sort_values("start")
        if service_spans.empty:
            continue
        joined = pd.merge_asof(service_logs.sort_values("time_s"), service_spans, left_on="time_s", right_on="start",
                               direction="nearest", tolerance=association_seconds)
        joined = joined.loc[joined.operation.notna()]
        if not joined.empty:
            linked_logs.append(joined)
    if linked_logs:
        joined_logs = pd.concat(linked_logs, ignore_index=True)
        # Some RE2 releases retain an empty/renamed cluster-id field after
        # CSV parsing.  Resolve template columns explicitly rather than via
        # pandas attribute access, which is not stable for every schema.
        template_column = next((name for name in ("cluster_id", "cluster_id_x", "log_template", "log_template_x")
                                if name in joined_logs.columns), None)
        template = (joined_logs[template_column] if template_column is not None
                    else pd.Series("unknown", index=joined_logs.index))
        fallback_column = next((name for name in ("log_template", "log_template_x")
                                if name in joined_logs.columns and name != template_column), None)
        if fallback_column is not None:
            template = template.fillna(joined_logs[fallback_column])
        template = template.fillna("unknown").astype(str)
        merge_counts(output, pd.DataFrame({"window": bins(start, joined_logs.time_s.to_numpy(float)), "service": joined_logs.service,
                                           "pattern": "spanlog:" + joined_logs.service + "/" + joined_logs.operation + "->" + template}))

    # Nezha's native event graph is constructed from traces and logs. Metrics
    # are intentionally excluded here: treating per-KPI threshold crossings as
    # Nezha events turns this compatibility layer into a generic anomaly
    # ranker and was found to inflate RE2 localization substantially.
    return output, num_bins


def rank(record: Record, min_support: int, min_score: float, association_seconds: float) -> tuple[list[str], dict]:
    per_window, num_bins = event_patterns(record, association_seconds)
    boundary_window = max(1, num_bins // 2)
    reference: Counter = Counter()
    production: Counter = Counter()
    for window, values in per_window.items():
        if window < boundary_window:
            reference.update(values)
        else:
            production.update(values)
    contributions: dict[str, float] = defaultdict(float)
    retained = 0
    for (service, pattern), prod_count in production.items():
        ref_count = reference.get((service, pattern), 0)
        if prod_count <= min_support:
            continue
        score = prod_count / (prod_count + ref_count)
        if score < min_score:
            continue
        contributions[service] += score * math.log1p(prod_count)
        retained += 1
    # A score tie means that Nezha retained no comparative evidence.  Breaking
    # such ties alphabetically creates a hidden service-name prior (and happens
    # to favour several RE2 culprit names).  Use a deterministic incident-local
    # hash so the complete ranking is reproducible but label- and name-order
    # agnostic.
    tie_key = {
        service: hashlib.sha256(
            f"{record.incident_id}\0{service}".encode("utf-8")
        ).hexdigest()
        for service in SERVICES
    }
    ordered = sorted(
        SERVICES,
        key=lambda service: (-contributions.get(service, 0.0), tie_key[service]),
    )
    return ordered, {"windows": num_bins, "reference_patterns": int(sum(reference.values())),
                     "production_patterns": int(sum(production.values())), "retained_patterns": retained,
                     "service_scores": {service: contributions.get(service, 0.0) for service in SERVICES}}


def evaluate(rows: list[Record], args: argparse.Namespace) -> tuple[dict, list[dict]]:
    predictions: list[dict] = []
    hits = np.zeros(5, dtype=float)
    for index, row in enumerate(rows, 1):
        ranking, diagnostics = rank(row, args.min_support, args.min_score, args.association_seconds)
        position = ranking.index(row.root) + 1
        hits += np.asarray([position <= k for k in range(1, 6)], dtype=float)
        predictions.append({"incident_id": row.incident_id, "ground_truth_service": row.root,
                            "ranking": ranking[:5], "rank": position, "diagnostics": diagnostics})
        print(f"[{index}/{len(rows)}] {row.incident_id}: rank={position} top5={ranking[:5]}", flush=True)
    metrics = {"A@1": float(hits[0] / len(rows)), "A@5": float(hits[4] / len(rows)),
               "Avg@5": float(hits.mean() / len(rows)), "cases": len(rows)}
    return metrics, predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-support", type=int, default=5)
    parser.add_argument("--min-score", type=float, default=0.67)
    parser.add_argument("--association-seconds", type=float, default=30.0)
    args = parser.parse_args()
    rows = records_for(args.split)
    metrics, predictions = evaluate(rows, args)
    args.output.mkdir(parents=True, exist_ok=True)
    config = {"variant": "Nezha-RE2 compatibility variant", "window_seconds": BIN_SECONDS,
              "reference_policy": "first half of predefined observation range",
              "production_policy": "second half of predefined observation range",
              "log_span_association": "nearest same-service span within association_seconds",
              "uses_injection_time_at_inference": False, "uses_labels_at_inference": False,
              "min_support": args.min_support, "min_score": args.min_score,
              "association_seconds": args.association_seconds}
    (args.output / "summary.json").write_text(json.dumps({"metrics": metrics, "config": config}, indent=2), encoding="utf-8")
    (args.output / "predictions_private.json").write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
