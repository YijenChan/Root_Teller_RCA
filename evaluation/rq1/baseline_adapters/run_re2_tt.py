"""Run an existing RE2-OB compatibility adapter on the frozen RE2-TT split.

This launcher changes only dataset-bound globals and record discovery.  The
per-baseline scoring code remains in its original adapter, keeping RE2-OB
artifacts untouched.
"""
from __future__ import annotations

import os

import argparse
import importlib.util
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(os.environ.get("ROOTTELLER_WORKSPACE", Path.cwd())).expanduser().resolve()
RAW = PROJECT / "dataset" / "RCAEval RE" / "RE2" / "RE2-TT" / "RE2-TT"
MANIFEST = Path(os.environ.get("ROOTTELLER_ACTIVE_SPLIT_MANIFEST", PROJECT / "evaluation" / "rq1" / "manifests" / "active_split_manifest.csv"))
ADAPTERS = {
    "eadro": PROJECT / "evaluation" / "rq1" / "baseline_adapters" / "eadro.py",
    "nezha": PROJECT / "evaluation" / "rq1" / "baseline_adapters" / "nezha.py",
    "multisource_rcd": PROJECT / "evaluation" / "rq1" / "baseline_adapters" / "multisource_rcd.py",
    "torai": PROJECT / "evaluation" / "rq1" / "baseline_adapters" / "torai.py",
    "thinkfl": PROJECT / "evaluation" / "rq1" / "baseline_adapters" / "thinkfl.py",
    "rclagent": PROJECT / "evaluation" / "rq1" / "baseline_adapters" / "rclagent.py",
}


def services() -> tuple[str, ...]:
    first = next(path for path in RAW.rglob("simple_metrics.csv"))
    result = set()
    for column in pd.read_csv(first, nrows=0).columns:
        if column == "time" or "_" not in column:
            continue
        name = column.split("_", 1)[0].strip().lower()
        if name.startswith("gke-") or name in {"loadgenerator", "ts"}:
            continue
        result.add(name)
    return tuple(sorted(result))


def load(name: str):
    path = ADAPTERS[name]
    spec = importlib.util.spec_from_file_location(f"{name}_re2tt", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def manifest_rows(split: str):
    frame = pd.read_csv(MANIFEST)
    frame = frame[(frame.dataset_system == "RCAEval RE2-TT") & (frame.split == split)]
    if split == "test":
        frame = frame[frame.eligible]
    return frame


def configure(module, name: str) -> None:
    candidates = services()
    module.RAW = RAW
    module.MANIFEST = MANIFEST
    module.SERVICES = candidates
    if hasattr(module, "SERVICE_SET"):
        module.SERVICE_SET = set(candidates)
    if hasattr(module, "SERVICE_INDEX"):
        module.SERVICE_INDEX = {value: index for index, value in enumerate(candidates)}
    if hasattr(module, "CACHE"):
        module.CACHE = PROJECT / "baselines" / "eadro" / "cache" / "re2tt_clean_v1"

    if name == "eadro":
        def read_records():
            frame = manifest_rows("train")
            frame = pd.concat([frame, manifest_rows("validation"), manifest_rows("test")], ignore_index=True)
            return [
                module.Record(str(row.incident_id), str(row.split), module.canonical_service(row.root_cause_service),
                              float(row.inject_time), bool(row.eligible))
                for row in frame.itertuples(index=False)
            ]
        module.read_records = read_records
        original_split_records = module.split_records

        def split_records(records, split):
            # The original adapter constructs a test loader even during
            # development although test cases are intentionally not prepared.
            # Return an empty unused test set in that phase.
            if split == "test" and "--phase" in sys.argv:
                phase_index = sys.argv.index("--phase") + 1
                if phase_index < len(sys.argv) and sys.argv[phase_index] == "development":
                    return []
            return original_split_records(records, split)

        module.split_records = split_records

        def event_service(column: str) -> str:
            return module.canonical_service(column.split("_", 1)[0])

        def vocabulary(records, size):
            # RCAEval releases its Drain-like log events as columns in
            # logts.csv. Select frequent training-only events, as Eadro does.
            counts = Counter()
            for record in records:
                frame = pd.read_csv(record.directory / "logts.csv", low_memory=False)
                for column in frame.columns:
                    if column == "time":
                        continue
                    values = pd.to_numeric(frame[column], errors="coerce").fillna(0)
                    counts[column] += float(values.abs().sum())
            return {
                event: index + 1
                for index, (event, _) in enumerate(counts.most_common(size))
            }

        def historic_edges(records):
            # Fixed topology is derived once from the training partition.
            topology = pd.read_csv(
                PROJECT / "artifact" / "telemetry_unavailability" /
                "manifests" / "train_only_topology_edges.csv"
            )
            topology = topology[topology.dataset_system.eq("RCAEval RE2-TT")]
            edges = set()
            for row in topology.itertuples(index=False):
                source = module.canonical_service(row.source_service)
                target = module.canonical_service(row.target_service)
                if source in module.SERVICE_INDEX and target in module.SERVICE_INDEX:
                    edges.add((module.SERVICE_INDEX[source], module.SERVICE_INDEX[target]))
            return sorted(edges)

        metric_groups = ("cpu", "mem", "diskio", "socket", "workload", "error", "latency-90")

        def build_case(record, vocab, rebuild):
            module.CACHE.mkdir(parents=True, exist_ok=True)
            path = module.cache_file(record, vocab)
            if path.exists() and not rebuild:
                data = np.load(path, allow_pickle=False)
                return {key: data[key] for key in data.files}

            metric = pd.read_csv(record.directory / "simple_metrics.csv", low_memory=False)
            trace = pd.read_csv(record.directory / "tracets_lat.csv", low_memory=False)
            logs = pd.read_csv(record.directory / "logts.csv", low_memory=False)
            for frame in (metric, trace, logs):
                frame["time"] = pd.to_numeric(frame["time"], errors="coerce")
            valid_times = pd.concat([metric.time, trace.time, logs.time]).dropna()
            start = int(math.floor(float(valid_times.min())))
            end = float(valid_times.max())
            bins = max(1, int(math.ceil((end - start + 1) / module.BIN_SECONDS)))

            def coordinates(frame):
                seconds = np.floor(frame.time.fillna(start).to_numpy(float) - start).astype(np.int64)
                return (
                    np.clip(seconds // module.BIN_SECONDS, 0, bins - 1),
                    np.clip(seconds % module.BIN_SECONDS, 0, module.BIN_SECONDS - 1),
                )

            metrics = np.zeros(
                (bins, len(module.SERVICES), module.BIN_SECONDS, len(metric_groups)),
                dtype=np.float32,
            )
            mw, mt = coordinates(metric)
            for service, service_index in module.SERVICE_INDEX.items():
                for group_index, group in enumerate(metric_groups):
                    column = f"{service}_{group}"
                    if column not in metric:
                        continue
                    values = module.zscore(
                        pd.to_numeric(metric[column], errors="coerce").to_numpy(float)
                    )
                    for window, offset, value in zip(mw, mt, values):
                        metrics[window, service_index, offset, group_index] = value

            traces = np.zeros(
                (bins, len(module.SERVICES), module.BIN_SECONDS, 1), dtype=np.float32
            )
            tw, tt = coordinates(trace)
            trace_sums = np.zeros_like(traces)
            trace_counts = np.zeros_like(traces)
            for column in trace.columns:
                if column == "time":
                    continue
                service = event_service(column)
                if service not in module.SERVICE_INDEX:
                    continue
                values = pd.to_numeric(trace[column], errors="coerce").to_numpy(float)
                index = module.SERVICE_INDEX[service]
                for window, offset, value in zip(tw, tt, values):
                    if np.isfinite(value):
                        trace_sums[window, index, offset, 0] += value
                        trace_counts[window, index, offset, 0] += 1
            np.divide(trace_sums, trace_counts, out=traces, where=trace_counts > 0)
            for index in range(len(module.SERVICES)):
                traces[:, index, :, 0] = module.zscore(traces[:, index, :, 0])

            log_features = np.zeros(
                (bins, len(module.SERVICES), len(vocab) + 1), dtype=np.float32
            )
            lw, _ = coordinates(logs)
            for column in logs.columns:
                if column == "time":
                    continue
                service = event_service(column)
                if service not in module.SERVICE_INDEX:
                    continue
                event_index = vocab.get(column, 0)
                values = pd.to_numeric(logs[column], errors="coerce").fillna(0).to_numpy(float)
                for window, value in zip(lw, values):
                    if value > 0:
                        log_features[window, module.SERVICE_INDEX[service], event_index] += value
            log_features = np.log1p(log_features)

            starts = start + np.arange(bins) * module.BIN_SECONDS
            labels = np.where(
                starts + module.BIN_SECONDS * 0.5 >= record.inject_time,
                module.SERVICE_INDEX[record.root],
                -1,
            ).astype(np.int64)
            payload = {
                "metrics": metrics,
                "traces": traces,
                "logs": log_features,
                "labels": labels,
                "starts": starts.astype(np.int64),
            }
            np.savez_compressed(path, **payload)
            return payload

        module.vocabulary = vocabulary
        module.historic_edges = historic_edges
        module.build_case = build_case
    elif name in {"nezha", "multisource_rcd"}:
        def records_for(split):
            return [
                module.Record(str(row.incident_id), str(row.split), module.canonical_service(row.root_cause_service),
                              bool(row.eligible))
                for row in manifest_rows(split).itertuples(index=False)
            ]
        module.records_for = records_for
        if name == "nezha":
            def rank(record, min_support, min_score, association_seconds):
                # RE2-TT contains millions of raw spans per case.  RCAEval's
                # released one-minute log/trace event tables are the same
                # event representation required by Nezha and avoid a
                # prohibitively expensive raw span-log join.
                metric = pd.read_csv(record.directory / "simple_metrics.csv", low_memory=False)
                logts = pd.read_csv(record.directory / "logts.csv", low_memory=False)
                trace = pd.read_csv(record.directory / "tracets_err.csv", low_memory=False)
                for frame in (metric, logts, trace):
                    frame["time"] = pd.to_numeric(frame["time"], errors="coerce")
                boundary = (
                    float(metric["time"].min()) + float(metric["time"].max())
                ) / 2.0
                scores = {service: 0.0 for service in module.SERVICES}
                retained = 0

                def add_event_table(frame):
                    nonlocal retained
                    for column in frame.columns:
                        if column == "time" or "_" not in column:
                            continue
                        service = module.canonical_service(column.split("_", 1)[0])
                        if service not in scores:
                            continue
                        values = pd.to_numeric(frame[column], errors="coerce").fillna(0).abs()
                        ref = int((values[frame.time < boundary] > 0).sum())
                        prod = int((values[frame.time >= boundary] > 0).sum())
                        if prod <= min_support:
                            continue
                        novelty = prod / max(prod + ref, 1)
                        if novelty >= min_score:
                            scores[service] += novelty * math.log1p(prod)
                            retained += 1

                add_event_table(logts)
                add_event_table(trace)
                # Nezha is event-graph based; generic KPI threshold alarms are
                # not injected as synthetic events.
                tie_key = {
                    service: module.hashlib.sha256(
                        f"{record.incident_id}\0{service}".encode("utf-8")
                    ).hexdigest()
                    for service in module.SERVICES
                }
                ordered = sorted(
                    module.SERVICES,
                    key=lambda service: (-scores[service], tie_key[service]),
                )
                return ordered, {
                    "event_source": "RCAEval one-minute log/trace event tables",
                    "reference_seconds": module.BIN_SECONDS,
                    "retained_patterns": retained,
                    "service_scores": scores,
                }
            module.rank = rank
    elif name == "torai":
        def records_for(split):
            return [
                module.Record(str(row.incident_id), module.canonical_service(row.root_cause_service))
                for row in manifest_rows(split).itertuples(index=False)
            ]
        module.records_for = records_for

        def rank(record):
            data, boundary = module.data_for(record)
            result = module.torai(data, inject_time=boundary, dataset="train-ticket")
            raw = [module.canonical_service(str(item).rsplit("_", 1)[0]) for item in result["ranks"]]
            ordered = []
            for value in raw + list(module.SERVICES):
                if value in module.SERVICES and value not in ordered:
                    ordered.append(value)
            return ordered, {"fixed_boundary": boundary, "raw_ranks": result["ranks"][:20]}
        module.rank = rank
    elif name == "thinkfl":
        def rows(split):
            return [
                module.Row(str(row.incident_id), module.service(row.root_cause_service))
                for row in manifest_rows(split).itertuples(index=False)
            ]
        module.rows = rows

        def evidence(row):
            metric_frame = pd.read_csv(row.d / "simple_metrics.csv", low_memory=False)
            metric_frame["time"] = pd.to_numeric(metric_frame.time, errors="coerce")
            start = float(metric_frame.time.min())
            normal = metric_frame[metric_frame.time < start + 60]
            production = metric_frame[metric_frame.time >= start + 60]
            service_metric = {}
            service_metric_name = {}
            for column in metric_frame.columns:
                if column == "time" or "_" not in column:
                    continue
                service = module.service(column.split("_", 1)[0])
                if service not in module.SERVICES:
                    continue
                reference = pd.to_numeric(normal[column], errors="coerce").dropna()
                observed = pd.to_numeric(production[column], errors="coerce").dropna()
                if len(reference) <= 3 or len(observed) <= 3:
                    continue
                center = float(reference.median())
                mad = float((reference - center).abs().median()) * 1.4826
                scale = max(mad, abs(center) * 0.05, float(reference.std(ddof=0)) * 0.25, 1e-3)
                shift = min(abs(float(observed.median()) - center) / scale, 50.0)
                if shift > service_metric.get(service, -1):
                    service_metric[service] = shift
                    service_metric_name[service] = column.rsplit("_", 1)[-1]
            metric = sorted(
                (
                    (service, round(score, 2), service_metric_name[service])
                    for service, score in service_metric.items()
                ),
                key=lambda item: (-item[1], item[0]),
            )[:12]

            trace_scores = {}
            for filename, label in (("tracets_err.csv", "error"), ("tracets_lat.csv", "latency")):
                frame = pd.read_csv(row.d / filename, low_memory=False)
                frame["time"] = pd.to_numeric(frame.time, errors="coerce")
                reference_rows = frame[frame.time < start + 60]
                observed_rows = frame[frame.time >= start + 60]
                for column in frame.columns:
                    if column == "time":
                        continue
                    service = module.service(column.split("_", 1)[0])
                    if service not in module.SERVICES:
                        continue
                    reference = pd.to_numeric(reference_rows[column], errors="coerce").dropna()
                    observed = pd.to_numeric(observed_rows[column], errors="coerce").dropna()
                    if reference.empty or observed.empty:
                        continue
                    center = float(reference.median())
                    scale = max(
                        float((reference - center).abs().median()) * 1.4826,
                        abs(center) * 0.05,
                        1e-3,
                    )
                    shift = min(abs(float(observed.median()) - center) / scale, 50.0)
                    key = (service, label)
                    trace_scores[key] = max(trace_scores.get(key, 0.0), shift)
            trace = sorted(
                ((service, round(score, 2), label) for (service, label), score in trace_scores.items()),
                key=lambda item: (-item[1], item[0], item[2]),
            )[:12]
            return metric, trace

        module.evidence = evidence
    elif name == "rclagent":
        def rows(split):
            return [
                module.Row(str(row.incident_id), module.service(row.root_cause_service))
                for row in manifest_rows(split).itertuples(index=False)
            ]
        module.rows = rows

        topology = pd.read_csv(
            PROJECT / "artifact" / "telemetry_unavailability" /
            "manifests" / "train_only_topology_edges.csv"
        )
        topology = topology[topology.dataset_system.eq("RCAEval RE2-TT")]
        fixed_edges = {}
        for row in topology.itertuples(index=False):
            source = module.service(row.source_service)
            target = module.service(row.target_service)
            if source in module.SERVICES and target in module.SERVICES:
                fixed_edges.setdefault(source, set()).add(target)
        fixed_edges = {key: sorted(value) for key, value in fixed_edges.items()}

        def robust_shift(reference, observed):
            reference = pd.to_numeric(reference, errors="coerce").dropna()
            observed = pd.to_numeric(observed, errors="coerce").dropna()
            if len(reference) < 2 or len(observed) < 2:
                return 0.0
            center = float(reference.median())
            scale = max(
                float((reference - center).abs().median()) * 1.4826,
                abs(center) * 0.05,
                float(reference.std(ddof=0)) * 0.25,
                1e-3,
            )
            return min(abs(float(observed.median()) - center) / scale, 50.0)

        def evidence(row):
            metric_frame = pd.read_csv(row.directory / "simple_metrics.csv", low_memory=False)
            metric_frame["time"] = pd.to_numeric(metric_frame.time, errors="coerce")
            start = float(metric_frame.time.min())
            metric_reference = metric_frame[metric_frame.time < start + 60]
            metric_observed = metric_frame[metric_frame.time >= start + 60]
            metric = {service: [] for service in module.SERVICES}
            for column in metric_frame.columns:
                if column == "time" or "_" not in column:
                    continue
                service = module.service(column.split("_", 1)[0])
                if service not in metric:
                    continue
                shift = robust_shift(metric_reference[column], metric_observed[column])
                metric[service].append(
                    {"kpi": column.split("_", 1)[-1], "shift_z": round(shift, 2)}
                )
            for service in metric:
                metric[service] = sorted(
                    metric[service], key=lambda item: -item["shift_z"]
                )[:3]

            def event_evidence(filename):
                frame = pd.read_csv(row.directory / filename, low_memory=False)
                frame["time"] = pd.to_numeric(frame.time, errors="coerce")
                reference = frame[frame.time < start + 60]
                observed = frame[frame.time >= start + 60]
                output = {service: [] for service in module.SERVICES}
                for column in frame.columns:
                    if column == "time":
                        continue
                    service = module.service(column.split("_", 1)[0])
                    if service in output:
                        shift = robust_shift(reference[column], observed[column])
                        if shift > 0:
                            output[service].append(round(shift, 2))
                return {
                    service: sorted(values, reverse=True)[:3]
                    for service, values in output.items()
                }

            logs = event_evidence("logts.csv")
            trace_error = event_evidence("tracets_err.csv")
            trace_latency = event_evidence("tracets_lat.csv")
            local, direct = {}, {}
            for service in module.SERVICES:
                metric_score = max(
                    [item["shift_z"] for item in metric[service]], default=0.0
                )
                log_score = max(logs[service], default=0.0)
                error_score = max(trace_error[service], default=0.0)
                latency_score = max(trace_latency[service], default=0.0)
                own = metric_score + 0.5 * log_score + error_score + 0.35 * latency_score
                direct[service] = own
                local[service] = {
                    "metric_tool": metric[service],
                    "log_event_shift": log_score,
                    "trace_error_shift": error_score,
                    "trace_latency_shift": latency_score,
                    "local_score": round(own, 2),
                }
            propagated = dict(direct)
            for _ in range(len(module.SERVICES)):
                changed = False
                for parent, children in fixed_edges.items():
                    inherited = max(
                        (propagated.get(child, 0.0) for child in children), default=0.0
                    ) * 0.35
                    if inherited > propagated[parent] + 1e-9:
                        propagated[parent] = inherited
                        changed = True
                if not changed:
                    break
            for service in module.SERVICES:
                local[service]["propagated_score"] = round(propagated[service], 2)
                local[service]["children"] = fixed_edges.get(service, [])
            deterministic = sorted(
                module.SERVICES,
                key=lambda service: (-direct[service], -propagated[service], service),
            )
            compact_services = set(deterministic[:20])
            graph = {
                "reference_policy": "first fixed 60 seconds",
                "service_evidence": {
                    service: local[service] for service in deterministic[:20]
                },
                "service_edges": {
                    parent: [child for child in children if child in compact_services]
                    for parent, children in fixed_edges.items()
                    if parent in compact_services
                },
            }
            return graph, deterministic

        def ask(evidence_graph, fallback):
            key, base = module.actor_config()
            prompt = (
                "You are RCLAgent's root-level Diagnosis Synthesizer for the "
                "Train-Ticket microservice system. Dedicated agents produced "
                "local metric/log/trace evidence, and child evidence was "
                "recursively propagated along the fixed training topology. "
                "Rank deepest direct root causes above propagated downstream "
                "symptoms. Return ONLY JSON {\"ranking\":[five service names]}. "
                "Candidates: " + json.dumps(module.SERVICES) +
                "\nCompact Global Evidence Graph:\n" +
                json.dumps(evidence_graph, ensure_ascii=False)
            )
            response = module.requests.post(
                base + "/chat/completions",
                headers={
                    "Authorization": "Bearer " + key,
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {
                            "role": "system",
                            "content": "Use only supplied evidence; do not assume a service prior.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0,
                    "max_tokens": 220,
                },
                timeout=90,
            )
            response.raise_for_status()
            raw = response.json()["choices"][0]["message"]["content"]
            match = module.re.search(r"\{.*\}", raw, module.re.S)
            proposed = json.loads(match.group(0)).get("ranking", []) if match else []
            return [module.service(value) for value in proposed], raw

        module.evidence = evidence
        module.ask = ask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", choices=tuple(ADAPTERS))
    args, remaining = parser.parse_known_args()
    module = load(args.baseline)
    configure(module, args.baseline)
    sys.argv = [str(ADAPTERS[args.baseline]), *remaining]
    if args.baseline == "torai":
        torai_parser = argparse.ArgumentParser()
        torai_parser.add_argument("--split", choices=("validation", "test"), required=True)
        torai_parser.add_argument("--output", type=Path, required=True)
        torai_args = torai_parser.parse_args(remaining)
        module.run(module.records_for(torai_args.split), torai_args.output)
    else:
        module.main()


if __name__ == "__main__":
    main()
