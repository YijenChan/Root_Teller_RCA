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
ADAPTERS = {name: Path(__file__).resolve().with_name(name + ".py")
            for name in ("eadro", "nezha", "multisource_rcd", "torai", "thinkfl", "rclagent")}


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

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", choices=tuple(ADAPTERS))
    args, remaining = parser.parse_known_args()
    if args.baseline in {"nezha", "multisource_rcd", "torai"}:
        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
        from root_teller.baselines.classical_run import main as run
        sys.argv = [sys.argv[0], "--method", args.baseline, "--dataset", "re2_tt", *remaining]
        run()
        return
    if args.baseline in {"thinkfl", "rclagent"}:
        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
        from root_teller.baselines.run import main as run
        sys.argv = [sys.argv[0], "--method", args.baseline, "--dataset", "re2_tt", *remaining]
        run()
        return
    module = load(args.baseline)
    configure(module, args.baseline)
    sys.argv = [str(ADAPTERS[args.baseline]), *remaining]
    module.main()


if __name__ == "__main__":
    main()
