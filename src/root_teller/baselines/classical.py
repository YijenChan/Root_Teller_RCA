from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
import importlib
import importlib.util
import json
import logging
from pathlib import Path
import re
import sys
import tarfile
import threading
import types

import numpy as np
import pandas as pd

from .data import canonical, _json_capture


SOURCE_ROOT = Path(__file__).resolve().parents[3] / "baselines"
_LOCK = threading.RLock()


@dataclass
class Capture:
    metrics: pd.DataFrame
    logs: list[dict]
    spans: list[dict]
    start: float
    end: float

    @property
    def services(self):
        return sorted({c.split("_", 1)[0] for c in self.metrics if c != "time"}
                      | {r["service"] for r in self.spans} | {r["service"] for r in self.logs})


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def namespace(names):
    saved = {k: v for k, v in sys.modules.items() if any(k == n or k.startswith(n + ".") for n in names)}
    for key in saved:
        sys.modules.pop(key)
    try:
        yield
    finally:
        for key in list(sys.modules):
            if any(key == n or key.startswith(n + ".") for n in names):
                sys.modules.pop(key)
        sys.modules.update(saved)


def read_capture(directory, dataset, start=None, end=None):
    directory = Path(directory)
    logs, spans = [], []
    if dataset == "eadro_sn":
        columns = []
        for path in sorted((directory / "metrics").glob("*.csv")):
            frame = pd.read_csv(path).rename(columns={"timestamp": "time"})
            service = canonical(path.stem, dataset)
            columns.append(frame.set_index("time").rename(columns=lambda c: service + "_" + c))
        metric = pd.concat(columns, axis=1).reset_index() if columns else pd.DataFrame(columns=["time"])
        if (directory / "logs.json").exists():
            for service, lines in _json_capture(str(directory / "logs.json")).items():
                for line in lines:
                    match = re.match(r"\[(.*?)\]\s*<(.*?)>:\s*(.*)", line)
                    if match:
                        stamp = pd.to_datetime(match[1], format="%Y-%b-%d %H:%M:%S.%f", utc=True).timestamp()
                        logs.append({"service": canonical(service, dataset), "time": stamp, "text": match[3]})
        if (directory / "spans.json").exists():
            for group in _json_capture(str(directory / "spans.json")):
                processes = group.get("processes", {})
                for row in group.get("spans", []):
                    tags = {t["key"]: t.get("value") for t in row.get("tags", [])}
                    parents = [r["spanID"] for r in row.get("references", []) if r.get("refType") == "CHILD_OF"]
                    stamp = float(row["startTime"]) / 1e6 - 8 * 3600
                    service = canonical(processes.get(row.get("processID"), {}).get("serviceName", "unknown"), dataset)
                    spans.append({"trace": str(row["traceID"]), "id": str(row["spanID"]), "parent": str(parents[0]) if parents else "",
                                  "service": service, "time": stamp, "duration": float(row["duration"]),
                                  "operation": str(row.get("operationName", "request")),
                                  "error": bool(tags.get("error")) or int(tags.get("http.status_code", 0) or 0) >= 500})
    else:
        metric = pd.read_csv(directory / "simple_metrics.csv", low_memory=False)
        metric = metric.rename(columns={c: canonical(c.split("_", 1)[0], dataset) + "_" + c.split("_", 1)[1]
                                      for c in metric if c != "time" and "_" in c})
        if (directory / "logs.csv").exists():
            frame = pd.read_csv(directory / "logs.csv", low_memory=False)
            for row in frame.to_dict("records"):
                logs.append({"service": canonical(row["container_name"], dataset), "time": float(row["timestamp"]) / 1e9,
                             "text": str(row.get("message", row.get("log_template", "")))})
        if (directory / "traces.csv").exists():
            for row in pd.read_csv(directory / "traces.csv", dtype=str, keep_default_na=False).to_dict("records"):
                stamp = float(row["startTime"]) / 1e6 if row.get("startTime") else float(row["startTimeMillis"]) / 1e3
                code = float(row.get("statusCode") or 0)
                spans.append({"trace": row["traceID"], "id": row["spanID"], "parent": row.get("parentSpanID", ""),
                              "service": canonical(row["serviceName"], dataset), "time": stamp, "duration": float(row["duration"]),
                              "operation": row.get("operationName", "request"), "error": code >= 400 or code == 2})
    times = pd.to_numeric(metric["time"], errors="coerce")
    if not times.notna().any():
        raise ValueError("No observation time grid")
    start = float(times.min()) if start is None else float(start)
    end = float(times.max()) + 1 if end is None else float(end)
    if end <= start:
        raise ValueError("Empty observation interval")
    metric["time"] = times
    metric = metric[(times >= start) & (times < end)].copy()
    if dataset == "re2_ob":
        from root_teller.module1.config import SERVICES
        services = set(SERVICES)
    else:
        services = {c.split("_", 1)[0] for c in metric if c != "time" and "_" in c}
        services = {s for s in services if s not in {"ts", "loadgenerator", "unknown"} and not s.startswith("gke-")}
    metric = metric[[c for c in metric if c == "time" or c.split("_", 1)[0] in services]]
    logs = [r for r in logs if start <= r["time"] < end and r["service"] in services]
    spans = [r for r in spans if start <= r["time"] < end and r["service"] in services]
    return Capture(metric, logs, spans, start, end)


def healthy_captures(workspace):
    workspace = Path(workspace)
    base = workspace / "dataset/Eadro-SN/SN Dataset/SN Dataset/no fault"
    captures = sorted(p for p in base.glob("SN.*") if p.is_dir())
    cache = workspace / "cache/baseline_sn/healthy_raw"
    for archive in sorted(base.glob("SN.*.tar.xz")):
        destination = cache / archive.name.removesuffix(".tar.xz")
        if not destination.exists():
            cache.mkdir(parents=True, exist_ok=True)
            with tarfile.open(archive, "r:xz") as handle:
                handle.extractall(cache, filter="data")
        captures.append(destination)
    if not captures:
        for cache in sorted((workspace / "cache/rq1_multidataset_clean/eadro_sn").glob("**/healthy_raw")):
            captures.extend(sorted(p for p in cache.glob("SN.*") if p.is_dir()))
            if captures:
                break
    if not captures:
        raise FileNotFoundError("Released SN no-fault captures are required")
    return [read_capture(p, "eadro_sn") for p in captures]


def slice_capture(capture, start, end):
    return Capture(capture.metrics[(capture.metrics.time >= start) & (capture.metrics.time < end)].copy(),
                   [r for r in capture.logs if start <= r["time"] < end],
                   [r for r in capture.spans if start <= r["time"] < end], start, end)


def frames(capture, templates):
    from root_teller.module1.features import normalize_template
    length = int(np.ceil(capture.end - capture.start))
    grid = pd.RangeIndex(length)
    metric = capture.metrics.copy()
    metric["time"] = np.floor(metric.time - capture.start).astype(int)
    metric = metric.set_index("time").apply(pd.to_numeric, errors="coerce").groupby(level=0).mean().reindex(grid).ffill().fillna(0)
    log, err, latency = pd.DataFrame(index=grid), pd.DataFrame(index=grid), pd.DataFrame(index=grid)
    for service, template in templates:
        key = service + "_log-" + str(templates[(service, template)])
        log[key] = 0.
    for row in capture.logs:
        key = (row["service"], normalize_template(row["text"]))
        log.at[int(row["time"] - capture.start), key[0] + "_log-" + str(templates[key])] += 1
    grouped = defaultdict(list)
    for row in capture.spans:
        grouped[(row["service"], int(row["time"] - capture.start))].append(row)
    for service in sorted({r["service"] for r in capture.spans}):
        err[service + "_error"] = 0.
        latency[service + "_latency"] = 0.
    for (service, index), rows in grouped.items():
        err.at[index, service + "_error"] = sum(int(r["error"]) for r in rows)
        latency.at[index, service + "_latency"] = np.mean([r["duration"] for r in rows])
    return {"metric": metric, "logts": log, "tracets_err": err, "tracets_lat": latency}


def causal_inputs(reference, production):
    from root_teller.module1.features import normalize_template
    captures = [*reference, production]
    templates = {key: i for i, key in enumerate(sorted({(r["service"], normalize_template(r["text"])) for c in captures for r in c.logs}))}
    prepared = [frames(c, templates) for c in captures]
    boundary = sum(len(v["metric"]) for v in prepared[:-1])
    if boundary < 2 or len(prepared[-1]["metric"]) < 2:
        raise ValueError("Reference and production telemetry are required")
    data = {}
    for name in prepared[0]:
        frame = pd.concat([v[name] for v in prepared], ignore_index=True).fillna(0)
        frame.insert(0, "time", np.arange(len(frame), dtype=float))
        data[name] = frame
    return data, float(boundary)


@contextmanager
def causal_method(method, source_root=SOURCE_ROOT):
    from causallearn.utils.PCUtils import SkeletonDiscovery
    from causallearn.utils.cit import CIT
    root = Path(source_root) / ("torai" if method == "torai" else "multisource_rcd")
    with _LOCK, namespace(("RCAEval", "_rt_skeleton", "_rt_causal_graph")):
        package = types.ModuleType("RCAEval")
        package.__path__ = [str(root / "RCAEval")]
        subpackage = types.ModuleType("RCAEval.e2e")
        subpackage.__path__ = [str(root / "RCAEval/e2e")]
        sys.modules["RCAEval"], sys.modules["RCAEval.e2e"] = package, subpackage
        graph = _load("_rt_causal_graph", root / "lib/causallearn/graph/GraphClass.py")
        skeleton = _load("_rt_skeleton", root / "lib/causallearn/utils/PCUtils/SkeletonDiscovery.py")
        skeleton.CausalGraph = graph.CausalGraph
        cached = []

        def chisq(data, x, y, conditioning, cardinalities=None):
            if not cached or cached[0] is not data:
                cached[:] = [data, CIT(data, "chisq")]
            return cached[1](x, y, list(conditioning))

        skeleton.chisq = chisq
        old = getattr(SkeletonDiscovery, "local_skeleton_discovery", None)
        SkeletonDiscovery.local_skeleton_discovery = skeleton.local_skeleton_discovery
        try:
            name = "torai" if method == "torai" else "mmrcd"
            module = importlib.import_module("RCAEval.e2e." + name)
            if method == "torai":
                importlib.import_module("RCAEval.e2e.rcd").CI_TEST = chisq
            else:
                module.CI_TEST = chisq
            yield getattr(module, name)
        finally:
            if old is None:
                del SkeletonDiscovery.local_skeleton_discovery
            else:
                SkeletonDiscovery.local_skeleton_discovery = old


def causal_rank(method, reference, production, source_root=SOURCE_ROOT):
    data, boundary = causal_inputs(reference, production)
    if method == "multisource_rcd":
        data["metric"] = pd.concat([data["metric"], data["logts"].drop(columns="time")], axis=1)
        data["tracets_err"], data["tracets_lat"] = pd.DataFrame(), pd.DataFrame()
    with causal_method(method, source_root) as diagnose:
        result = diagnose(data, inject_time=boundary, dataset=None, gamma=5, localized=True, bins=5, seed=0)
    ordered = []
    for indicator in result["ranks"]:
        matches = [s for s in production.services if str(indicator) == s or str(indicator).startswith(s + "_")]
        if matches:
            service = max(matches, key=len)
            if service not in ordered:
                ordered.append(service)
    return ordered, {"upstream_ranks": [str(r) for r in result["ranks"]], "reference_samples": int(boundary),
                     "input_modalities": ["metric", "log"] if method == "multisource_rcd" else ["metric", "log", "trace"]}


@contextmanager
def nezha_modules(source_root=SOURCE_ROOT):
    class Logger:
        def __init__(self, *args):
            pass
        def getlog(self):
            return logging.getLogger("root_teller.nezha")

    names = ("log", "log_parsing", "alarm", "data_integrate", "pattern_miner", "pattern_ranker")
    with _LOCK, namespace(names):
        log = types.ModuleType("log")
        log.Logger = Logger
        sys.modules["log"] = log
        for name in names[1:]:
            _load(name, Path(source_root) / "nezha" / (name + ".py"))
        sys.modules["log_parsing"].pod_to_service = lambda text, pod: (text, pod.rsplit("-", 2)[0])
        yield sys.modules["data_integrate"], sys.modules["pattern_miner"], sys.modules["pattern_ranker"]


def nezha_graphs(capture, integrate, miner):
    spans = capture.spans
    by_service = defaultdict(list)
    for row in spans:
        by_service[row["service"]].append(row)
    for rows in by_service.values():
        rows.sort(key=lambda r: r["time"])
    log_rows = defaultdict(list)
    for row in capture.logs:
        matches = by_service[row["service"]]
        if not matches:
            continue
        position = bisect_left([r["time"] for r in matches], row["time"])
        choices = matches[max(0, position - 1):position + 1]
        span = min(choices, key=lambda r: abs(r["time"] - row["time"]))
        if abs(span["time"] - row["time"]) <= 30:
            log_rows[(span["trace"], span["id"])].append(row)
    traces = {}
    for row in sorted(spans, key=lambda r: (r["trace"], r["time"], r["id"])):
        trace = traces.setdefault(row["trace"], integrate.Trace(row["trace"]))
        pod = row["service"] + "-instance-0"
        span = integrate.Span(row["id"], row["parent"], pod)
        entries = [(row["time"], row["service"] + " " + row["operation"] + " start")]
        entries.extend((r["time"], r["text"]) for r in log_rows[(row["trace"], row["id"])])
        entries.append((max([row["time"] + row["duration"] / 1e6] + [t for t, _ in entries]),
                        row["service"] + " " + row["operation"] + " end"))
        for timestamp, text in entries:
            event_id = integrate.log_parsing(text, pod, miner)
            span.append_event(integrate.Event(event_id, pod, "sn", timestamp=int(timestamp * 1e9), spanid=row["id"], parentid=row["parent"]))
        span.sort_events()
        trace.append_spans(span)
    graphs = []
    for trace in traces.values():
        trace.sort_spans()
        graph = integrate.generate_event_graph(trace, miner)
        graph.get_support()
        graphs.append(graph)
    return graphs


def nezha_rank(reference, production, source_root=SOURCE_ROOT):
    from drain3 import TemplateMiner
    from drain3.template_miner_config import TemplateMinerConfig
    config = TemplateMinerConfig()
    config.profiling_enabled = False
    config.drain_sim_th = .9
    config.drain_depth = 4
    config.drain_max_children = 100
    config.drain_max_clusters = 1024
    config.drain_extra_delimiters = ["_"]
    miner = TemplateMiner(config=config)
    with nezha_modules(source_root) as (integrate, mining, ranking):
        normal = [g for c in reference for g in nezha_graphs(c, integrate, miner)]
        abnormal = nezha_graphs(production, integrate, miner)
        normal_support = mining.get_pattern_support(normal)
        abnormal_support = mining.get_pattern_support(abnormal)
        ranking.get_pattern = lambda *a, **k: (abnormal_support, abnormal, [])
        result, abnormal_patterns = ranking.pattern_ranker(normal_support, normal, "observed-production", "sn", miner)
        ordered = []
        for row in result:
            service = row["pod"].rsplit("-", 2)[0]
            if service in production.services and service not in ordered:
                ordered.append(service)
    return ordered, {"ranked_patterns": result, "abnormal_patterns": abnormal_patterns,
                     "reference_graphs": len(normal), "production_graphs": len(abnormal), "input_modalities": ["log", "trace"]}
