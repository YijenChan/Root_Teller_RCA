from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


def canonical(value: str, dataset: str) -> str:
    value = str(value).strip().lower()
    if dataset == "eadro_sn":
        value = re.sub(r"^socialnetwork-", "", value)
        value = re.sub(r"-\d+$", "", value)
        return {"nginx-web-server": "nginx-thrift"}.get(value, value)
    return {"frontendservice": "frontend", "frontend-external": "frontend",
            "redis-cart": "redis"}.get(value, value)


@dataclass(frozen=True)
class CaseSpec:
    dataset: str
    incident_id: str  # Private: never sent to an LLM.
    fold: int
    role: str
    target: str  # Private evaluator label.
    directory: Path
    start: float | None = None
    end: float | None = None

    @property
    def opaque_id(self) -> str:
        return self.dataset + "-" + hashlib.sha256(self.incident_id.encode()).hexdigest()[:16]


def load_specs(workspace: Path, manifests: Path, dataset: str, fold: int) -> list[CaseSpec]:
    assignments = json.loads((manifests / f"{dataset}_nested_folds.json").read_text(encoding="utf-8"))
    rows = [r for r in assignments if int(r["outer_fold"]) == fold]
    if not rows:
        raise ValueError("No assignments in requested fold")
    if dataset.startswith("re2_"):
        system = {"re2_ob": "RCAEval RE2-OB", "re2_tt": "RCAEval RE2-TT"}[dataset]
        with (manifests / "case_catalog.csv").open(encoding="utf-8-sig", newline="") as f:
            labels = {r["incident_id"]: r["root_cause_service"] for r in csv.DictReader(f)
                      if r["dataset_system"] == system}
        name = "RE2-OB" if dataset == "re2_ob" else "RE2-TT"
        raw = workspace / "dataset" / "RCAEval RE" / "RE2" / name / name
        result = [CaseSpec(dataset, r["incident_id"], fold, r["role"],
                           canonical(labels[r["incident_id"]], dataset), raw / r["incident_id"])
                  for r in rows]
    else:
        raw = workspace / "dataset" / "Eadro-SN" / "SN Dataset" / "SN Dataset" / "data"
        lookup = {}
        for ci, annotation in enumerate(sorted(raw.glob("SN.fault-*.json"))):
            meta = json.loads(annotation.read_text(encoding="utf-8"))
            cap = raw / ("SN." + annotation.name.removeprefix("SN.fault-").removesuffix(".json"))
            for fi, fault in enumerate(meta["faults"]):
                end = meta["faults"][fi + 1]["start"] if fi + 1 < len(meta["faults"]) else meta["end"]
                lookup[f"sn-c{ci:02d}-f{fi:02d}"] = (cap, float(fault["start"]), float(end),
                                                    canonical(fault["name"], dataset))
        result = []
        for r in rows:
            cap, start, end, target = lookup[r["incident_id"]]
            if canonical(r["root_cause_service"], dataset) != target:
                raise ValueError("SN manifest and annotation service mismatch")
            result.append(CaseSpec(dataset, r["incident_id"], fold, r["role"], target, cap, start, end))
    for spec in result:
        if not spec.directory.is_dir():
            raise FileNotFoundError(f"Missing raw case {spec.opaque_id}")
    roles = defaultdict(set)
    for spec in result:
        family = (spec.incident_id.split("/")[0] if dataset.startswith("re2_")
                  else spec.directory.name)
        roles[family].add(spec.role)
    if any(len(v) != 1 for v in roles.values()):
        raise ValueError("Family/capture leakage in fold assignments")
    return result


def _finite(value: float) -> float | None:
    return round(float(value), 5) if np.isfinite(value) else None


@lru_cache(maxsize=8)
def _json_capture(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@lru_cache(maxsize=2)
def _healthy_metrics(workspace: str) -> dict:
    base = Path(workspace) / "cache" / "rq1_multidataset_clean" / "eadro_sn" / "healthy_raw"
    captures = sorted(base.glob("SN.*"))
    if not captures:
        for interval in (1, 60):
            captures = sorted((base.parent / f"aggregation_{interval}s" / "healthy_raw").glob("SN.*"))
            if captures:
                break
    if not captures:
        raise FileNotFoundError("Extract the released SN no-fault captures before baseline evaluation")
    columns = defaultdict(list)
    for cap in captures:
        for f in sorted((cap / "metrics").glob("*.csv")):
            frame = pd.read_csv(f)
            for col in frame:
                if col != "timestamp":
                    x = pd.to_numeric(frame[col], errors="coerce").dropna().to_numpy()
                    columns[(canonical(f.stem, "eadro_sn"), col)].extend(x.tolist())
    return {key: (float(np.mean(v)), float(np.std(v))) for key, v in columns.items() if v}


def _metric_summaries(frame: pd.DataFrame, times: np.ndarray, start: float, end: float,
                      keys: dict[str, tuple[str, str]], reference: dict | None = None) -> dict:
    result = defaultdict(list)
    for col, (svc, kpi) in keys.items():
        values = pd.to_numeric(frame[col], errors="coerce").to_numpy(float)
        keep = (times >= start) & (times < end) & np.isfinite(values)
        if not np.any(keep):
            continue
        x = values[keep]
        prefix = (times >= start) & (times < start + 60) & np.isfinite(values)
        if reference is None:
            healthy = values[prefix]
            mean = float(np.mean(healthy)) if len(healthy) else 0.0
            std = float(np.std(healthy)) if len(healthy) else 0.0
        else:
            if (svc, kpi) not in reference:
                continue
            mean, std = reference[(svc, kpi)]
        denominator = max(std, abs(mean) * .01, 1e-6)
        bins = np.floor((times[keep] - start) / 60).astype(int)
        means = pd.Series(x).groupby(bins).mean()
        z = np.abs((means.to_numpy() - mean) / denominator)
        peak = int(np.argmax(z))
        result[svc].append({"metric": kpi, "reference_mean": _finite(mean),
                            "reference_std": _finite(std), "mean": _finite(np.mean(x)),
                            "p95": _finite(np.quantile(x, .95)), "max_bin_shift_z": _finite(z[peak]),
                            "peak_bin": int(means.index[peak]), "samples": int(len(x))})
    return {s: sorted(v, key=lambda r: -(r["max_bin_shift_z"] or 0))[:6] for s, v in result.items()}


def _logs_view(records: list[tuple[str, float, str, str]], start: float, end: float) -> tuple[dict, dict]:
    counts = defaultdict(Counter)
    lines = defaultdict(list)
    seen_lines = defaultdict(set)
    totals = Counter()
    for svc, timestamp, message, level in records:
        if not start <= timestamp < end:
            continue
        totals[svc] += 1
        warning = (level.upper() in {"WARN", "WARNING", "ERROR", "FATAL", "CRITICAL"}
                   or bool(re.search(r"(?i)\b(error|warn|warning|exception|failed|failure|timeout)\b", message)))
        if warning:
            message = message[:900]
            counts[svc][message] += 1
            if message not in seen_lines[svc] and len(lines[svc]) < 512:
                lines[svc].append({"id": f"L-{svc}-{len(lines[svc])}", "relative_seconds": round(timestamp-start, 3),
                                   "text": message})
                seen_lines[svc].add(message)
    summaries = {s: {"total_logs": totals[s], "warning_or_error_logs": sum(counts[s].values()),
                     "frequent_messages": [{"message": m, "count": n} for m, n in counts[s].most_common(8)]}
                 for s in sorted(totals)}
    return summaries, dict(lines)


def _traces_view(records: list[dict], start: float, end: float) -> tuple[dict, list]:
    spans = [r for r in records if start <= r["time"] < end]
    by_id = {(r["trace"], r["id"]): r["service"] for r in spans}
    services = defaultdict(list)
    edges = Counter()
    for r in spans:
        services[r["service"]].append(r)
        parent = by_id.get((r["trace"], r["parent"]))
        if parent and parent != r["service"]:
            edges[(parent, r["service"])] += 1
    view = {}
    for svc, group in services.items():
        durations = [r["duration_us"] for r in group if np.isfinite(r["duration_us"])]
        view[svc] = {"span_count": len(group), "error_spans": sum(r["error"] for r in group),
                     "mean_duration_us": _finite(np.mean(durations)) if durations else None,
                     "p95_duration_us": _finite(np.quantile(durations, .95)) if durations else None}
    return view, [{"caller": a, "callee": b, "calls": n} for (a, b), n in sorted(edges.items())]


def _re2(spec: CaseSpec):
    frame = pd.read_csv(spec.directory / "simple_metrics.csv", low_memory=False)
    times = pd.to_numeric(frame["time"], errors="coerce").to_numpy(float)
    start, end = float(np.nanmin(times)), float(np.nanmax(times)) + 1
    keys = {c: (canonical(c.split("_", 1)[0], spec.dataset), c.split("_", 1)[1])
            for c in frame if c != "time" and "_" in c}
    metric = _metric_summaries(frame, times, start, end, keys)
    logs = pd.read_csv(spec.directory / "logs.csv", low_memory=False)
    # RCAEval `time` is a wall-clock string. Use the released numeric timestamp.
    lt = pd.to_numeric(logs["timestamp"], errors="coerce").to_numpy(float)
    scale = 1e9 if np.nanmedian(lt) > 1e16 else 1e6 if np.nanmedian(lt) > 1e13 else 1
    lr = [(canonical(str(r.container_name), spec.dataset), float(t / scale), str(r.message), str(r.level))
          for r, t in zip(logs.itertuples(index=False), lt)]
    log, log_lines = _logs_view(lr, start, end)
    traces = pd.read_csv(spec.directory / "traces.csv", low_memory=False)
    records = []
    for r in traces.itertuples(index=False):
        timestamp = float(r.startTime) / 1e6
        code = float(r.statusCode) if pd.notna(r.statusCode) else 0
        records.append({"trace": str(r.traceID), "id": str(r.spanID), "parent": str(r.parentSpanID),
                        "service": canonical(r.serviceName, spec.dataset), "time": timestamp,
                        "duration_us": float(r.duration), "error": code >= 400 or code == 2})
    trace, edges = _traces_view(records, start, end)
    return start, end, metric, log, log_lines, trace, edges


def _sn(spec: CaseSpec, workspace: Path):
    assert spec.start is not None and spec.end is not None
    start, end = spec.start, spec.end
    metric = {}
    refs = _healthy_metrics(str(workspace))
    for f in sorted((spec.directory / "metrics").glob("*.csv")):
        frame = pd.read_csv(f)
        times = pd.to_numeric(frame["timestamp"], errors="coerce").to_numpy(float)
        svc = canonical(f.stem, spec.dataset)
        keys = {c: (svc, c) for c in frame if c != "timestamp"}
        metric.update(_metric_summaries(frame, times, start, end, keys, refs))
    lr = []
    for svc, lines in _json_capture(str(spec.directory / "logs.json")).items():
        for line in lines:
            match = re.match(r"\[(.*?)\]\s*<(.*?)>:\s*(.*)", line)
            if not match:
                continue
            # The benchmark's log timestamp is UTC; host local timezone must not affect parsing.
            stamp = datetime.strptime(match[1], "%Y-%b-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc).timestamp()
            lr.append((canonical(svc, spec.dataset), stamp, match[3], match[2]))
    log, log_lines = _logs_view(lr, start, end)
    records = []
    for group in _json_capture(str(spec.directory / "spans.json")):
        processes = group.get("processes", {})
        for span in group.get("spans", []):
            svc = canonical(processes.get(span.get("processID"), {}).get("serviceName", "unknown"), spec.dataset)
            tags = {r["key"]: r.get("value") for r in span.get("tags", [])}
            parents = [r["spanID"] for r in span.get("references", []) if r.get("refType") == "CHILD_OF"]
            records.append({"trace": span["traceID"], "id": span["spanID"], "parent": parents[0] if parents else "",
                            "service": svc, "time": float(span["startTime"]) / 1e6 - 8*3600,
                            "duration_us": float(span["duration"]), "error": bool(tags.get("error", False))})
    trace, edges = _traces_view(records, start, end)
    return start, end, metric, log, log_lines, trace, edges


def diagnostic_document(spec: CaseSpec, workspace: Path, cache: Path) -> dict:
    destination = cache / "documents" / spec.dataset / f"{spec.opaque_id}.json"
    if destination.exists():
        document = json.loads(destination.read_text(encoding="utf-8"))
        # Normalize cached pilot observations to the paper's log deduplication rule.
        for svc, lines in document.get("log_lines", {}).items():
            seen = set()
            unique = []
            for record in lines:
                if record["text"] not in seen:
                    unique.append(record)
                    seen.add(record["text"])
            document["log_lines"][svc] = unique
        return document
    start, end, metric, log, log_lines, trace, edges = _sn(spec, workspace) if spec.dataset == "eadro_sn" else _re2(spec)
    services = sorted(set(metric) | set(log) | (set(trace) - {"unknown"}))
    if spec.target not in services:
        raise ValueError("Evaluator target absent from observed service roster; do not silently repair")
    document = {"opaque_id": spec.opaque_id, "duration_seconds": round(end-start, 3),
                "services": services, "metric_reference": "released independent no-fault captures" if spec.dataset == "eadro_sn" else "fixed first 60 seconds",
                "metric": metric, "log": log, "trace": trace, "edges": edges, "log_lines": log_lines,
                "availability": {"metrics": bool(metric), "logs": bool(log), "traces": bool(trace)}}
    # Retrieval uses incident date only, not the fault injection metadata or opaque ID.
    document["recording_day"] = int(start // 86400)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return document


def diagnostic_text(document: dict) -> str:
    # A fixed diagnostic handler: retain every observed service, compact the
    # most changed KPIs and representative errors without using any label.
    services = {}
    for svc in document["services"]:
        metrics = document["metric"].get(svc, [])
        logs = document["log"].get(svc, {})
        traces = document["trace"].get(svc, {})
        services[svc] = {"metric_shifts": [[m["metric"], m["max_bin_shift_z"], m["peak_bin"]] for m in metrics[:3]],
                         "warning_logs": logs.get("warning_or_error_logs", 0),
                         "error_samples": [[x["message"][:180], x["count"]] for x in logs.get("frequent_messages", [])[:2]],
                         "trace": {k: traces.get(k) for k in ["span_count", "error_spans", "p95_duration_us"]}}
    return json.dumps({"duration_seconds": document["duration_seconds"], "metric_reference": document["metric_reference"],
                       "services": services, "edges": [[e["caller"], e["callee"]] for e in document["edges"]]},
                      ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def score_ranking(ranking: list[str], target: str) -> dict:
    rank = ranking.index(target) + 1 if target in ranking else None
    hits = [int(rank is not None and rank <= k) for k in range(1, 6)]
    return {"rank": rank, "A@1": hits[0], "A@3": hits[2], "Avg@5": sum(hits)/5}
