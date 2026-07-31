from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SCRIPT_ROOT = Path(__file__).resolve().parent
RCA_ROOT = Path()
SN_ROOT = Path()
OUTPUT_NAME = "rq2_three_datasets_outerfold_v1"
FINAL_ROOT = Path()
STAGING_ROOT = Path()
LEGACY_CASE_MANIFEST = SCRIPT_ROOT / "manifests" / "final_case_manifest.csv"

CONDITIONS = (
    "GMO_METRIC",
    "GMO_LOG",
    "GMO_TRACE",
    "IAMI_METRIC",
    "IAMI_LOG",
    "IAMI_TRACE",
)
SELECTED_MODALITY = {
    "GMO_METRIC": "metric",
    "GMO_LOG": "log",
    "GMO_TRACE": "trace",
    "IAMI_METRIC": "metric",
    "IAMI_LOG": "log",
    "IAMI_TRACE": "trace",
}
TRACE_CLOCK_SHIFT_SECONDS = 8 * 60 * 60
LOG_TIME = re.compile(
    r"\[(\d{4}-[A-Za-z]{3}-\d{2} \d\d:\d\d:\d\d(?:\.\d+)?)\]"
)


@dataclass(frozen=True)
class Case:
    dataset: str
    raw_incident_id: str
    opaque_id: str
    outer_fold: int
    root_cause_service: str
    fault_type: str
    inject_time: float
    window_start: float
    window_end: float
    source_path: str
    source_kind: str
    capture_id: str = ""
    repetition_id: str = ""


def log(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def opaque_id(dataset: str, raw_incident_id: str) -> str:
    return hashlib.sha256(f"{dataset}|{raw_incident_id}".encode()).hexdigest()[:20]


def canonical_sn_service(value: str) -> str:
    service = value.strip().lower()
    service = re.sub(r"^socialnetwork-", "", service)
    service = re.sub(r"-\d+$", "", service)
    return {"nginx-web-server": "nginx-thrift"}.get(service, service)


def csv_time_bounds(path: Path, column: str, scale: float = 1.0) -> tuple[float, float]:
    minimum = math.inf
    maximum = -math.inf
    seen = False
    for chunk in pd.read_csv(path, usecols=[column], chunksize=250_000):
        values = pd.to_numeric(chunk[column], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if not len(values):
            continue
        seen = True
        minimum = min(minimum, float(values.min()))
        maximum = max(maximum, float(values.max()))
    if not seen:
        raise ValueError(f"no numeric timestamps in {path}")
    return minimum / scale, maximum / scale


def scan_re2(system: str) -> list[Case]:
    system_root = RCA_ROOT / system / system
    family_dirs = sorted(
        path
        for path in system_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    fault_names = ("cpu", "delay", "disk", "loss", "mem", "socket")
    roots = sorted(
        {
            next(path.name[: -(len(fault) + 1)] for fault in fault_names if path.name.endswith(f"_{fault}"))
            for path in family_dirs
        }
    )
    root_index = {value: index for index, value in enumerate(roots)}
    fault_index = {value: index for index, value in enumerate(sorted(fault_names))}
    manifest = pd.read_csv(LEGACY_CASE_MANIFEST, encoding="utf-8-sig")
    manifest = manifest.loc[
        manifest["dataset_system"].astype(str) == f"RCAEval {system}"
    ]
    bounds = {
        str(row.incident_id): (
            min(
                float(row.metric_start),
                float(row.log_start),
                float(row.trace_start),
            ),
            max(
                float(row.metric_end),
                float(row.log_end),
                float(row.trace_end),
            ),
            float(row.inject_time),
        )
        for row in manifest.itertuples()
        if "/" in str(row.incident_id)
    }
    cases: list[Case] = []
    for family in family_dirs:
        fault = next(
            (value for value in fault_names if family.name.endswith(f"_{value}")),
            None,
        )
        if fault is None:
            raise ValueError(f"unrecognized RE2 family: {family}")
        root = family.name[: -(len(fault) + 1)]
        fold = (root_index[root] + fault_index[fault]) % 3
        for repetition in ("1", "2", "3"):
            source = family / repetition
            inject = float((source / "inject_time.txt").read_text().strip())
            raw_id = f"{family.name}/{repetition}"
            if raw_id not in bounds:
                raise KeyError(f"{system}: missing common-window metadata for {raw_id}")
            start, end, manifest_inject = bounds[raw_id]
            if abs(manifest_inject - inject) > 1e-6:
                raise ValueError(
                    f"{system}/{raw_id}: inject-time mismatch "
                    f"{inject} != {manifest_inject}"
                )
            if not start < inject < end:
                raise ValueError(
                    f"{system}/{family.name}/{repetition}: onset {inject} not inside "
                    f"common window [{start}, {end}]"
                )
            cases.append(
                Case(
                    dataset=f"RCAEval {system}",
                    raw_incident_id=raw_id,
                    opaque_id=opaque_id(f"RCAEval {system}", raw_id),
                    outer_fold=fold,
                    root_cause_service=root,
                    fault_type=fault,
                    inject_time=inject,
                    window_start=start,
                    window_end=end,
                    source_path=str(source),
                    source_kind="rcaeval_case_directory",
                    repetition_id=repetition,
                )
            )
    if len(cases) != 90:
        raise AssertionError(f"{system}: expected 90 cases, got {len(cases)}")
    return cases


def parse_sn_log_time(line: str) -> float | None:
    match = LOG_TIME.search(line)
    if not match:
        return None
    text = match.group(1)
    for fmt in ("%Y-%b-%d %H:%M:%S.%f", "%Y-%b-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).timestamp()
        except ValueError:
            pass
    return None


def scan_sn() -> list[Case]:
    data_root = SN_ROOT / "data"
    annotations = sorted(data_root.glob("SN.fault-*.json"))
    cases: list[Case] = []
    for capture_index, annotation_path in enumerate(annotations):
        stem = annotation_path.name.removeprefix("SN.fault-").removesuffix(".json")
        capture = data_root / f"SN.{stem}"
        metadata = json.loads(annotation_path.read_text(encoding="utf-8"))
        faults = metadata["faults"]
        for fault_index, fault in enumerate(faults):
            inject = float(fault["start"])
            previous_end = (
                float(metadata["start"])
                if fault_index == 0
                else float(faults[fault_index - 1]["start"])
                + float(faults[fault_index - 1]["duration"])
            )
            next_start = (
                float(faults[fault_index + 1]["start"])
                if fault_index + 1 < len(faults)
                else float(metadata["end"])
            )
            # Use the clean inter-injection history immediately before onset.
            # This preserves an observed pre-onset segment for IAMI while
            # excluding the active portion of the previous injected fault.
            start = min(previous_end, inject)
            end = next_start
            if not start < inject < end:
                raise ValueError(
                    f"Eadro-SN c{capture_index} f{fault_index}: onset outside window"
                )
            raw_id = f"sn-c{capture_index:02d}-f{fault_index:02d}"
            cases.append(
                Case(
                    dataset="Eadro-SN",
                    raw_incident_id=raw_id,
                    opaque_id=opaque_id("Eadro-SN", raw_id),
                    outer_fold=capture_index,
                    root_cause_service=canonical_sn_service(str(fault["name"])),
                    fault_type=str(fault["fault"]),
                    inject_time=inject,
                    window_start=start,
                    window_end=end,
                    source_path=str(capture),
                    source_kind="eadro_capture_directory",
                    capture_id=capture.name,
                    repetition_id=str(fault_index),
                )
            )
    if len(cases) != 36:
        raise AssertionError(f"Eadro-SN: expected 36 cases, got {len(cases)}")
    return cases


def read_rcaeval_edges(path: Path, max_start_ms: float | None = None) -> set[tuple[str, str]]:
    usecols = ["traceID", "spanID", "serviceName", "parentSpanID", "startTimeMillis"]
    seen: dict[tuple[str, str], str] = {}
    pending: dict[tuple[str, str], list[str]] = defaultdict(list)
    edges: set[tuple[str, str]] = set()
    for chunk in pd.read_csv(
        path,
        usecols=usecols,
        dtype=str,
        keep_default_na=False,
        chunksize=250_000,
    ):
        if max_start_ms is not None:
            times = pd.to_numeric(chunk["startTimeMillis"], errors="coerce")
            chunk = chunk.loc[times < max_start_ms]
        for trace, span, service, parent in zip(
            chunk["traceID"],
            chunk["spanID"],
            chunk["serviceName"],
            chunk["parentSpanID"],
        ):
            trace = trace.strip()
            span = span.strip()
            service = service.strip()
            parent = parent.strip()
            if not trace or not span or not service:
                continue
            key = (trace, span)
            for child in pending.pop(key, []):
                if child != service:
                    edges.add((service, child))
            seen[key] = service
            if parent:
                parent_service = seen.get((trace, parent))
                if parent_service is None:
                    pending[(trace, parent)].append(service)
                elif parent_service != service:
                    edges.add((parent_service, service))
    return edges


def sn_trace_span_time(span: dict[str, Any]) -> float:
    return float(span.get("startTime", 0)) / 1e6 - TRACE_CLOCK_SHIFT_SECONDS


def read_sn_edges(path: Path, start: float | None = None, end: float | None = None) -> set[tuple[str, str]]:
    traces = json.loads(path.read_text(encoding="utf-8"))
    edges: set[tuple[str, str]] = set()
    for trace in traces:
        processes = {
            key: canonical_sn_service(str(value.get("serviceName", "")))
            for key, value in trace.get("processes", {}).items()
        }
        retained = [
            span
            for span in trace.get("spans", [])
            if (start is None or sn_trace_span_time(span) >= start)
            and (end is None or sn_trace_span_time(span) < end)
        ]
        span_service = {
            str(span.get("spanID", "")): processes.get(span.get("processID"), "")
            for span in retained
        }
        for span in retained:
            child = processes.get(span.get("processID"), "")
            for reference in span.get("references", []):
                parent = span_service.get(str(reference.get("spanID", "")), "")
                if parent and child and parent != child:
                    edges.add((parent, child))
    return edges


def write_edges(path: Path, edges: Iterable[tuple[str, str]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(set(edges))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_service", "target_service"])
        writer.writerows(rows)
    return sha256_file(path)


def build_topologies(cases: list[Case], root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    result: dict[tuple[str, int], dict[str, Any]] = {}
    by_dataset: dict[str, list[Case]] = defaultdict(list)
    for case in cases:
        by_dataset[case.dataset].append(case)
    for dataset, dataset_cases in sorted(by_dataset.items()):
        folds = sorted({case.outer_fold for case in dataset_cases})
        case_edges: dict[str, set[tuple[str, str]]] = {}
        capture_edges: dict[str, set[tuple[str, str]]] = {}
        if dataset.startswith("RCAEval"):
            for index, case in enumerate(dataset_cases, 1):
                case_edges[case.opaque_id] = read_rcaeval_edges(
                    Path(case.source_path) / "traces.csv"
                )
                if index % 15 == 0 or index == len(dataset_cases):
                    log(
                        "topology_scan_progress",
                        dataset=dataset,
                        completed=index,
                        total=len(dataset_cases),
                    )
        else:
            for case in dataset_cases:
                if case.capture_id not in capture_edges:
                    capture_edges[case.capture_id] = read_sn_edges(
                        Path(case.source_path) / "spans.json"
                    )
        for fold in folds:
            development = [case for case in dataset_cases if case.outer_fold != fold]
            edges: set[tuple[str, str]] = set()
            if dataset.startswith("RCAEval"):
                for case in development:
                    edges.update(case_edges[case.opaque_id])
            else:
                seen_captures: set[str] = set()
                for case in development:
                    if case.capture_id in seen_captures:
                        continue
                    seen_captures.add(case.capture_id)
                    edges.update(capture_edges[case.capture_id])
            safe_dataset = dataset.replace(" ", "_")
            edge_path = root / "_manifests" / "topology" / safe_dataset / f"fold_{fold}.csv"
            digest = write_edges(edge_path, edges)
            result[(dataset, fold)] = {
                "edge_path": str(edge_path),
                "edge_sha256": digest,
                "edge_count": len(edges),
                "development_incidents": len(development),
            }
            log(
                "topology_built",
                dataset=dataset,
                fold=fold,
                edges=len(edges),
                development_incidents=len(development),
            )
    return result


def atomic_to_csv(frame: pd.DataFrame, destination: Path) -> None:
    temp = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    os.replace(temp, destination)


def rca_metric(case: Case, condition: str, out: Path) -> dict[str, Any]:
    source = Path(case.source_path) / "metrics.csv"
    frame = pd.read_csv(source, low_memory=False)
    value_columns = [column for column in frame.columns if column != "time"]
    values = frame[value_columns].apply(pd.to_numeric, errors="coerce").astype("float64")
    original_mask = values.notna().to_numpy(dtype=bool)
    times = pd.to_numeric(frame["time"], errors="coerce").to_numpy(dtype=float)
    if condition == "GMO_METRIC":
        available = np.zeros_like(original_mask, dtype=bool)
    else:
        available = original_mask & (times[:, None] < case.inject_time)
    frame[value_columns] = frame[value_columns].where(available, np.nan)
    atomic_to_csv(frame, out / "metrics.csv")
    np.savez_compressed(
        out / "metric_availability_mask.npz",
        mask=available,
        columns=np.asarray(value_columns, dtype=str),
        time=times,
    )
    return {
        "input_rows": len(frame),
        "available_rows": int(available.any(axis=1).sum()),
        "available_cells": int(available.sum()),
        "columns": len(value_columns),
    }


def rca_log(case: Case, condition: str, out: Path) -> dict[str, Any]:
    source = Path(case.source_path) / "logs.csv"
    header = pd.read_csv(source, nrows=0)
    if condition == "GMO_LOG":
        frame = header
        input_rows = sum(1 for _ in source.open("rb")) - 1
    else:
        original = pd.read_csv(source, low_memory=False)
        input_rows = len(original)
        times = pd.to_numeric(original["timestamp"], errors="coerce")
        frame = original.loc[times < case.inject_time * 1e9].copy()
    atomic_to_csv(frame, out / "logs.csv")
    (out / "log_availability.json").write_text(
        json.dumps(
            {
                "channel_available_before_onset": condition == "IAMI_LOG",
                "channel_available_at_or_after_onset": False,
                "unavailable_is_not_zero_events": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"input_rows": input_rows, "available_rows": len(frame)}


def rca_trace(
    case: Case,
    condition: str,
    out: Path,
    topology: dict[str, Any],
) -> dict[str, Any]:
    source = Path(case.source_path) / "traces.csv"
    header = pd.read_csv(source, nrows=0)
    if condition == "GMO_TRACE":
        frame = header
        input_rows = sum(1 for _ in source.open("rb")) - 1
        edge_source = Path(topology["edge_path"])
        shutil.copy2(edge_source, out / "topology_override.csv")
        topology_kind = "outer_development_historical"
        edge_hash = topology["edge_sha256"]
        edge_count = topology["edge_count"]
    else:
        original = pd.read_csv(source, low_memory=False)
        input_rows = len(original)
        times = pd.to_numeric(original["startTimeMillis"], errors="coerce")
        frame = original.loc[times < case.inject_time * 1e3].copy()
        edges = read_rcaeval_edges(
            source, max_start_ms=case.inject_time * 1e3
        )
        edge_hash = write_edges(out / "topology_override.csv", edges)
        edge_count = len(edges)
        topology_kind = "retained_pre_onset_spans"
    atomic_to_csv(frame, out / "traces.csv")
    (out / "trace_availability.json").write_text(
        json.dumps(
            {
                "topology_kind": topology_kind,
                "topology_sha256": edge_hash,
                "topology_edges": edge_count,
                "channel_available_at_or_after_onset": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "input_rows": input_rows,
        "available_rows": len(frame),
        "topology_edges": edge_count,
        "topology_sha256": edge_hash,
    }


def sn_metric(case: Case, condition: str, out: Path) -> dict[str, Any]:
    source_dir = Path(case.source_path) / "metrics"
    metric_dir = out / "metrics"
    mask_dir = out / "metric_masks"
    metric_dir.mkdir()
    mask_dir.mkdir()
    total_rows = available_rows = available_cells = 0
    for source in sorted(source_dir.glob("*.csv")):
        frame = pd.read_csv(source, low_memory=False)
        times = pd.to_numeric(frame["timestamp"], errors="coerce").to_numpy(float)
        in_window = (times >= case.window_start) & (times < case.window_end)
        frame = frame.loc[in_window].copy()
        times = times[in_window]
        value_columns = [column for column in frame.columns if column != "timestamp"]
        values = frame[value_columns].apply(pd.to_numeric, errors="coerce").astype("float64")
        original_mask = values.notna().to_numpy(dtype=bool)
        if condition == "GMO_METRIC":
            available = np.zeros_like(original_mask, dtype=bool)
        else:
            available = original_mask & (times[:, None] < case.inject_time)
        frame[value_columns] = frame[value_columns].where(available, np.nan)
        atomic_to_csv(frame, metric_dir / source.name)
        np.savez_compressed(
            mask_dir / f"{source.stem}.npz",
            mask=available,
            columns=np.asarray(value_columns, dtype=str),
            timestamp=times,
        )
        total_rows += len(frame)
        available_rows += int(available.any(axis=1).sum())
        available_cells += int(available.sum())
    return {
        "input_rows": total_rows,
        "available_rows": available_rows,
        "available_cells": available_cells,
        "service_files": len(list(metric_dir.glob("*.csv"))),
    }


def sn_log(case: Case, condition: str, out: Path) -> dict[str, Any]:
    payload = json.loads((Path(case.source_path) / "logs.json").read_text(encoding="utf-8"))
    result: dict[str, list[str]] = {}
    parsed = retained = unparsed = 0
    for service, lines in payload.items():
        kept: list[str] = []
        for raw in lines:
            line = str(raw)
            timestamp = parse_sn_log_time(line)
            if timestamp is None:
                unparsed += 1
                continue
            if not (case.window_start <= timestamp < case.window_end):
                continue
            parsed += 1
            if condition == "IAMI_LOG" and timestamp < case.inject_time:
                kept.append(line)
        result[service] = kept
        retained += len(kept)
    if condition == "GMO_LOG":
        result = {service: [] for service in payload}
        retained = 0
    (out / "logs.json").write_text(
        json.dumps(result, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out / "log_availability.json").write_text(
        json.dumps(
            {
                "channel_available_before_onset": condition == "IAMI_LOG",
                "channel_available_at_or_after_onset": False,
                "unavailable_is_not_zero_events": True,
                "unparsed_lines_excluded": unparsed,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "input_window_rows": parsed,
        "available_rows": retained,
        "unparsed_source_lines": unparsed,
    }


def filter_sn_traces(
    source: Path, start: float, end: float
) -> tuple[list[dict[str, Any]], int]:
    traces = json.loads(source.read_text(encoding="utf-8"))
    output: list[dict[str, Any]] = []
    retained_count = 0
    for trace in traces:
        spans = [
            span
            for span in trace.get("spans", [])
            if start <= sn_trace_span_time(span) < end
        ]
        if not spans:
            continue
        retained_ids = {str(span.get("spanID", "")) for span in spans}
        cleaned = []
        for span in spans:
            cloned = dict(span)
            cloned["references"] = [
                reference
                for reference in span.get("references", [])
                if str(reference.get("spanID", "")) in retained_ids
            ]
            cleaned.append(cloned)
        cloned_trace = dict(trace)
        cloned_trace["spans"] = cleaned
        output.append(cloned_trace)
        retained_count += len(cleaned)
    return output, retained_count


def sn_trace(
    case: Case,
    condition: str,
    out: Path,
    topology: dict[str, Any],
) -> dict[str, Any]:
    source = Path(case.source_path) / "spans.json"
    if condition == "GMO_TRACE":
        traces: list[dict[str, Any]] = []
        available_rows = 0
        shutil.copy2(Path(topology["edge_path"]), out / "topology_override.csv")
        edge_hash = topology["edge_sha256"]
        edge_count = topology["edge_count"]
        topology_kind = "outer_development_historical"
    else:
        traces, available_rows = filter_sn_traces(
            source, case.window_start, case.inject_time
        )
        temp = out / "spans.json.tmp"
        temp.write_text(json.dumps(traces, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temp, out / "spans.json")
        edges = read_sn_edges(source, case.window_start, case.inject_time)
        edge_hash = write_edges(out / "topology_override.csv", edges)
        edge_count = len(edges)
        topology_kind = "retained_pre_onset_spans"
    if condition == "GMO_TRACE":
        (out / "spans.json").write_text("[]\n", encoding="utf-8")
    (out / "trace_availability.json").write_text(
        json.dumps(
            {
                "topology_kind": topology_kind,
                "topology_sha256": edge_hash,
                "topology_edges": edge_count,
                "channel_available_at_or_after_onset": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "available_spans": available_rows,
        "topology_edges": edge_count,
        "topology_sha256": edge_hash,
    }


def materialize_view(
    case: Case,
    condition: str,
    root: Path,
    topology: dict[str, Any],
) -> dict[str, Any]:
    view = root / case.dataset.replace(" ", "_") / condition / case.opaque_id
    metadata_path = view / "condition_metadata.json"
    if metadata_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    partial = view.with_name(f"{view.name}.partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True, exist_ok=False)
    modality = SELECTED_MODALITY[condition]
    if case.source_kind == "rcaeval_case_directory":
        if modality == "metric":
            counts = rca_metric(case, condition, partial)
        elif modality == "log":
            counts = rca_log(case, condition, partial)
        else:
            counts = rca_trace(case, condition, partial, topology)
    else:
        if modality == "metric":
            counts = sn_metric(case, condition, partial)
        elif modality == "log":
            counts = sn_log(case, condition, partial)
        else:
            counts = sn_trace(case, condition, partial, topology)
    metadata = {
        "schema_version": 1,
        "opaque_incident_id": case.opaque_id,
        "dataset": case.dataset,
        "outer_fold": case.outer_fold,
        "condition": condition,
        "selected_modality": modality,
        "unchanged_modalities": "resolved through private_manifest.csv; never copied into prompts",
        "availability_semantics": (
            "unavailable throughout the complete observation window"
            if condition.startswith("GMO_")
            else "available before the private incident boundary and unavailable thereafter"
        ),
        "private_boundary_not_exposed_as_feature": True,
        "counts": counts,
    }
    (partial / "condition_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    view.parent.mkdir(parents=True, exist_ok=True)
    os.replace(partial, view)
    return metadata


def write_private_manifest(cases: list[Case], root: Path) -> Path:
    destination = root / "_manifests" / "private_manifest.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(cases[0]).keys())
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in sorted(cases, key=lambda item: (item.dataset, item.opaque_id)):
            writer.writerow(asdict(case))
    return destination


def write_condition_manifest(
    cases: list[Case],
    topologies: dict[tuple[str, int], dict[str, Any]],
    root: Path,
) -> Path:
    destination = root / "_manifests" / "condition_manifest.csv"
    fields = [
        "opaque_incident_id",
        "dataset",
        "outer_fold",
        "condition",
        "selected_modality",
        "view_relative_path",
        "topology_sha256",
    ]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in sorted(cases, key=lambda item: (item.dataset, item.opaque_id)):
            for condition in CONDITIONS:
                topology_hash = (
                    topologies[(case.dataset, case.outer_fold)]["edge_sha256"]
                    if condition == "GMO_TRACE"
                    else ""
                )
                writer.writerow(
                    {
                        "opaque_incident_id": case.opaque_id,
                        "dataset": case.dataset,
                        "outer_fold": case.outer_fold,
                        "condition": condition,
                        "selected_modality": SELECTED_MODALITY[condition],
                        "view_relative_path": (
                            Path(case.dataset.replace(" ", "_"))
                            / condition
                            / case.opaque_id
                        ).as_posix(),
                        "topology_sha256": topology_hash,
                    }
                )
    return destination


def validate_output(cases: list[Case], root: Path) -> dict[str, Any]:
    errors: list[str] = []
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for case in cases:
        for condition in CONDITIONS:
            view = (
                root
                / case.dataset.replace(" ", "_")
                / condition
                / case.opaque_id
            )
            metadata_path = view / "condition_metadata.json"
            if not metadata_path.exists():
                errors.append(f"missing metadata: {view}")
                continue
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            text = metadata_path.read_text(encoding="utf-8")
            for forbidden in (
                case.raw_incident_id,
                case.root_cause_service,
                case.source_path,
                str(case.inject_time),
            ):
                if forbidden and forbidden in text:
                    errors.append(f"private field leaked in {metadata_path}: {forbidden}")
            modality = SELECTED_MODALITY[condition]
            if modality == "metric":
                expected = (
                    view / "metrics.csv"
                    if case.source_kind == "rcaeval_case_directory"
                    else view / "metrics"
                )
            elif modality == "log":
                expected = (
                    view / "logs.csv"
                    if case.source_kind == "rcaeval_case_directory"
                    else view / "logs.json"
                )
            else:
                expected = (
                    view / "traces.csv"
                    if case.source_kind == "rcaeval_case_directory"
                    else view / "spans.json"
                )
            if not expected.exists():
                errors.append(f"missing selected modality: {expected}")
            if modality == "trace" and not (view / "topology_override.csv").exists():
                errors.append(f"missing topology override: {view}")
            counts[case.dataset][condition] += 1
            if metadata.get("condition") != condition:
                errors.append(f"condition mismatch: {metadata_path}")
    expected_by_dataset = {"RCAEval RE2-OB": 90, "RCAEval RE2-TT": 90, "Eadro-SN": 36}
    for dataset, expected in expected_by_dataset.items():
        for condition in CONDITIONS:
            actual = counts[dataset][condition]
            if actual != expected:
                errors.append(f"{dataset}/{condition}: {actual} != {expected}")
    report = {
        "status": "PASS" if not errors else "FAIL",
        "incidents": len(cases),
        "conditions": len(CONDITIONS),
        "views": sum(sum(value.values()) for value in counts.values()),
        "by_dataset_condition": {
            dataset: dict(values) for dataset, values in sorted(counts.items())
        },
        "errors": errors,
    }
    report_path = root / "_manifests" / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if errors:
        raise AssertionError(json.dumps(report, indent=2))
    return report


def write_readme(cases: list[Case], report: dict[str, Any], root: Path) -> None:
    sizes = sum(
        path.stat().st_size for path in root.rglob("*") if path.is_file()
    )
    text = f"""# RQ2 GMO/IAMI corrupted telemetry views

This directory contains the fold-aligned telemetry-unavailability inputs for
RCAEval RE2-OB, RCAEval RE2-TT, and Eadro-SN.

- Incidents: {len(cases)}
- Conditions per incident: {len(CONDITIONS)}
- Materialized views: {report['views']}
- Size at validation: {sizes / (1024 ** 3):.3f} GiB
- Validation: {report['status']}

## Conditions

`GMO_METRIC`, `GMO_LOG`, `GMO_TRACE`, `IAMI_METRIC`, `IAMI_LOG`, and
`IAMI_TRACE` implement the manuscript definitions. Only the damaged modality
is materialized. Unchanged modalities are resolved internally through the
private manifest and must not be copied into an LLM prompt with their source
path.

## Fold policy

- RE2-OB and RE2-TT: three grouped outer folds. The three repeated injections
  of each root-service/fault family stay in the same fold.
- Eadro-SN: four-fold leave-one-capture-out.
- GMO-Trace: the call map is constructed only from the corresponding
  outer-development folds.
- IAMI-Trace: invocation edges are reconstructed only from retained pre-onset
  spans.

## Eadro-SN interval policy

The pre-onset segment starts after the preceding injected fault ends (or at
the capture start for the first injection). The interval ends at the next
fault onset or capture end. This retains a clean pre-onset segment for IAMI
without including the active portion of the previous injected fault.

`_manifests/private_manifest.csv` contains evaluator-only labels, original
paths, and construction timestamps. It must never be exposed as inference
input. Public view directories use opaque incident identifiers.
"""
    (root / "README.md").write_text(text, encoding="utf-8")


def prepare(dry_run: bool, resume: bool) -> None:
    cases = scan_re2("RE2-OB") + scan_re2("RE2-TT") + scan_sn()
    summary: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for case in cases:
        summary[case.dataset][case.outer_fold] += 1
    log(
        "population",
        incidents=len(cases),
        views=len(cases) * len(CONDITIONS),
        folds={key: dict(value) for key, value in summary.items()},
    )
    if dry_run:
        return
    if FINAL_ROOT.exists():
        raise FileExistsError(
            f"validated output already exists: {FINAL_ROOT}; move it explicitly before rebuilding"
        )
    if STAGING_ROOT.exists() and not resume:
        raise FileExistsError(
            f"staging output exists: {STAGING_ROOT}; use --resume or remove it explicitly"
        )
    STAGING_ROOT.mkdir(parents=True, exist_ok=True)
    private_manifest = write_private_manifest(cases, STAGING_ROOT)
    topologies = build_topologies(cases, STAGING_ROOT)
    condition_manifest = write_condition_manifest(cases, topologies, STAGING_ROOT)
    total = len(cases) * len(CONDITIONS)
    completed = 0
    for case in sorted(cases, key=lambda item: (item.dataset, item.opaque_id)):
        topology = topologies[(case.dataset, case.outer_fold)]
        for condition in CONDITIONS:
            materialize_view(case, condition, STAGING_ROOT, topology)
            completed += 1
            if completed % 25 == 0 or completed == total:
                log("progress", completed=completed, total=total)
    report = validate_output(cases, STAGING_ROOT)
    write_readme(cases, report, STAGING_ROOT)
    provenance = {
        "private_manifest_sha256": sha256_file(private_manifest),
        "condition_manifest_sha256": sha256_file(condition_manifest),
        "script_sha256": sha256_file(Path(__file__)),
        "python": sys.version,
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }
    (STAGING_ROOT / "_manifests" / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(STAGING_ROOT, FINAL_ROOT)
    log("complete", output=str(FINAL_ROOT), report=report)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize six GMO/IAMI views for RCAEval RE2-OB, "
            "RCAEval RE2-TT, and Eadro-SN."
        )
    )
    parser.add_argument(
        "--re2-root",
        type=Path,
        required=True,
        help="Local RE2 directory containing RE2-OB/RE2-OB and RE2-TT/RE2-TT",
    )
    parser.add_argument(
        "--eadro-sn-root",
        type=Path,
        required=True,
        help="Local Eadro-SN root containing the data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Final output directory for the 18 dataset-condition groups",
    )
    parser.add_argument(
        "--case-manifest",
        type=Path,
        default=SCRIPT_ROOT / "manifests" / "final_case_manifest.csv",
        help="Protocol manifest for RCAEval common observation windows",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    global RCA_ROOT, SN_ROOT, FINAL_ROOT, STAGING_ROOT, LEGACY_CASE_MANIFEST
    RCA_ROOT = args.re2_root.expanduser().resolve()
    SN_ROOT = args.eadro_sn_root.expanduser().resolve()
    FINAL_ROOT = args.output_dir.expanduser().resolve()
    STAGING_ROOT = FINAL_ROOT.with_name(f"{FINAL_ROOT.name}.incomplete")
    LEGACY_CASE_MANIFEST = args.case_manifest.expanduser().resolve()

    required = {
        "RE2 root": RCA_ROOT,
        "Eadro-SN root": SN_ROOT,
        "case manifest": LEGACY_CASE_MANIFEST,
    }
    missing = [f"{label}: {path}" for label, path in required.items() if not path.exists()]
    if missing:
        parser.error("missing required input(s): " + "; ".join(missing))
    FINAL_ROOT.parent.mkdir(parents=True, exist_ok=True)
    prepare(args.dry_run, args.resume)


if __name__ == "__main__":
    main()
