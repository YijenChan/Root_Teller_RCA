from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    CONDITIONS,
    SERVICE_INDEX,
    SERVICES,
    FeatureConfig,
    Paths,
    canonical_service,
)
from root_teller.paths import workspace_root


_NUMBER = re.compile(
    r"(?<![A-Za-z])(?:[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)(?![A-Za-z])"
)
_HEX = re.compile(r"\b(?:0x)?[0-9a-fA-F]{8,}\b")
_UUID = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_SPACE = re.compile(r"\s+")
_SBERT_MODEL = None
_SBERT_CACHE: dict[str, np.ndarray] = {}


@dataclass(frozen=True)
class CaseSpec:
    incident_id: str
    split: str
    eligible: bool
    root_cause_service: str
    fault_type: str
    inject_time: float


def load_case_specs(paths: Paths) -> list[CaseSpec]:
    frame = pd.read_csv(paths.manifest_dir / "split_manifest.csv")
    frame = frame.loc[frame["dataset_system"].eq("RCAEval RE2-OB")]
    records: list[CaseSpec] = []
    for row in frame.itertuples(index=False):
        records.append(
            CaseSpec(
                incident_id=str(row.incident_id),
                split=str(row.split),
                eligible=bool(row.eligible),
                root_cause_service=canonical_service(row.root_cause_service),
                fault_type=str(row.fault_type),
                inject_time=float(row.inject_time),
            )
        )
    return records


def normalize_template(value: object) -> str:
    text = str(value).strip().lower()
    text = _UUID.sub("<uuid>", text)
    text = _HEX.sub("<hex>", text)
    text = _NUMBER.sub("<num>", text)
    return _SPACE.sub(" ", text)


def _hash_template(template: str, dimension: int) -> tuple[int, float]:
    digest = hashlib.blake2b(template.encode("utf-8"), digest_size=8).digest()
    integer = int.from_bytes(digest, byteorder="little", signed=False)
    return integer % dimension, 1.0 if ((integer >> 8) & 1) else -1.0


def _sbert_embeddings(
    templates: list[str], config: FeatureConfig
) -> dict[str, np.ndarray]:
    global _SBERT_MODEL
    missing = [template for template in templates if template not in _SBERT_CACHE]
    if missing:
        if _SBERT_MODEL is None:
            from sentence_transformers import SentenceTransformer

            _SBERT_MODEL = SentenceTransformer(
                config.log_sbert_model,
                cache_folder=str(workspace_root() / "cache" / "huggingface"),
            )
        encoded = _SBERT_MODEL.encode(
            missing,
            batch_size=256,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        for template, embedding in zip(missing, encoded):
            _SBERT_CACHE[template] = np.asarray(embedding, dtype=np.float32)
    return {template: _SBERT_CACHE[template] for template in templates}


def _metric_group(column: str) -> str:
    name = column.lower()
    if "latency" in name or "duration" in name:
        return "latency"
    if "error" in name or "fail" in name or "drop" in name:
        return "error"
    if "request" in name:
        return "request"
    if "socket" in name:
        return "socket"
    if "cpu" in name:
        return "cpu"
    if "memory" in name:
        return "memory"
    if "disk" in name or "blkio" in name or "_fs-" in name:
        return "disk"
    if "network" in name:
        return "network"
    if "bytes" in name:
        return "bytes"
    return "other"


def _view_dir(paths: Paths, spec: CaseSpec, condition: str) -> Path:
    family = "GMO" if condition.startswith("GMO_") else "IAMI"
    return (
        paths.corrupted_root
        / family
        / condition
        / "RE2"
        / "RE2-OB"
        / "RE2-OB"
        / Path(spec.incident_id)
    )


def _selected_path(
    paths: Paths, spec: CaseSpec, condition: str, modality: str
) -> Path:
    clean_name = {
        "METRIC": "metrics.csv",
        "LOG": "logs.csv",
        "TRACE": "traces.csv",
    }[modality]
    if condition.endswith(f"_{modality}"):
        return _view_dir(paths, spec, condition) / clean_name
    return paths.clean_root / Path(spec.incident_id) / clean_name


def _time_grid(paths: Paths, spec: CaseSpec, seconds: int) -> tuple[float, int]:
    metric_path = paths.clean_root / Path(spec.incident_id) / "metrics.csv"
    timestamps = pd.read_csv(metric_path, usecols=["time"])["time"]
    timestamps = pd.to_numeric(timestamps, errors="coerce").dropna()
    start = float(timestamps.min())
    duration = max(float(timestamps.max()) - start, 1.0)
    bins = max(1, int(math.ceil(duration / seconds)))
    return start, bins


def _bin_indices(
    timestamps: np.ndarray, start: float, bins: int, seconds: int
) -> np.ndarray:
    result = np.floor((timestamps - start) / seconds).astype(np.int64)
    return np.clip(result, 0, bins - 1)


def _condition_bin_mask(
    condition: str,
    modality: str,
    start: float,
    bins: int,
    seconds: int,
    inject_time: float,
) -> np.ndarray:
    if condition == f"GMO_{modality}":
        return np.zeros(bins, dtype=bool)
    if condition == f"IAMI_{modality}":
        starts = start + np.arange(bins, dtype=np.float64) * seconds
        return starts < inject_time
    return np.ones(bins, dtype=bool)


def _safe_nanmean(values: np.ndarray, axis: int) -> np.ndarray:
    counts = np.sum(np.isfinite(values), axis=axis)
    sums = np.nansum(values, axis=axis)
    output = np.zeros_like(sums, dtype=np.float32)
    np.divide(sums, counts, out=output, where=counts > 0)
    return output


def extract_metrics(
    path: Path,
    condition: str,
    spec: CaseSpec,
    start: float,
    bins: int,
    config: FeatureConfig,
) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path, low_memory=False)
    timestamps = pd.to_numeric(frame["time"], errors="coerce").to_numpy(float)
    bin_index = _bin_indices(timestamps, start, bins, config.bin_seconds)
    output = np.zeros(
        (len(SERVICES), bins, len(config.metric_groups)), dtype=np.float32
    )
    cell_counts = np.zeros_like(output, dtype=np.int32)
    columns: dict[tuple[int, int], list[str]] = defaultdict(list)
    for column in frame.columns:
        if column == "time" or "_" not in column:
            continue
        prefix = column.split("_", 1)[0]
        if prefix.startswith("gke-") or prefix == "loadgenerator":
            continue
        service = canonical_service(prefix)
        if service not in SERVICE_INDEX:
            continue
        group = _metric_group(column)
        columns[(SERVICE_INDEX[service], config.metric_groups.index(group))].append(
            column
        )

    for (service_index, group_index), selected in columns.items():
        values = (
            frame[selected]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=float, copy=True)
        )
        for offset, column in enumerate(selected):
            if "total" in column.lower():
                delta = np.diff(values[:, offset], prepend=np.nan)
                delta[delta < 0] = np.nan
                values[:, offset] = delta
        values = np.log1p(np.abs(values))
        row_values = _safe_nanmean(values, axis=1)
        valid_rows = np.any(np.isfinite(values), axis=1)
        for bin_id in range(bins):
            chosen = (bin_index == bin_id) & valid_rows
            if np.any(chosen):
                output[service_index, bin_id, group_index] = float(
                    np.mean(row_values[chosen])
                )
                cell_counts[service_index, bin_id, group_index] = int(
                    np.sum(chosen)
                )

    base_mask = _condition_bin_mask(
        condition,
        "METRIC",
        start,
        bins,
        config.bin_seconds,
        spec.inject_time,
    )
    service_observed = np.any(cell_counts > 0, axis=2)
    mask = service_observed & base_mask[None, :]
    output[~mask] = 0.0
    return output, mask


def _log_columns(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return next(csv.reader(handle))


def extract_logs(
    path: Path,
    condition: str,
    spec: CaseSpec,
    start: float,
    bins: int,
    config: FeatureConfig,
) -> tuple[np.ndarray, np.ndarray]:
    content_dimension = config.log_content_dim
    dimension = content_dimension + config.log_extra_dim
    output = np.zeros((len(SERVICES), bins, dimension), dtype=np.float32)
    content_counts = np.zeros((len(SERVICES), bins), dtype=np.float32)
    header = _log_columns(path)
    if path.stat().st_size > len(",".join(header)) + 4:
        usecols = ["timestamp", "container_name", "message"]
        for optional in ("log_template", "level", "error"):
            if optional in header:
                usecols.append(optional)
        for chunk in pd.read_csv(
            path,
            usecols=usecols,
            dtype=str,
            keep_default_na=False,
            chunksize=250_000,
            low_memory=False,
        ):
            timestamps = (
                pd.to_numeric(chunk["timestamp"], errors="coerce").to_numpy(float)
                / 1e9
            )
            bin_ids = _bin_indices(timestamps, start, bins, config.bin_seconds)
            services = chunk["container_name"].map(canonical_service)
            # Always derive templates from the same source field. Some later
            # RCAEval repetitions omit ``log_template`` entirely; mixing the
            # released template column in train with message-derived templates
            # in validation creates a collection-version shortcut.
            templates = chunk["message"].map(normalize_template)
            if "level" in chunk:
                levels = chunk["level"].str.lower()
            else:
                levels = pd.Series("", index=chunk.index)
            if "error" in chunk:
                errors = chunk["error"].str.lower()
            else:
                errors = pd.Series("", index=chunk.index)

            table = pd.DataFrame(
                {
                    "service": services,
                    "bin": bin_ids,
                    "template": templates,
                    "is_error": (
                        levels.isin(["error", "fatal", "critical"])
                        | errors.str.len().gt(0)
                    ).astype(np.float32),
                }
            )
            table = table.loc[table["service"].isin(SERVICE_INDEX)]
            grouped = (
                table.groupby(["service", "bin", "template"], sort=False)
                .agg(count=("template", "size"), errors=("is_error", "sum"))
                .reset_index()
            )
            sbert = (
                _sbert_embeddings(
                    grouped["template"].astype(str).unique().tolist(), config
                )
                if config.log_backend == "sbert"
                else None
            )
            for row in grouped.itertuples(index=False):
                service_index = SERVICE_INDEX[row.service]
                bin_id = int(row.bin)
                if sbert is None:
                    hash_index, sign = _hash_template(
                        str(row.template), config.log_hash_dim
                    )
                    output[service_index, bin_id, hash_index] += (
                        sign * math.log1p(float(row.count))
                    )
                else:
                    output[
                        service_index, bin_id, :content_dimension
                    ] += sbert[str(row.template)] * float(row.count)
                    content_counts[service_index, bin_id] += float(row.count)
                output[
                    service_index, bin_id, content_dimension
                ] += float(row.count)
                output[
                    service_index, bin_id, content_dimension + 1
                ] += float(row.errors)
                output[
                    service_index, bin_id, content_dimension + 2
                ] += 1.0

    if config.log_backend == "sbert":
        np.divide(
            output[:, :, :content_dimension],
            content_counts[:, :, None],
            out=output[:, :, :content_dimension],
            where=content_counts[:, :, None] > 0,
        )
    output[:, :, content_dimension:] = np.log1p(
        output[:, :, content_dimension:]
    )
    mask_1d = _condition_bin_mask(
        condition,
        "LOG",
        start,
        bins,
        config.bin_seconds,
        spec.inject_time,
    )
    mask = np.broadcast_to(mask_1d, (len(SERVICES), bins)).copy()
    output[~mask] = 0.0
    return output, mask


def extract_traces(
    paths: Paths,
    path: Path,
    condition: str,
    spec: CaseSpec,
    start: float,
    bins: int,
    config: FeatureConfig,
) -> tuple[np.ndarray, np.ndarray, set[tuple[str, str]]]:
    output = np.zeros((len(SERVICES), bins, config.trace_dim), dtype=np.float32)
    aggregate: dict[tuple[int, int], list[float]] = defaultdict(
        lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    seen: dict[tuple[str, str], str] = {}
    pending: dict[tuple[str, str], list[str]] = defaultdict(list)
    edges: set[tuple[str, str]] = set()
    usecols = [
        "traceID",
        "spanID",
        "serviceName",
        "methodName",
        "startTimeMillis",
        "duration",
        "statusCode",
        "parentSpanID",
    ]
    if path.stat().st_size > 160:
        for chunk in pd.read_csv(
            path,
            usecols=usecols,
            dtype=str,
            keep_default_na=False,
            chunksize=250_000,
            low_memory=False,
        ):
            timestamps = (
                pd.to_numeric(chunk["startTimeMillis"], errors="coerce")
                .fillna(start * 1000)
                .to_numpy(float)
                / 1000
            )
            bin_ids = _bin_indices(timestamps, start, bins, config.bin_seconds)
            duration = (
                pd.to_numeric(chunk["duration"], errors="coerce")
                .fillna(0)
                .clip(lower=0)
                .to_numpy(float)
            )
            status = (
                pd.to_numeric(chunk["statusCode"], errors="coerce")
                .fillna(0)
                .to_numpy(float)
            )
            methods = chunk["methodName"].astype(str).to_numpy()
            services = chunk["serviceName"].map(canonical_service).to_numpy()
            for service, bin_id, latency, code, method in zip(
                services, bin_ids, duration, status, methods
            ):
                if service not in SERVICE_INDEX:
                    continue
                key = (SERVICE_INDEX[service], int(bin_id))
                values = aggregate[key]
                values[0] += 1.0
                values[1] += float(code != 0)
                values[2] += float(latency)
                values[3] = max(values[3], float(latency))
                values[4] += float(len(method))
                values[5] += float("error" in method.lower())

            for trace_id, span_id, service, parent_id in zip(
                chunk["traceID"],
                chunk["spanID"],
                services,
                chunk["parentSpanID"],
            ):
                if service not in SERVICE_INDEX:
                    continue
                key = (str(trace_id), str(span_id))
                for child in pending.pop(key, []):
                    if child != service:
                        edges.add((service, child))
                seen[key] = service
                if parent_id:
                    parent_key = (str(trace_id), str(parent_id))
                    parent = seen.get(parent_key)
                    if parent is None:
                        pending[parent_key].append(service)
                    elif parent != service:
                        edges.add((parent, service))

    for (service_index, bin_id), values in aggregate.items():
        count = max(values[0], 1.0)
        output[service_index, bin_id] = np.array(
            [
                math.log1p(values[0]),
                values[1] / count,
                math.log1p(values[2] / count),
                math.log1p(values[3]),
                values[4] / count / 32.0,
                values[5] / count,
            ],
            dtype=np.float32,
        )

    mask_1d = _condition_bin_mask(
        condition,
        "TRACE",
        start,
        bins,
        config.bin_seconds,
        spec.inject_time,
    )
    mask = np.broadcast_to(mask_1d, (len(SERVICES), bins)).copy()
    output[~mask] = 0.0

    override = _view_dir(paths, spec, condition) / "topology_override.csv"
    if condition.endswith("_TRACE") and override.exists():
        edges = set()
        with override.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                source = canonical_service(row["source_service"])
                target = canonical_service(row["target_service"])
                if source in SERVICE_INDEX and target in SERVICE_INDEX:
                    edges.add((source, target))
    return output, mask, edges


def extract_case(
    paths: Paths,
    spec: CaseSpec,
    condition: str,
    config: FeatureConfig,
) -> dict[str, object]:
    if condition not in CONDITIONS:
        raise ValueError(f"unsupported condition: {condition}")
    start, bins = _time_grid(paths, spec, config.bin_seconds)
    metric_x, metric_mask = extract_metrics(
        _selected_path(paths, spec, condition, "METRIC"),
        condition,
        spec,
        start,
        bins,
        config,
    )
    log_x, log_mask = extract_logs(
        _selected_path(paths, spec, condition, "LOG"),
        condition,
        spec,
        start,
        bins,
        config,
    )
    trace_x, trace_mask, edges = extract_traces(
        paths,
        _selected_path(paths, spec, condition, "TRACE"),
        condition,
        spec,
        start,
        bins,
        config,
    )
    return {
        "incident_id": spec.incident_id,
        "condition": condition,
        "split": spec.split,
        "eligible": spec.eligible,
        "root_cause_service": spec.root_cause_service,
        "fault_type": spec.fault_type,
        "inject_time": spec.inject_time,
        "start_time": start,
        "bin_seconds": config.bin_seconds,
        "services": SERVICES,
        "metric_x": metric_x,
        "metric_mask": metric_mask,
        "log_x": log_x,
        "log_mask": log_mask,
        "trace_x": trace_x,
        "trace_mask": trace_mask,
        "edges": tuple(sorted(edges)),
    }


def cache_path(
    paths: Paths, spec: CaseSpec, condition: str, config: FeatureConfig
) -> Path:
    identity = json.dumps(config.to_dict(), sort_keys=True).encode("utf-8")
    version = hashlib.sha256(identity).hexdigest()[:12]
    return (
        paths.cache_root
        / version
        / spec.split
        / condition
        / Path(spec.incident_id).with_suffix(".npz")
    )


def save_case(path: Path, bundle: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        key: value
        for key, value in bundle.items()
        if isinstance(value, np.ndarray)
    }
    metadata = {
        key: value
        for key, value in bundle.items()
        if key not in arrays
    }
    metadata["services"] = list(metadata["services"])
    metadata["edges"] = [list(edge) for edge in metadata["edges"]]
    np.savez_compressed(
        path,
        **arrays,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
    )


def load_case(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
        for key in archive.files:
            if key != "metadata_json":
                metadata[key] = archive[key]
    metadata["services"] = tuple(metadata["services"])
    metadata["edges"] = tuple(tuple(edge) for edge in metadata["edges"])
    return metadata
