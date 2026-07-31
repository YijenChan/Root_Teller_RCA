from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import tarfile
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from root_teller.module1 import data as data_module
from root_teller.module1 import evaluate as evaluate_module
from root_teller.module1 import features as feature_module
from root_teller.module1 import model as model_module
from root_teller.module1.baseline import metrics
from root_teller.module1.config import FeatureConfig
from root_teller.module1.data import ReferenceStats, normalize_case, to_torch_case, weak_role_targets
from root_teller.module1.features import normalize_template
from root_teller.module1.model import ModelConfig, PerceptionRCA
from root_teller.module1.train import seed_everything
from root_teller.paths import workspace_root


PROJECT = workspace_root()
ROOT = PROJECT / "dataset" / "Eadro-SN" / "SN Dataset" / "SN Dataset"
DATA = ROOT / "data"
HEALTHY = ROOT / "no fault"
CACHE = PROJECT / "cache" / "rq1_multidataset_clean" / "eadro_sn"
RUN_ROOT = PROJECT / "runs" / "rq1_multidataset_clean" / "eadro_sn"
BIN_SECONDS = 60
TRACE_CLOCK_SHIFT = 8 * 60 * 60
LOG_TIME = re.compile(r"\[(\d{4}-[A-Za-z]{3}-\d{2} \d\d:\d\d:\d\d(?:\.\d+)?)\]")


def canonical(value: str) -> str:
    service = value.strip().lower()
    service = re.sub(r"^socialnetwork-", "", service)
    service = re.sub(r"-\d+$", "", service)
    aliases = {"nginx-web-server": "nginx-thrift"}
    return aliases.get(service, service)


def capture_pairs() -> list[tuple[Path, Path]]:
    pairs = []
    for annotation in sorted(DATA.glob("SN.fault-*.json")):
        stem = annotation.name.removeprefix("SN.fault-").removesuffix(".json")
        pairs.append((DATA / f"SN.{stem}", annotation))
    return pairs


def services() -> tuple[str, ...]:
    first = capture_pairs()[0][0] / "metrics"
    return tuple(sorted(canonical(path.stem) for path in first.glob("*.csv")))


def configure(candidate_services: tuple[str, ...]) -> dict[str, int]:
    index = {service: offset for offset, service in enumerate(candidate_services)}
    for module in (data_module, model_module, evaluate_module):
        module.SERVICE_INDEX = index
    return index


def _bin(timestamp: float, start: float, bins: int) -> int | None:
    value = int(math.floor((timestamp - start) / BIN_SECONDS))
    return value if 0 <= value < bins else None


def _metric_features(
    capture: Path,
    candidate_services: tuple[str, ...],
    start: float,
    end: float,
) -> tuple[np.ndarray, np.ndarray]:
    bins = max(1, int(math.ceil((end - start) / BIN_SECONDS)))
    config = FeatureConfig()
    output = np.zeros((len(candidate_services), bins, len(config.metric_groups)), np.float32)
    counts = np.zeros_like(output, dtype=np.int32)
    index = {service: offset for offset, service in enumerate(candidate_services)}
    for path in (capture / "metrics").glob("*.csv"):
        service = canonical(path.stem)
        if service not in index:
            continue
        frame = pd.read_csv(path)
        times = pd.to_numeric(frame["timestamp"], errors="coerce").to_numpy(float)
        for column in frame.columns:
            if column == "timestamp":
                continue
            group = feature_module._metric_group(column)
            group_index = config.metric_groups.index(group)
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
            values = np.log1p(np.abs(values))
            for timestamp, value in zip(times, values):
                bin_id = _bin(timestamp, start, bins)
                if bin_id is None or not np.isfinite(value):
                    continue
                output[index[service], bin_id, group_index] += float(value)
                counts[index[service], bin_id, group_index] += 1
    np.divide(output, counts, out=output, where=counts > 0)
    mask = np.any(counts > 0, axis=2)
    return output, mask


def _parse_log_time(line: str) -> float | None:
    match = LOG_TIME.search(line)
    if not match:
        return None
    try:
        # The released strings are local Asia/Shanghai wall time. On the
        # experiment host datetime.timestamp converts them to the epoch used
        # by the metric and fault annotation streams.
        return datetime.strptime(match.group(1), "%Y-%b-%d %H:%M:%S.%f").timestamp()
    except ValueError:
        return None


def _log_features(
    capture: Path,
    candidate_services: tuple[str, ...],
    start: float,
    end: float,
) -> tuple[np.ndarray, np.ndarray]:
    bins = max(1, int(math.ceil((end - start) / BIN_SECONDS)))
    config = FeatureConfig(log_backend="sbert")
    dimension = config.log_content_dim + config.log_extra_dim
    output = np.zeros((len(candidate_services), bins, dimension), np.float32)
    counts = np.zeros((len(candidate_services), bins), np.float32)
    index = {service: offset for offset, service in enumerate(candidate_services)}
    payload = json.loads((capture / "logs.json").read_text(encoding="utf-8"))
    records = []
    for raw_service, lines in payload.items():
        service = canonical(raw_service)
        if service not in index:
            continue
        for line in lines:
            timestamp = _parse_log_time(str(line))
            if timestamp is None:
                continue
            bin_id = _bin(timestamp, start, bins)
            if bin_id is None:
                continue
            template = normalize_template(line)
            records.append((index[service], bin_id, template, "error" in line.lower()))
    embeddings = feature_module._sbert_embeddings(
        sorted({item[2] for item in records}), config
    ) if records else {}
    for service_index, bin_id, template, is_error in records:
        output[service_index, bin_id, : config.log_content_dim] += embeddings[template]
        counts[service_index, bin_id] += 1
        output[service_index, bin_id, config.log_content_dim] += 1
        output[service_index, bin_id, config.log_content_dim + 1] += float(is_error)
    np.divide(
        output[:, :, : config.log_content_dim],
        counts[:, :, None],
        out=output[:, :, : config.log_content_dim],
        where=counts[:, :, None] > 0,
    )
    output[:, :, config.log_content_dim :] = np.log1p(
        output[:, :, config.log_content_dim :]
    )
    # An empty log bin is an observed zero-count bin, not missing telemetry.
    return output, np.ones((len(candidate_services), bins), dtype=bool)


def _trace_features(
    capture: Path,
    candidate_services: tuple[str, ...],
    start: float,
    end: float,
) -> tuple[np.ndarray, np.ndarray, tuple[tuple[str, str], ...]]:
    bins = max(1, int(math.ceil((end - start) / BIN_SECONDS)))
    output = np.zeros((len(candidate_services), bins, 6), np.float32)
    aggregate: dict[tuple[int, int], list[float]] = defaultdict(
        lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    index = {service: offset for offset, service in enumerate(candidate_services)}
    edges = set()
    traces = json.loads((capture / "spans.json").read_text(encoding="utf-8"))
    for trace in traces:
        processes = {
            key: canonical(value["serviceName"])
            for key, value in trace.get("processes", {}).items()
        }
        span_service = {
            span["spanID"]: processes.get(span.get("processID"), "")
            for span in trace.get("spans", [])
        }
        for span in trace.get("spans", []):
            service = processes.get(span.get("processID"), "")
            if service not in index:
                continue
            timestamp = float(span.get("startTime", 0)) / 1e6 - TRACE_CLOCK_SHIFT
            bin_id = _bin(timestamp, start, bins)
            if bin_id is not None:
                tags = {tag["key"]: tag.get("value") for tag in span.get("tags", [])}
                error = bool(tags.get("error")) or int(tags.get("http.status_code", 0) or 0) >= 500
                duration = max(float(span.get("duration", 0)), 0.0)
                values = aggregate[(index[service], bin_id)]
                values[0] += 1
                values[1] += float(error)
                values[2] += duration
                values[3] = max(values[3], duration)
                values[4] += len(str(span.get("operationName", "")))
                values[5] += float(bool(span.get("logs")))
            for reference in span.get("references", []):
                parent = span_service.get(reference.get("spanID"), "")
                if parent in index and parent != service:
                    edges.add((parent, service))
    for (service_index, bin_id), values in aggregate.items():
        count = max(values[0], 1.0)
        output[service_index, bin_id] = np.asarray(
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
    return output, np.ones((len(candidate_services), bins), dtype=bool), tuple(sorted(edges))


def build_case(
    capture: Path,
    fault: dict[str, object],
    end: float,
    capture_index: int,
    fault_index: int,
    candidate_services: tuple[str, ...],
) -> dict[str, object]:
    start = float(fault["start"])
    metric_x, metric_mask = _metric_features(capture, candidate_services, start, end)
    log_x, log_mask = _log_features(capture, candidate_services, start, end)
    trace_x, trace_mask, edges = _trace_features(
        capture, candidate_services, start, end
    )
    return {
        "incident_id": f"sn-c{capture_index:02d}-f{fault_index:02d}",
        "capture_id": capture.name,
        "condition": "CLEAN",
        "split": "loco",
        "eligible": True,
        "root_cause_service": canonical(str(fault["name"])),
        "fault_type": str(fault["fault"]),
        "inject_time": start,
        "start_time": start,
        "bin_seconds": BIN_SECONDS,
        "services": candidate_services,
        "metric_x": metric_x,
        "metric_mask": metric_mask,
        "log_x": log_x,
        "log_mask": log_mask,
        "trace_x": trace_x,
        "trace_mask": trace_mask,
        "edges": edges,
    }


def case_cache_path(capture_index: int, fault_index: int) -> Path:
    return CACHE / "cases" / f"c{capture_index:02d}_f{fault_index:02d}.pt"


def build_cases(overwrite: bool = False) -> list[dict[str, object]]:
    candidate_services = services()
    cases = []
    for capture_index, (capture, annotation) in enumerate(capture_pairs()):
        metadata = json.loads(annotation.read_text(encoding="utf-8"))
        faults = metadata["faults"]
        for fault_index, fault in enumerate(faults):
            destination = case_cache_path(capture_index, fault_index)
            if destination.exists() and not overwrite:
                case = torch.load(destination, weights_only=False)
            else:
                end = (
                    float(faults[fault_index + 1]["start"])
                    if fault_index + 1 < len(faults)
                    else float(metadata["end"])
                )
                case = build_case(
                    capture,
                    fault,
                    end,
                    capture_index,
                    fault_index,
                    candidate_services,
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                torch.save(case, destination)
            cases.append(case)
            print(json.dumps({"case": case["incident_id"], "bins": case["metric_x"].shape[1]}), flush=True)
    return cases


def _extract_healthy() -> list[Path]:
    destination = CACHE / "healthy_raw"
    destination.mkdir(parents=True, exist_ok=True)
    captures = []
    for archive in sorted(HEALTHY.glob("SN.*.tar.xz")):
        expected = destination / archive.name.removesuffix(".tar.xz")
        if not expected.exists():
            with tarfile.open(archive, "r:xz") as handle:
                handle.extractall(destination, filter="data")
        captures.append(expected)
    return captures


def fit_healthy_reference(candidate_services: tuple[str, ...]) -> ReferenceStats:
    cache_path = CACHE / "healthy_reference.pt"
    if cache_path.exists():
        return ReferenceStats.from_state_dict(torch.load(cache_path, weights_only=False))
    pseudo_cases = []
    for capture in _extract_healthy():
        metric_files = list((capture / "metrics").glob("*.csv"))
        timestamps = pd.to_numeric(pd.read_csv(metric_files[0], usecols=["timestamp"])["timestamp"])
        start, end = float(timestamps.min()), float(timestamps.max()) + 1
        metric_x, metric_mask = _metric_features(capture, candidate_services, start, end)
        log_x, log_mask = _log_features(capture, candidate_services, start, end)
        trace_x, trace_mask, edges = _trace_features(capture, candidate_services, start, end)
        pseudo_cases.append(
            {
                "start_time": start,
                "inject_time": end + 1,
                "bin_seconds": BIN_SECONDS,
                "services": candidate_services,
                "metric_x": metric_x,
                "metric_mask": metric_mask,
                "log_x": log_x,
                "log_mask": log_mask,
                "trace_x": trace_x,
                "trace_mask": trace_mask,
                "edges": edges,
            }
        )
    reference = data_module.fit_reference(pseudo_cases)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(reference.state_dict(), cache_path)
    return reference


@torch.no_grad()
def evaluate(
    model: PerceptionRCA,
    cases: list[dict[str, object]],
    reference: ReferenceStats,
    device: torch.device,
) -> list[dict[str, object]]:
    model.eval()
    predictions = []
    for case in cases:
        tensor = to_torch_case(case, reference, device)
        output = model(tensor)
        order = torch.argsort(output["localization_probabilities"], descending=True).cpu().tolist()
        rank = order.index(int(tensor["target_index"])) + 1
        predictions.append(
            {
                "incident_id": case["incident_id"],
                "capture_id": case["capture_id"],
                "target": case["root_cause_service"],
                "rank": rank,
                "top5": [case["services"][index] for index in order[:5]],
            }
        )
    return predictions


def train_loco(
    cases: list[dict[str, object]],
    epochs: int = 20,
    base_seed: int = 20260724,
    run_root: Path | None = None,
) -> dict[str, object]:
    output_root = run_root or RUN_ROOT
    candidate_services = services()
    configure(candidate_services)
    reference = fit_healthy_reference(candidate_services)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_config = ModelConfig()
    all_predictions = []
    fold_summaries = []
    for fold in range(4):
        train_cases = [case for case in cases if case["incident_id"][:6] != f"sn-c{fold:02d}"]
        test_cases = [case for case in cases if case["incident_id"][:6] == f"sn-c{fold:02d}"]
        seed = base_seed + fold
        seed_everything(seed)
        first = train_cases[0]
        model = PerceptionRCA(
            first["metric_x"].shape[2],
            first["log_x"].shape[2],
            first["trace_x"].shape[2],
            model_config,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        tensor_cases = [to_torch_case(case, reference, device) for case in train_cases]
        targets = [
            torch.as_tensor(
                weak_role_targets(normalize_case(case, reference), reference),
                dtype=torch.long,
                device=device,
            )
            for case in train_cases
        ]
        counts = np.zeros(3)
        for target in targets:
            for role in range(3):
                counts[role] += int((target == role).sum().item())
        inverse = 1.0 / np.maximum(counts, 1)
        weights = torch.as_tensor(inverse / inverse.mean(), dtype=torch.float32, device=device)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + 17)
        for epoch in range(epochs):
            model.train()
            for index in torch.randperm(len(tensor_cases), generator=generator).tolist():
                optimizer.zero_grad(set_to_none=True)
                output = model(tensor_cases[index])
                role_loss = F.cross_entropy(
                    output["role_logits"], targets[index], weight=weights, ignore_index=-100
                )
                target = torch.as_tensor(
                    [int(tensor_cases[index]["target_index"])],
                    dtype=torch.long,
                    device=device,
                )
                loss = role_loss + F.cross_entropy(
                    output["localization_logits"].unsqueeze(0), target
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        predictions = evaluate(model, test_cases, reference, device)
        fold_metrics = metrics([int(item["rank"]) for item in predictions])
        fold_summaries.append({"fold": fold, "test_capture": test_cases[0]["capture_id"], "metrics": fold_metrics})
        all_predictions.extend(predictions)
        fold_dir = output_root / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "reference": reference.state_dict(),
                "model_config": asdict(model_config),
                "services": list(candidate_services),
                "epochs": epochs,
                "seed": seed,
            },
            fold_dir / "checkpoint.pt",
        )
        print(json.dumps(fold_summaries[-1]), flush=True)
    overall = metrics([int(item["rank"]) for item in all_predictions])
    result = {
        "dataset": "Eadro-SN",
        "protocol": "grouped-4-fold-leave-one-capture-out",
        "condition": "clean",
        "window_seconds": BIN_SECONDS,
        "case_interval": "fault_start_to_next_fault_start_or_capture_end",
        "healthy_reference": "three released no-fault captures",
        "epochs": epochs,
        "base_seed": base_seed,
        "cases": len(all_predictions),
        "candidate_services": len(candidate_services),
        "metrics": overall,
        "folds": fold_summaries,
        "predictions": sorted(all_predictions, key=lambda item: item["incident_id"]),
        "label_leakage": False,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "module1_loco_results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["build", "train", "all"], default="all")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    configure(services())
    started = time.time()
    cases = build_cases(args.overwrite)
    if args.stage in {"train", "all"}:
        result = train_loco(cases, args.epochs)
        print(json.dumps(result | {"predictions": "saved"}, indent=2), flush=True)
    print(json.dumps({"elapsed_seconds": round(time.time() - started, 3)}), flush=True)


if __name__ == "__main__":
    main()
