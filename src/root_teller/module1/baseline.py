from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .config import CONDITIONS, SERVICE_INDEX, FeatureConfig, Paths
from .features import cache_path, load_case, load_case_specs


def _fit_reference(cases: list[dict[str, object]]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    values: dict[str, list[np.ndarray]] = defaultdict(list)
    for case in cases:
        start = float(case["start_time"])
        inject = float(case["inject_time"])
        seconds = int(case["bin_seconds"])
        bins = case["metric_x"].shape[1]
        pre = start + np.arange(bins) * seconds < inject
        for modality in ("metric", "log", "trace"):
            x = case[f"{modality}_x"]
            mask = case[f"{modality}_mask"] & pre[None, :]
            for service in range(x.shape[0]):
                selected = x[service, mask[service]]
                if len(selected):
                    values[f"{modality}:{service}"].append(selected)
    reference: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for key, chunks in values.items():
        joined = np.concatenate(chunks, axis=0)
        median = np.nanmedian(joined, axis=0)
        q25 = np.nanpercentile(joined, 25, axis=0)
        q75 = np.nanpercentile(joined, 75, axis=0)
        scale = np.maximum((q75 - q25) / 1.349, 1e-3)
        reference[key] = (median.astype(np.float32), scale.astype(np.float32))
    return reference


def anomaly_scores(
    case: dict[str, object],
    reference: dict[str, tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    modality_scores = []
    for modality in ("metric", "log", "trace"):
        x = case[f"{modality}_x"]
        mask = case[f"{modality}_mask"]
        scores = np.zeros(x.shape[0], dtype=np.float32)
        for service in range(x.shape[0]):
            median, scale = reference[f"{modality}:{service}"]
            z = np.abs((x[service] - median) / scale)
            if np.any(mask[service]):
                per_bin = np.max(z[mask[service]], axis=1)
                scores[service] = float(np.percentile(per_bin, 95))
        modality_scores.append(scores)
    stacked = np.stack(modality_scores, axis=1)
    available = np.stack(
        [
            np.any(case[f"{modality}_mask"], axis=1)
            for modality in ("metric", "log", "trace")
        ],
        axis=1,
    )
    stacked[~available] = np.nan
    return np.nanmean(stacked, axis=1)


def metrics(ranks: list[int]) -> dict[str, float]:
    result: dict[str, float] = {}
    for k in range(1, 6):
        result[f"A@{k}"] = float(np.mean([rank <= k for rank in ranks]))
    result["Avg@5"] = float(np.mean([result[f"A@{k}"] for k in range(1, 6)]))
    return {key: round(value, 6) for key, value in result.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", choices=["validation", "test"], default="validation"
    )
    parser.add_argument(
        "--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS)
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--log-backend", choices=["hash", "sbert"], default="sbert")
    args = parser.parse_args()
    paths = Paths()
    config = FeatureConfig(log_backend=args.log_backend)
    specs = load_case_specs(paths)
    train_specs = [spec for spec in specs if spec.split == "train"]
    train_cases = [
        load_case(cache_path(paths, spec, "CLEAN", config))
        for spec in train_specs
    ]
    reference = _fit_reference(train_cases)
    results: dict[str, object] = {}
    for condition in args.conditions:
        eval_specs = [
            spec
            for spec in specs
            if spec.split == args.split
            and (args.split != "test" or spec.eligible)
        ]
        ranks: list[int] = []
        predictions = []
        for spec in eval_specs:
            case = load_case(cache_path(paths, spec, condition, config))
            scores = anomaly_scores(case, reference)
            order = np.argsort(-scores)
            target = SERVICE_INDEX[spec.root_cause_service]
            rank = int(np.where(order == target)[0][0]) + 1
            ranks.append(rank)
            predictions.append(
                {
                    "incident_id": spec.incident_id,
                    "target": spec.root_cause_service,
                    "rank": rank,
                    "top5": [case["services"][index] for index in order[:5]],
                }
            )
        results[condition] = {
            "metrics": metrics(ranks),
            "predictions": predictions,
        }
    payload = json.dumps(results, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
