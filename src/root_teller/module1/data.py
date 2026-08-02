from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

from .config import SERVICE_INDEX


MODALITIES = ("metric", "log", "trace")


@dataclass
class ReferenceStats:
    centers: dict[str, np.ndarray]
    scales: dict[str, np.ndarray]
    thresholds: dict[str, float]

    def state_dict(self) -> dict[str, object]:
        return {
            "centers": self.centers,
            "scales": self.scales,
            "thresholds": self.thresholds,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "ReferenceStats":
        return cls(
            centers=state["centers"],
            scales=state["scales"],
            thresholds=state["thresholds"],
        )


def fit_reference(cases: Iterable[dict[str, object]]) -> ReferenceStats:
    cases = list(cases)
    values: dict[str, list[list[np.ndarray]]] = {
        modality: [[] for _ in range(len(cases[0]["services"]))]
        for modality in MODALITIES
    }
    for case in cases:
        start = float(case["start_time"])
        inject = float(case["inject_time"])
        seconds = int(case["bin_seconds"])
        bins = case["metric_x"].shape[1]
        pre_injection = start + np.arange(bins) * seconds < inject
        for modality in MODALITIES:
            matrix = case[f"{modality}_x"]
            mask = case[f"{modality}_mask"] & pre_injection[None, :]
            for service_index in range(matrix.shape[0]):
                selected = matrix[service_index, mask[service_index]]
                if len(selected):
                    values[modality][service_index].append(selected)

    centers: dict[str, np.ndarray] = {}
    scales: dict[str, np.ndarray] = {}
    for modality in MODALITIES:
        service_centers = []
        service_scales = []
        pooled = np.concatenate(
            [
                chunk
                for service_chunks in values[modality]
                for chunk in service_chunks
            ],
            axis=0,
        )
        pooled_scale = np.nanstd(pooled, axis=0)
        positive = pooled_scale[pooled_scale > 1e-6]
        generic_floor = float(np.percentile(positive, 10)) if len(positive) else 0.05
        feature_floor = np.maximum(pooled_scale * 0.05, generic_floor * 0.05)
        for chunks in values[modality]:
            joined = np.concatenate(chunks, axis=0)
            center = np.nanmean(joined, axis=0)
            scale = np.nanstd(joined, axis=0)
            scale = np.maximum(scale, feature_floor)
            scale = np.maximum(scale, 1e-3)
            service_centers.append(center.astype(np.float32))
            service_scales.append(scale.astype(np.float32))
        centers[modality] = np.stack(service_centers)
        scales[modality] = np.stack(service_scales)

    deviations: dict[str, list[np.ndarray]] = {modality: [] for modality in MODALITIES}
    for case in cases:
        start = float(case["start_time"])
        inject = float(case["inject_time"])
        seconds = int(case["bin_seconds"])
        bins = case["metric_x"].shape[1]
        pre_injection = start + np.arange(bins) * seconds < inject
        for modality in MODALITIES:
            normalized = (
                case[f"{modality}_x"] - centers[modality][:, None, :]
            ) / scales[modality][:, None, :]
            mask = case[f"{modality}_mask"] & pre_injection[None, :]
            deviation = np.max(np.abs(normalized), axis=2)
            deviations[modality].append(deviation[mask])
    thresholds = {
        modality: float(np.percentile(np.concatenate(chunks), 99))
        for modality, chunks in deviations.items()
    }
    return ReferenceStats(centers, scales, thresholds)


def normalize_case(
    case: dict[str, object], reference: ReferenceStats
) -> dict[str, object]:
    normalized = dict(case)
    for modality in MODALITIES:
        values = (
            case[f"{modality}_x"] - reference.centers[modality][:, None, :]
        ) / reference.scales[modality][:, None, :]
        values = np.clip(values, -20.0, 20.0).astype(np.float32)
        mask = case[f"{modality}_mask"].astype(bool)
        values[~mask] = 0.0
        normalized[f"{modality}_x"] = values
        normalized[f"{modality}_mask"] = mask
    return normalized


def modality_quality(mask: torch.Tensor) -> torch.Tensor:
    """Equation 7 quality factor for an [N,T] boolean mask."""
    mask = mask.bool()
    nodes, steps = mask.shape
    available_fraction = mask.float().mean(dim=1)
    qualities = []
    for node in range(nodes):
        valid = torch.nonzero(mask[node], as_tuple=False).flatten()
        if len(valid) == 0:
            qualities.append(mask.new_tensor(0.0, dtype=torch.float32))
            continue
        staleness = (steps - 1 - valid[-1].float()) / max(steps, 1)
        longest_missing = 0
        current = 0
        for value in mask[node].tolist():
            if value:
                current = 0
            else:
                current += 1
                longest_missing = max(longest_missing, current)
        continuity = 1.0 - longest_missing / max(steps, 1)
        quality = (
            available_fraction[node] + (1.0 - staleness) + continuity
        ) / 3.0
        qualities.append(quality)
    return torch.stack(qualities)


def anomaly_state(
    case: dict[str, object], reference: ReferenceStats
) -> tuple[np.ndarray, np.ndarray]:
    nodes = len(case["services"])
    bins = case["metric_x"].shape[1]
    onset = np.full(nodes, -1, dtype=np.int64)
    anomalous = np.zeros(nodes, dtype=bool)
    for modality in MODALITIES:
        values = case[f"{modality}_x"]
        mask = case[f"{modality}_mask"]
        deviation = np.max(np.abs(values), axis=2)
        crossing = (deviation > reference.thresholds[modality]) & mask
        for node in range(nodes):
            indices = np.flatnonzero(crossing[node])
            if len(indices):
                anomalous[node] = True
                first = int(indices[0])
                onset[node] = first if onset[node] < 0 else min(onset[node], first)
    return anomalous, onset


def weak_role_targets(
    case: dict[str, object],
    reference: ReferenceStats,
) -> np.ndarray:
    """Build root/propagation/normal/ignore targets from clean training data."""
    anomalous, onset = anomaly_state(case, reference)
    nodes = len(case["services"])
    root = SERVICE_INDEX[str(case["root_cause_service"])]
    adjacency: dict[int, list[int]] = {index: [] for index in range(nodes)}
    for source, target in case["edges"]:
        if source in SERVICE_INDEX and target in SERVICE_INDEX:
            adjacency[SERVICE_INDEX[source]].append(SERVICE_INDEX[target])
    reachable = {root}
    frontier = [root]
    while frontier:
        source = frontier.pop()
        for target in adjacency[source]:
            consistent = (
                onset[source] < 0
                or onset[target] < 0
                or onset[source] <= onset[target] + 1
            )
            if consistent and target not in reachable:
                reachable.add(target)
                frontier.append(target)
    targets = np.full(nodes, -100, dtype=np.int64)
    full_coverage = np.ones(nodes, dtype=bool)
    for modality in MODALITIES:
        full_coverage &= np.any(case[f"{modality}_mask"], axis=1)
    targets[full_coverage & ~anomalous] = 2
    for node in reachable:
        if node != root and anomalous[node]:
            targets[node] = 1
    targets[root] = 0
    return targets


def to_torch_case(
    case: dict[str, object],
    reference: ReferenceStats,
    device: torch.device,
) -> dict[str, object]:
    normalized = normalize_case(case, reference)
    result = dict(normalized)
    for modality in MODALITIES:
        result[f"{modality}_x"] = torch.as_tensor(
            normalized[f"{modality}_x"], dtype=torch.float32, device=device
        )
        result[f"{modality}_mask"] = torch.as_tensor(
            normalized[f"{modality}_mask"], dtype=torch.bool, device=device
        )
    _, onset = anomaly_state(normalized, reference)
    result["onset"] = torch.as_tensor(onset, dtype=torch.long, device=device)
    result["target_index"] = SERVICE_INDEX[str(case["root_cause_service"])]
    return result


def apply_structured_dropout(
    case: dict[str, object],
    generator: torch.Generator,
    full_probability: float,
    suffix_probability: float,
) -> dict[str, object]:
    result = dict(case)
    draw = torch.rand((), generator=generator).item()
    if draw >= full_probability + suffix_probability:
        return result
    modality_index = int(
        torch.randint(0, len(MODALITIES), (), generator=generator).item()
    )
    modality = MODALITIES[modality_index]
    mask = case[f"{modality}_mask"].clone()
    values = case[f"{modality}_x"].clone()
    if draw < full_probability:
        mask[:] = False
    else:
        steps = mask.shape[1]
        cut = int(
            torch.randint(max(1, steps // 3), max(2, 2 * steps // 3), (), generator=generator).item()
        )
        mask[:, cut:] = False
    values[~mask] = 0.0
    result[f"{modality}_mask"] = mask
    result[f"{modality}_x"] = values
    return result

