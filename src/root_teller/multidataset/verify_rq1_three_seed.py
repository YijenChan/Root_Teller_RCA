"""Independent structural and metric audit for the three-seed RQ1 runs."""

from __future__ import annotations

import json
from pathlib import Path

from root_teller.module1.baseline import metrics

from .rq1_three_seed import AGGREGATE_ROOT, DATASETS, PROJECT, SEEDS


EXPECTED_CASES = {"re2_ob": 90, "re2_tt": 90, "eadro_sn": 36}


def verify_run(dataset: str, seed: int) -> dict[str, object]:
    workspace = (
        PROJECT
        / "runs"
        / f"rq1_protocol_v2_seed{seed}"
        / dataset
        / "module2_workspace"
    )
    run_root = (
        workspace
        / "runs"
        / "module2_re2ob"
        / f"rq1_protocol_v2_seed{seed}"
    )
    case_root = run_root / "cases"
    summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    labels = json.loads(
        (
            workspace
            / "cache"
            / "module2_re2ob"
            / "private_evaluator"
            / "test_labels.json"
        ).read_text(encoding="utf-8")
    )
    case_files = sorted(case_root.glob("*.json"))
    if len(case_files) != EXPECTED_CASES[dataset]:
        raise ValueError(f"{dataset} seed {seed}: unexpected case count")
    if set(path.stem for path in case_files) != set(labels):
        raise ValueError(f"{dataset} seed {seed}: label/case IDs differ")
    ranks = []
    for path in case_files:
        case = json.loads(path.read_text(encoding="utf-8"))
        ranking = case["default_exhaustive"]["ranking"]
        if len(ranking) != len(set(ranking)):
            raise ValueError(f"{dataset} seed {seed}: duplicate ranking in {path.stem}")
        target = labels[path.stem]["root_cause_service"]
        if target not in ranking:
            raise ValueError(f"{dataset} seed {seed}: target absent in {path.stem}")
        ranks.append(ranking.index(target) + 1)
    recomputed = metrics(ranks)
    stored = summary["default_exhaustive_metrics"]
    for key in ("A@1", "A@2", "A@3", "A@4", "A@5", "Avg@5"):
        if abs(float(recomputed[key]) - float(stored[key])) > 1e-6:
            raise ValueError(f"{dataset} seed {seed}: metric mismatch for {key}")
    if summary["failures"]:
        raise ValueError(f"{dataset} seed {seed}: non-empty failures")
    return {
        "dataset": dataset,
        "seed": seed,
        "cases": len(case_files),
        "metrics": recomputed,
        "rank_consistent": True,
        "duplicate_free": True,
        "denominator_verified": True,
        "failures": 0,
    }


def main() -> None:
    runs = [
        verify_run(dataset, seed)
        for dataset in DATASETS
        for seed in SEEDS
    ]
    result = {
        "status": "PASS",
        "checks": [
            "case denominator",
            "private-label/case identity",
            "target present in full ranking",
            "duplicate-free rankings",
            "stored/recomputed A@1..A@5 and Avg@5",
            "zero terminal failures",
        ],
        "runs": runs,
    }
    AGGREGATE_ROOT.mkdir(parents=True, exist_ok=True)
    destination = AGGREGATE_ROOT / "verification.json"
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
