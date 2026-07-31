"""Paper-aligned three-seed orchestration for Root-Teller RQ1.

Each run delegates training, fold selection, refitting, Evidence Pack export,
collaborative diagnosis, and private evaluation to ``rq1_protocol_v2``. This
keeps every dataset/seed workspace isolated while using the same maximum-300-
epoch, five-epoch validation, patience-50 protocol described in the paper.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from root_teller.paths import workspace_root


PROJECT = workspace_root()
SEEDS = (41, 42, 43)
DATASETS = ("re2_ob", "re2_tt", "eadro_sn")
AGGREGATE_ROOT = PROJECT / "runs" / "rq1_root_teller_three_seed"


def run_protocol(
    *, dataset: str, seed: int, stage: str, workers: int, resume: bool
) -> None:
    command = [
        sys.executable,
        "-m",
        "root_teller.multidataset.rq1_protocol_v2",
        "--dataset",
        dataset,
        "--seed",
        str(seed),
        "--stage",
        stage,
        "--workers",
        str(workers),
    ]
    if resume:
        command.append("--resume")
    subprocess.run(command, cwd=PROJECT, check=True)


def result_path(dataset: str, seed: int) -> Path:
    return (
        PROJECT
        / "runs"
        / f"rq1_protocol_v2_seed{seed}"
        / dataset
        / "verified_rq1_result.json"
    )


def aggregate() -> dict[str, object]:
    payload: dict[str, object] = {
        "protocol": (
            "three independent seeds; grouped outer folds; complete telemetry; "
            "full-range access; no SRE feedback"
        ),
        "seeds": list(SEEDS),
        "datasets": {},
    }
    for dataset in DATASETS:
        runs = [
            json.loads(result_path(dataset, seed).read_text(encoding="utf-8"))
            for seed in SEEDS
        ]
        metrics = {}
        for metric in ("A@1", "A@3", "Avg@5"):
            values = [float(run[metric]) for run in runs]
            metrics[metric] = {
                "mean": round(float(np.mean(values)), 6),
                "sample_std": round(float(np.std(values, ddof=1)), 6),
                "values": values,
            }
        payload["datasets"][dataset] = {"runs": runs, "aggregate": metrics}
    AGGREGATE_ROOT.mkdir(parents=True, exist_ok=True)
    destination = AGGREGATE_ROOT / "aggregate.json"
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("train", "export", "module2", "verify", "all", "aggregate"),
        required=True,
    )
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.stage == "aggregate":
        aggregate()
        return
    if args.dataset is None or args.seed is None:
        parser.error("--dataset and --seed are required unless --stage aggregate is used")
    run_protocol(
        dataset=args.dataset,
        seed=args.seed,
        stage=args.stage,
        workers=args.workers,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
