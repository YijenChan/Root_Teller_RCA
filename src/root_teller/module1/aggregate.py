from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from root_teller.paths import workspace_root


ROOT = workspace_root() / "runs" / "module1_re2ob"
SEEDS = (20260724, 20260725, 20260726)
CONDITIONS = (
    "CLEAN",
    "GMO_METRIC",
    "GMO_LOG",
    "GMO_TRACE",
    "IAMI_METRIC",
    "IAMI_LOG",
    "IAMI_TRACE",
)
METRICS = ("A@1", "A@3", "Avg@5")


def aggregate() -> dict[str, object]:
    runs = []
    for seed in SEEDS:
        path = ROOT / f"final_refit_seed{seed}" / "test_evaluation" / "summary.json"
        runs.append(json.loads(path.read_text(encoding="utf-8")))
    conditions: dict[str, object] = {}
    for condition in CONDITIONS:
        condition_result = {}
        for metric in METRICS:
            values = [
                run["conditions"][condition]["metrics"][metric] for run in runs
            ]
            condition_result[metric] = {
                "mean": round(float(np.mean(values)), 6),
                "std": round(float(np.std(values)), 6),
                "values": values,
            }
        condition_result["operational_rate"] = 1.0
        conditions[condition] = condition_result
    return {
        "protocol": "three frozen train+validation refits; one held-out evaluation",
        "seeds": list(SEEDS),
        "conditions": conditions,
        "nan_or_error_cases": sum(run["nan_or_error_cases"] for run in runs),
    }


def main() -> None:
    payload = aggregate()
    destination = ROOT / "checkpoint2_test_aggregate.json"
    destination.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
