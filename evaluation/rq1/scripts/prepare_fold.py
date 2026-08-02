"""Materialize one active RE2 split from the frozen nested-fold manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MANIFESTS = ROOT / "evaluation" / "rq1" / "manifests"
DATASETS = {
    "re2_ob": ("RCAEval RE2-OB", "re2_ob_nested_folds.json"),
    "re2_tt": ("RCAEval RE2-TT", "re2_tt_nested_folds.json"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument(
        "--output", type=Path, default=MANIFESTS / "active_split_manifest.csv"
    )
    args = parser.parse_args()

    system, fold_name = DATASETS[args.dataset]
    with (MANIFESTS / "case_catalog.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        catalog = {
            row["incident_id"]: row
            for row in csv.DictReader(stream)
            if row["dataset_system"] == system
        }
    folds = json.loads((MANIFESTS / fold_name).read_text(encoding="utf-8"))
    selected = [row for row in folds if int(row["outer_fold"]) == args.fold]
    if len(selected) != 90:
        raise RuntimeError(f"expected 90 rows for one RE2 fold, got {len(selected)}")

    output = []
    for assignment in selected:
        row = dict(catalog[assignment["incident_id"]])
        row["split"] = assignment["role"]
        row["eligible"] = "True"
        row["outer_fold"] = str(args.fold)
        output.append(row)
    fields = [*next(iter(catalog.values())).keys(), "outer_fold"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)
    counts = {role: sum(row["split"] == role for row in output) for role in ("train", "validation", "test")}
    print(json.dumps({"dataset": args.dataset, "fold": args.fold, "output": str(args.output), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()

