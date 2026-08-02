"""Aggregate full-protocol verification summaries without inventing results."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in args.summaries:
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload.get("metrics", payload.get("default_exhaustive_metrics", payload))
        rows.append({
            "dataset": payload["dataset"],
            "seed": int(payload["seed"]),
            "cases": int(payload["cases"]),
            "A@1": float(metrics["A@1"]),
            "A@3": float(metrics["A@3"]),
            "Avg@5": float(metrics["Avg@5"]),
        })
    grouped = {}
    for row in rows:
        grouped.setdefault(row["dataset"], []).append(row)
    exported = []
    for dataset, items in sorted(grouped.items()):
        if {item["seed"] for item in items} != {41, 42, 43}:
            raise RuntimeError(f"{dataset}: expected seeds 41, 42, and 43")
        if len({item["cases"] for item in items}) != 1:
            raise RuntimeError(f"{dataset}: inconsistent case denominators")
        for metric in ("A@1", "A@3", "Avg@5"):
            values = [item[metric] for item in items]
            exported.append({
                "dataset": dataset,
                "cases": items[0]["cases"],
                "metric": metric,
                "mean": statistics.mean(values),
                "sample_std": statistics.stdev(values),
                "seed_41": next(item[metric] for item in items if item["seed"] == 41),
                "seed_42": next(item[metric] for item in items if item["seed"] == 42),
                "seed_43": next(item[metric] for item in items if item["seed"] == 43),
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=exported[0].keys())
        writer.writeheader()
        writer.writerows(exported)


if __name__ == "__main__":
    main()

