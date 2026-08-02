from __future__ import annotations

import argparse
import json
import time

from .config import CONDITIONS, FeatureConfig, Paths
from .features import cache_path, extract_case, load_case_specs, save_case


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Module 1 feature cache.")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "validation", "test"],
        required=True,
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=CONDITIONS,
        required=True,
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-backend", choices=["hash", "sbert"], default="hash")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = Paths()
    config = FeatureConfig(log_backend=args.log_backend)
    specs = [
        spec
        for spec in load_case_specs(paths)
        if spec.split in args.splits
        and (spec.split != "test" or spec.eligible)
    ]
    completed = 0
    failures: list[dict[str, str]] = []
    started = time.time()
    for spec in specs:
        for condition in args.conditions:
            if spec.split == "train" and condition != "CLEAN":
                continue
            destination = cache_path(paths, spec, condition, config)
            if destination.exists() and not args.overwrite:
                print(f"SKIP {spec.split} {condition} {spec.incident_id}", flush=True)
                completed += 1
                continue
            try:
                print(f"BUILD {spec.split} {condition} {spec.incident_id}", flush=True)
                bundle = extract_case(paths, spec, condition, config)
                save_case(destination, bundle)
                completed += 1
            except Exception as error:
                failures.append(
                    {
                        "incident_id": spec.incident_id,
                        "condition": condition,
                        "error": repr(error),
                    }
                )
                print(f"FAIL {condition} {spec.incident_id}: {error!r}", flush=True)
    summary = {
        "completed": completed,
        "failures": failures,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    print(json.dumps(summary, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
