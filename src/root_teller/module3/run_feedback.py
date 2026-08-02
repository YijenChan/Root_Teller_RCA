from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from .config import Module3Config, Module3Paths
from .feedback import (
    false_feedback_run,
    final_steward_payload,
    truthful_feedback_run,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _repetition_stats(trace: list[dict[str, Any]]) -> tuple[int, int]:
    inspected = [
        item["feedback_event"]["entity_id"]
        for item in trace
        if item["feedback_event"]["verdict"] == "REJECT"
    ]
    consecutive = sum(
        left == right for left, right in zip(inspected[:-1], inspected[1:])
    )
    counts = Counter(inspected)
    reactivated = sum(count - 1 for count in counts.values() if count > 1)
    return consecutive, reactivated


def run(run_id: str) -> dict[str, Any]:
    config = Module3Config()
    paths = Module3Paths()
    output_root = paths.run_root / run_id
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"run directory is not empty: {output_root}")
    truthful_root = output_root / "truthful_feedback_cases"
    false_root = output_root / "false_feedback_cases"
    truthful_root.mkdir(parents=True, exist_ok=True)
    false_root.mkdir(parents=True, exist_ok=True)

    labels = {
        item["incident_id"]: item["target"]
        for item in json.loads(paths.private_labels.read_text(encoding="utf-8"))
    }
    case_files = sorted((paths.module2_run / "cases").glob("*.json"))
    truthful_rows = []
    truthful_results = []
    for case_file in case_files:
        case = json.loads(case_file.read_text(encoding="utf-8"))
        steward = final_steward_payload(case)
        incident_id = case["incident_id"]
        target = labels[incident_id]
        result = truthful_feedback_run(
            incident_id=incident_id,
            target=target,
            base_scores=steward["fused_scores"],
            decay=config.feedback_score_decay,
            budget=config.feedback_budget,
        )
        truthful_results.append(result)
        consecutive, reactivated = _repetition_stats(result["trace"])
        truthful_rows.append(
            {
                "incident_id": incident_id,
                "target_private": target,
                "initial_top1": result["initial_top1"],
                "initial_correct": result["initial_correct"],
                "selected_for_feedback": not result["initial_correct"],
                "success": result["success"],
                "rejected_rounds_to_correct": (
                    result["rejected_rounds_to_correct"]
                    if result["rejected_rounds_to_correct"] is not None
                    else config.feedback_budget + 1
                ),
                "final_top1": result["final_top1"],
                "consecutive_candidate_repetitions": consecutive,
                "rejected_candidate_reactivations": reactivated,
            }
        )
        public_result = dict(result)
        public_result.pop("target", None)
        (truthful_root / f"{incident_id}.json").write_text(
            json.dumps(public_result, indent=2) + "\n", encoding="utf-8"
        )

    selected = [row for row in truthful_rows if row["selected_for_feedback"]]
    successful = [row for row in selected if row["success"]]
    round_values = [int(row["rejected_rounds_to_correct"]) for row in selected]
    accuracy_by_round = []
    for round_index in range(config.feedback_budget + 1):
        correct = sum(
            bool(row["initial_correct"])
            or (
                bool(row["success"])
                and int(row["rejected_rounds_to_correct"]) <= round_index
            )
            for row in truthful_rows
        )
        accuracy_by_round.append(
            {
                "feedback_round": round_index,
                "top1_accuracy": correct / len(truthful_rows),
                "correct_cases": correct,
                "total_cases": len(truthful_rows),
            }
        )

    false_rows = []
    curve_rows = []
    for false_budget in config.false_rejection_budgets:
        budget_root = false_root / f"false_{false_budget:02d}"
        budget_root.mkdir(parents=True, exist_ok=True)
        budget_results = []
        for case_file in case_files:
            case = json.loads(case_file.read_text(encoding="utf-8"))
            steward = final_steward_payload(case)
            incident_id = case["incident_id"]
            target = labels[incident_id]
            result = false_feedback_run(
                incident_id=incident_id,
                target=target,
                base_scores=steward["fused_scores"],
                decay=config.feedback_score_decay,
                false_rejection_budget=false_budget,
                total_budget=config.feedback_budget,
            )
            budget_results.append(result)
            consecutive, reactivated = _repetition_stats(result["trace"])
            false_rows.append(
                {
                    "incident_id": incident_id,
                    "target_private": target,
                    "false_rejection_budget": false_budget,
                    "false_rejections_applied": result["false_rejections_applied"],
                    "success": result["success"],
                    "feedback_rounds_to_accept": (
                        result["feedback_rounds_to_accept"]
                        if result["feedback_rounds_to_accept"] is not None
                        else ""
                    ),
                    "capped_feedback_rounds": result["capped_feedback_rounds"],
                    "final_top1": result["final_top1"],
                    "consecutive_candidate_repetitions": consecutive,
                    "rejected_candidate_reactivations": reactivated,
                }
            )
            public_result = dict(result)
            public_result.pop("target", None)
            (budget_root / f"{incident_id}.json").write_text(
                json.dumps(public_result, indent=2) + "\n", encoding="utf-8"
            )
        capped = [int(item["capped_feedback_rounds"]) for item in budget_results]
        successes = [item for item in budget_results if item["success"]]
        uncapped = [
            int(item["feedback_rounds_to_accept"])
            for item in successes
            if item["feedback_rounds_to_accept"] is not None
        ]
        curve_rows.append(
            {
                "false_rejection_budget": false_budget,
                "mean_capped_feedback_rounds": round(statistics.mean(capped), 6),
                "median_capped_feedback_rounds": round(statistics.median(capped), 6),
                "success_rate_within_20": round(len(successes) / len(budget_results), 6),
                "mean_rounds_success_only": (
                    round(statistics.mean(uncapped), 6) if uncapped else ""
                ),
                "mean_false_rejections_applied": round(
                    statistics.mean(
                        int(item["false_rejections_applied"])
                        for item in budget_results
                    ),
                    6,
                ),
            }
        )

    _write_csv(output_root / "truthful_feedback_raw.csv", truthful_rows)
    _write_csv(output_root / "accuracy_by_feedback_round.csv", accuracy_by_round)
    _write_csv(output_root / "false_feedback_raw.csv", false_rows)
    _write_csv(output_root / "false_feedback_curve.csv", curve_rows)
    summary = {
        "schema_version": "module3-feedback-summary-1.0",
        "run_id": run_id,
        "config": config.to_dict(),
        "cases": len(truthful_rows),
        "initial_top1_correct": sum(row["initial_correct"] for row in truthful_rows),
        "initial_top1_incorrect": len(selected),
        "truthful_feedback": {
            "selected_cases": len(selected),
            "successes_within_budget": len(successful),
            "success_rate": len(successful) / len(selected) if selected else 1.0,
            "mean_rejected_rounds": statistics.mean(round_values) if round_values else 0,
            "median_rejected_rounds": statistics.median(round_values) if round_values else 0,
            "max_rejected_rounds": max(round_values) if round_values else 0,
            "consecutive_candidate_repetitions": sum(
                int(row["consecutive_candidate_repetitions"]) for row in selected
            ),
            "rejected_candidate_reactivations": sum(
                int(row["rejected_candidate_reactivations"]) for row in selected
            ),
        },
        "false_feedback_curve": curve_rows,
        "ground_truth_exposed_to_steward": False,
        "immutable_structural_evidence": True,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    args = _args()
    run(args.run_id)


if __name__ == "__main__":
    main()
