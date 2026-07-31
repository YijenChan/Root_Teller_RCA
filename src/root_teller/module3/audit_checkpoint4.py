from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from root_teller.paths import workspace_root

from .reporting import build_ledger, deterministic_discrepancies


PROJECT = workspace_root()
MODULE2 = PROJECT / "runs/module2_re2ob/checkpoint3_v2_1_default_clean_replay"
FEEDBACK = PROJECT / "runs/module3_re2ob/checkpoint4_feedback_clean_v1"
REPORTS = PROJECT / "runs/module3_re2ob/checkpoint4_reports_clean_v3"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit() -> dict[str, Any]:
    with (FEEDBACK / "truthful_feedback_raw.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        truthful_rows = list(csv.DictReader(stream))
    selected = [
        row for row in truthful_rows if row["selected_for_feedback"] == "True"
    ]
    successful_selected = [
        row for row in selected if row["success"] == "True"
    ]

    event_count = 0
    structural_mutations = 0
    feedback_target_leaks = 0
    for path in (FEEDBACK / "truthful_feedback_cases").glob("*.json"):
        payload = _json(path)
        for event in payload["overlay"]["feedback_events"]:
            event_count += 1
            structural_mutations += bool(event["structural_evidence_mutated"])
            feedback_target_leaks += any(
                key in event for key in ("target", "target_private", "ground_truth")
            )

    false_case_files = list(
        (FEEDBACK / "false_feedback_cases").glob("false_*/*.json")
    )
    false_structural_mutations = 0
    for path in false_case_files:
        payload = _json(path)
        for event in payload["overlay"]["feedback_events"]:
            false_structural_mutations += bool(
                event["structural_evidence_mutated"]
            )
    with (FEEDBACK / "false_feedback_raw.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        false_rows = list(csv.DictReader(stream))
    rejected_reactivations = sum(
        int(row["rejected_candidate_reactivations"]) for row in false_rows
    )

    invalid_reports = []
    status_counts: dict[str, int] = {}
    canonical_repair_count = 0
    advisory_verdicts: dict[str, int] = {}
    for report_path in sorted((REPORTS / "reports_json").glob("*.json")):
        result = _json(report_path)
        case = _json(MODULE2 / "cases" / report_path.name)
        rmg = _json(MODULE2 / "rmg" / report_path.name)
        ledger = build_ledger(case, rmg)
        discrepancies = deterministic_discrepancies(result["report"], ledger)
        if result["verifier_verdict"] == "PASS" and discrepancies:
            invalid_reports.append(
                {
                    "incident_id": result["incident_id"],
                    "discrepancies": discrepancies,
                }
            )
        status_counts[result["verifier_verdict"]] = (
            status_counts.get(result["verifier_verdict"], 0) + 1
        )
        for round_payload in result["rounds"]:
            canonical_repair_count += len(round_payload["canonical_repairs"])
            advisory = round_payload["verifier"]["llm_advisory_verdict"]
            advisory_verdicts[advisory] = advisory_verdicts.get(advisory, 0) + 1

    sensitive_hits = []
    scan_roots = [
        PROJECT / "src/root_teller/module3",
        FEEDBACK,
        REPORTS,
    ]
    for root in scan_roots:
        for path in root.rglob("*"):
            if path.suffix.lower() not in {".py", ".json", ".md", ".csv", ".txt"}:
                continue
            if path.name == "audit_checkpoint4.py":
                continue
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
            if "sk-" in text or "api_key =" in text or '"api_key"' in text:
                sensitive_hits.append(str(path))

    manifest_files = [
        PROJECT / "configs/module3_re2ob_feedback_frozen_2026-07-24.json",
        PROJECT / "src/root_teller/module3/config.py",
        PROJECT / "src/root_teller/module3/contracts.py",
        PROJECT / "src/root_teller/module3/feedback.py",
        PROJECT / "src/root_teller/module3/reporting.py",
        PROJECT / "src/root_teller/module3/run_feedback.py",
        PROJECT / "src/root_teller/module3/run_reports.py",
        PROJECT / "tests/test_module3_checkpoint4.py",
        FEEDBACK / "summary.json",
        FEEDBACK / "truthful_feedback_raw.csv",
        FEEDBACK / "false_feedback_raw.csv",
        FEEDBACK / "false_feedback_curve.csv",
        FEEDBACK / "false_feedback_curve_preview.png",
        FEEDBACK
        / "outputs/checkpoint4/RootTeller_Checkpoint4_Feedback_Analysis.xlsx",
        REPORTS / "summary.json",
    ]
    manifest = {
        str(path.relative_to(PROJECT)): _sha256(path)
        for path in manifest_files
        if path.exists()
    }

    result = {
        "schema_version": "checkpoint4-audit-1.0",
        "feedback": {
            "cases": len(truthful_rows),
            "selected_initial_top1_misses": len(selected),
            "successful_selected_within_20": len(successful_selected),
            "mean_rounds_success_only": (
                sum(int(row["rejected_rounds_to_correct"]) for row in successful_selected)
                / len(successful_selected)
            ),
            "truthful_feedback_events": event_count,
            "truthful_structural_evidence_mutations": structural_mutations,
            "feedback_event_ground_truth_field_leaks": feedback_target_leaks,
            "false_feedback_case_files": len(false_case_files),
            "false_feedback_structural_evidence_mutations": (
                false_structural_mutations
            ),
            "rejected_candidate_reactivations": rejected_reactivations,
        },
        "reports": {
            "files": len(list((REPORTS / "reports_json").glob("*.json"))),
            "status_counts": status_counts,
            "pass_reports_with_canonical_discrepancy": invalid_reports,
            "canonical_repairs": canonical_repair_count,
            "llm_advisory_verdicts": advisory_verdicts,
            "ground_truth_in_report_context": _json(REPORTS / "summary.json")[
                "ground_truth_in_report_context"
            ],
        },
        "sensitive_literal_hits": sensitive_hits,
        "checksums": manifest,
    }
    (REPORTS / "audit_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
