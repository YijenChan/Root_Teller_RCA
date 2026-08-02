from __future__ import annotations

import json
import os
from pathlib import Path
import unittest

from root_teller.module3.feedback import (
    FeedbackRMGOverlay,
    false_feedback_run,
    truthful_feedback_run,
)
from root_teller.module3.reporting import (
    build_ledger,
    canonicalize_report,
    deterministic_discrepancies,
)


WORKSPACE = Path(os.environ.get("ROOTTELLER_WORKSPACE", Path.cwd())).resolve()
RUN = WORKSPACE / "runs/module2_re2ob/checkpoint3_v2_1_default_clean_replay"


@unittest.skipUnless(RUN.exists(), "requires a prepared Root-Teller workspace")
class Module3Checkpoint4Tests(unittest.TestCase):
    def test_reject_reduces_confidence_without_deleting_hypothesis(self) -> None:
        overlay = FeedbackRMGOverlay("re2ob-test", {"a": 0.8, "b": 0.7})
        event = overlay.commit(entity="a", verdict="REJECT", round_index=1)
        ranking = overlay.ranking(0.9)
        self.assertEqual(event["structural_evidence_mutated"], False)
        self.assertEqual({item["entity_id"] for item in ranking}, {"a", "b"})
        a = next(item for item in ranking if item["entity_id"] == "a")
        self.assertAlmostEqual(a["feedback_confidence"], 0.72)

    def test_truthful_feedback_reaches_target_with_soft_penalty(self) -> None:
        result = truthful_feedback_run(
            incident_id="re2ob-test",
            target="b",
            base_scores={"a": 0.8, "b": 0.7, "c": 0.1},
            decay=0.9,
            budget=20,
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["final_top1"], "b")
        self.assertGreaterEqual(result["rejected_rounds_to_correct"], 1)

    def test_false_rejection_can_recommend_correct_candidate_again(self) -> None:
        result = false_feedback_run(
            incident_id="re2ob-test",
            target="a",
            base_scores={"a": 0.8, "b": 0.79},
            decay=0.9,
            false_rejection_budget=2,
            total_budget=20,
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["false_rejections_applied"], 2)
        rejected = [
            item["feedback_event"]["entity_id"] for item in result["trace"]
        ]
        self.assertGreaterEqual(rejected.count("a"), 2)

    def test_report_discrepancy_gate_rejects_unknown_evidence(self) -> None:
        case_file = next((RUN / "cases").glob("*.json"))
        case = json.loads(case_file.read_text(encoding="utf-8"))
        rmg = json.loads((RUN / "rmg" / case_file.name).read_text(encoding="utf-8"))
        ledger = build_ledger(case, rmg)
        report = {
            "root_cause_summary": f"{ledger['selected_entity']} is leading.",
            "ranked_alternatives": [],
            "propagation_chain": ledger["propagation_chain"],
            "evidence_ids": ["invented-evidence"],
            "limitations": ledger["required_limitations"],
            "unresolved_issues": [],
            "recommended_next_checks": [],
        }
        discrepancies = deterministic_discrepancies(report, ledger)
        self.assertTrue(any("invalid evidence" in item for item in discrepancies))

    def test_report_canonicalization_repairs_typed_fields_only(self) -> None:
        case_file = next((RUN / "cases").glob("*.json"))
        case = json.loads(case_file.read_text(encoding="utf-8"))
        rmg = json.loads((RUN / "rmg" / case_file.name).read_text(encoding="utf-8"))
        ledger = build_ledger(case, rmg)
        report = {
            "root_cause_summary": "A leading hypothesis.",
            "ranked_alternatives": ["invented-service"],
            "propagation_chain": ["invented-service"],
            "evidence_ids": ["invented-evidence"],
            "limitations": [],
            "unresolved_issues": [],
            "recommended_next_checks": [],
        }
        repaired, repairs = canonicalize_report(report, ledger)
        self.assertIn(ledger["selected_entity"], repaired["root_cause_summary"])
        self.assertFalse(deterministic_discrepancies(repaired, ledger))
        self.assertGreater(len(repairs), 0)
        self.assertEqual(report["evidence_ids"], ["invented-evidence"])


if __name__ == "__main__":
    unittest.main()
