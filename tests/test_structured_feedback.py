from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest
from concurrent.futures import ThreadPoolExecutor


from root_teller.module3.structured import FeedbackCoordinator, FeedbackState
from root_teller.module3.structured.adapters import from_synthetic_ledger
from root_teller.module3.structured.agents import (JSONClientInvestigator, JSONClientSteward, OfflineEvidenceSteward,
                                OfflineReportInterpreter, OfflineWindowInvestigator, TypedEvidenceGate)
from root_teller.module3.structured.contracts import default_item, digest, event_for

ROOT = Path(__file__).resolve().parent / "fixtures"
CONFIG = json.loads((ROOT / "feedback_config.json").read_text(encoding="utf-8"))
LEDGER = json.loads((ROOT / "feedback_ledger.json").read_text(encoding="utf-8"))
FIXTURES = json.loads((ROOT / "feedback_cases.json").read_text(encoding="utf-8"))


def engine():
    return FeedbackCoordinator(from_synthetic_ledger(LEDGER), CONFIG)


def fixture_event(case):
    event = {**copy.deepcopy(FIXTURES["event_defaults"]), **copy.deepcopy(case["event"])}
    event["items"] = [{**copy.deepcopy(FIXTURES["item_defaults"]), **item} for item in event["items"]]
    return event


def challenge(rt, feedback_id="TEST-1", claim_id="C02"):
    claim = rt.state.claims[claim_id]
    return event_for(rt.state, feedback_id, [default_item(item_id="I1", target_claim_id=claim_id,
                     entity_ids=claim["entity_ids"], window_ids=claim["window_ids"], stance="challenge")])


class FixtureTests(unittest.TestCase):
    def test_all_twelve_cp1_fixtures(self):
        for case in FIXTURES["cases"]:
            with self.subTest(fixture=case["id"]):
                rt = engine()
                event = fixture_event(case)
                if case["id"] == "F07":
                    rt.state.version = 2
                if case["id"] == "F10":
                    applied = rt.submit(event)
                    self.assertTrue(applied["committed"])
                before_raw = digest(rt.state.raw)
                result = rt.submit(event)
                expected = case["expected"]
                self.assertEqual(result["event_status"], expected["event_status"])
                vectors = copy.deepcopy(result["vector_sparse"])
                if case["id"] == "F09":
                    # CP1 provisional placeholder; runtime axis is stable content-derived ID.
                    self.assertTrue(vectors[0]["axis"].startswith("PENDING-"))
                    vectors[0]["axis"] = "PENDING-F09-I09"
                self.assertEqual(vectors, expected["vector_sparse"])
                gates = [{key: g[key] for key in ["item_id", "status", "evidence_value", "evidence_mask"]} for g in result["gate"]]
                self.assertEqual(gates, expected["gate"])
                for key in ["review_windows", "soft_prior_targets", "strong_state_changes", "new_interactions", "end_session"]:
                    self.assertEqual(result[key], expected[key], key)
                self.assertEqual(digest(rt.state.raw), before_raw)
                self.assertEqual(result["immutable_hash_before"], result["immutable_hash_after"])

    def test_gate_does_not_read_expected_annotations(self):
        altered = copy.deepcopy(LEDGER)
        for row in altered["claims"].values():
            row["fixture_evidence_verdict"] = "CONTRADICTED"
        rt = FeedbackCoordinator(from_synthetic_ledger(altered), CONFIG)
        event = fixture_event(FIXTURES["cases"][1])
        self.assertEqual(rt.submit(event)["gate"][0]["status"], "SUPPORTED")


class SafetyAndTransactionTests(unittest.TestCase):
    def test_atomic_multi_item_invalid_reference(self):
        rt = engine()
        event = challenge(rt)
        event["items"].append(default_item(item_id="I2", target_claim_id="C01", entity_ids=["S-P"], window_ids=["W3"], stance="support", evidence_refs=["E404"]))
        before = rt.state.diagnostic_digest()
        result = rt.submit(event)
        self.assertEqual(result["event_status"], "NEEDS_CLARIFICATION")
        self.assertEqual(rt.state.diagnostic_digest(), before)
        self.assertEqual(rt.state.challenge_counts, {})

    def test_invalid_targets_and_scopes_and_confirmation(self):
        variants = [("target_claim_id", "MISSING", "NEEDS_CLARIFICATION"), ("entity_ids", ["S-C"], "NEEDS_CLARIFICATION"),
                    ("window_ids", ["HIDDEN"], "NEEDS_CLARIFICATION")]
        for field, value, expected in variants:
            rt = engine(); event = challenge(rt); event["items"][0][field] = value
            self.assertEqual(rt.submit(event)["event_status"], expected)
            self.assertFalse(rt.state.challenge_counts)
        rt = engine(); event = challenge(rt); event["binding_confirmed"] = False
        self.assertEqual(rt.submit(event)["event_status"], "NEEDS_CONFIRMATION")

    def test_incident_mismatch(self):
        rt = engine(); event = challenge(rt); event["incident_id"] = "OTHER"
        self.assertEqual(rt.submit(event)["event_status"], "INCIDENT_MISMATCH")

    def test_duplicate_target_and_item(self):
        for same_item_id in [True, False]:
            rt = engine(); event = challenge(rt); second = copy.deepcopy(event["items"][0])
            second["stance"] = "support"
            if not same_item_id:
                second["item_id"] = "I2"
            event["items"].append(second)
            self.assertIn(rt.submit(event)["event_status"], {"DUPLICATE_ITEM", "NEEDS_CLARIFICATION"})
            self.assertEqual(rt.state.challenge_counts, {})

    def test_id_conflict_does_not_replace_receipt(self):
        rt = engine(); event = challenge(rt); original = rt.submit(event)
        changed = copy.deepcopy(event); changed["items"][0]["reason"] = "different"
        self.assertEqual(rt.submit(changed)["event_status"], "ID_CONFLICT")
        self.assertEqual(rt.submit(event)["original_event_status"], original["event_status"])
        self.assertEqual(rt.state.submissions, 1)

    def test_idempotence_survives_reload_and_new_version(self):
        rt = engine(); event = challenge(rt); rt.submit(event)
        restored = FeedbackCoordinator(FeedbackState.restore(json.loads(json.dumps(rt.state.artifact()))), CONFIG)
        replay = restored.submit(event)
        self.assertEqual(replay["event_status"], "IDEMPOTENT_REPLAY")
        self.assertEqual([r["stage"] for r in replay["pipeline_trace"]], ["IDEMPOTENT_RETURN"])
        self.assertEqual(restored.state.challenge_counts, {"C02": 1})
        self.assertEqual(restored.state.submissions, 1)

    def test_concurrent_same_payload_is_applied_once(self):
        rt = engine(); event = challenge(rt)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(rt.submit, [event, event]))
        self.assertEqual(sorted(r["new_interactions"] for r in results), [0, 1])
        self.assertEqual(rt.state.challenge_counts, {"C02": 1})

    def test_investigator_failure_rolls_back(self):
        class Failing:
            def inspect(self, task):
                raise RuntimeError("intentional backend failure")
        rt = engine(); before = rt.state.diagnostic_digest(); rt.investigator = Failing()
        result = rt.submit(challenge(rt))
        self.assertEqual(result["event_status"], "PROCESSING_FAILED")
        self.assertEqual(rt.state.diagnostic_digest(), before)
        self.assertEqual(result["soft_prior_targets"], [])
        self.assertEqual(result["new_interactions"], 1)

    def test_report_verification_failure_rolls_back(self):
        class BadInterpreter(OfflineReportInterpreter):
            def generate(self, snapshot, state):
                report = super().generate(snapshot, state)
                report["evidence_ids"].append("E404")
                return report
        rt = engine(); before = rt.state.diagnostic_digest(); rt.interpreter = BadInterpreter()
        result = rt.submit(challenge(rt))
        self.assertEqual(result["event_status"], "PROCESSING_FAILED")
        self.assertEqual(rt.state.diagnostic_digest(), before)

    def test_out_of_scope_investigator_reference_rolls_back(self):
        class Bad(OfflineWindowInvestigator):
            def inspect(self, task):
                result = super().inspect(task); result["evidence_refs"] = ["E404"]
                return result
        rt = engine(); rt.investigator = Bad()
        self.assertEqual(rt.submit(challenge(rt))["event_status"], "PROCESSING_FAILED")
        self.assertFalse(rt.state.challenge_counts)

    def test_plausible_but_unproven_root_support_is_capped(self):
        class Opinion(OfflineWindowInvestigator):
            def inspect(self, task):
                result = super().inspect(task)
                result["status"] = "SUPPORTED"
                result["evidence_refs"] = list(task["evidence"])[:1]
                return result
        rt = engine(); rt.investigator = Opinion()
        result = rt.submit(challenge(rt))
        self.assertTrue(result["committed"])
        self.assertTrue(all(r["status"] == "UNRESOLVED" and r["grounding_override"] for r in result["task_results"]))
        self.assertEqual(rt.state.claims["C02"]["state"], "unresolved")

    def test_no_window_wide_decay_or_direct_unmentioned_feedback(self):
        rt = engine(); result = rt.submit(fixture_event(FIXTURES["cases"][3]))
        self.assertEqual(rt.state.challenge_counts, {"C08": 1})
        self.assertEqual(rt.state.claims["C01"]["state"], "supported")
        self.assertEqual(rt.state.claims["C02"]["state"], "unresolved")
        self.assertTrue(all(f["target_claim_id"] != "C02" for f in rt.state.feedback_items))
        payment = next(row for row in result["snapshot"]["synthesis"]["ranked_candidates"] if row["entity_id"] == "S-P")
        factors = {row["window_id"]: row["hypothesis_prior_factor"] for row in payment["local_contributions"]}
        self.assertEqual(factors, {"W1": 1.0, "W2": 1.0, "W3": 0.9})

    def test_unsure_mask_requests_check_without_rejection(self):
        rt = engine(); result = rt.submit(fixture_event(FIXTURES["cases"][2]))
        self.assertEqual(result["vector_sparse"], [{"axis": "C05", "mask": 1, "stance_value": 0}])
        self.assertTrue(result["tasks"])
        self.assertEqual(rt.state.challenge_counts, {})

    def test_known_support_is_no_new_evidence_no_op(self):
        rt = engine(); before = rt.state.snapshots[-1]["synthesis"]["ranked_candidates"]
        result = rt.submit(fixture_event(FIXTURES["cases"][1]))
        self.assertEqual(result["tasks"], [])
        self.assertEqual(result["snapshot"]["synthesis"]["ranked_candidates"], before)

    def test_report_snapshot_and_relation_state_rebuilt(self):
        rt = engine(); initial = copy.deepcopy(rt.state.snapshots[-1])
        result = rt.submit(fixture_event(FIXTURES["cases"][10]))
        self.assertNotEqual(initial["snapshot_id"], result["snapshot"]["snapshot_id"])
        self.assertEqual(result["report"]["snapshot_id"], result["snapshot"]["snapshot_id"])
        c07 = next(c for c in result["snapshot"]["synthesis"]["derived_relations"] if c["claim_id"] == "C07")
        self.assertEqual(c07["state"], "contradicted")
        self.assertIn("C07", [c["claim_id"] for c in result["report"]["contradicted_claims"]])
        self.assertEqual(initial, rt.state.snapshots[0])

    def test_final_accept_requires_current_global_leader(self):
        for claim_id in ["C01", "C06"]:
            rt = engine(); claim = rt.state.claims[claim_id]
            event = event_for(rt.state, "ACCEPT", [default_item(item_id="I", target_claim_id=claim_id, entity_ids=claim["entity_ids"], window_ids=claim["window_ids"], stance="support")], "final_accept")
            self.assertEqual(rt.submit(event)["event_status"], "INVALID_FINAL_ACCEPT")
            self.assertFalse(rt.state.session_closed)
        rt = engine(); result = rt.submit(fixture_event(FIXTURES["cases"][11]))
        self.assertTrue(result["end_session"])
        self.assertEqual(rt.state.claims["C02"]["state"], "unresolved")
        self.assertEqual(rt.submit(challenge(rt, "AFTER-END"))["event_status"], "SESSION_CLOSED")

    def test_final_accept_does_not_rerun_diagnosis(self):
        class NeverRerank:
            requires_external_execution = True
            def reason(self, state, config):
                raise AssertionError("Final acceptance must not rerun diagnosis")
        rt = engine(); initial = copy.deepcopy(rt.state.snapshots[-1]["synthesis"]); rt.steward = NeverRerank()
        result = rt.submit(fixture_event(FIXTURES["cases"][11]))
        self.assertEqual(result["event_status"], "OPERATOR_ACCEPTED")
        self.assertEqual(result["snapshot"]["synthesis"], initial)
        self.assertIn("REUSE_ACCEPTED_DIAGNOSIS", [p["stage"] for p in result["pipeline_trace"]])

    def test_new_hypothesis_is_deduplicated_across_versions(self):
        rt = engine(); event = fixture_event(FIXTURES["cases"][8]); first = rt.submit(event)
        axis = first["vector_sparse"][0]["axis"]
        second = copy.deepcopy(event); second["feedback_id"] = "NEW-ID"; second["snapshot_version"] = rt.state.version
        result = rt.submit(second)
        self.assertEqual(result["vector_sparse"][0]["axis"], axis)
        self.assertEqual(len([cid for cid in rt.state.claims if cid.startswith("PENDING-")]), 1)
        self.assertEqual(rt.state.claims[axis]["state"], "pending")

    def test_immutable_tamper_rejected(self):
        rt = engine(); rt.state.raw["evidence"]["E01"]["facts"]["log_contains_timeout"] = False
        with self.assertRaisesRegex(ValueError, "immutable"):
            rt.submit(challenge(rt))
        with self.assertRaises(ValueError):
            artifact = rt.state.artifact()

    def test_budget_and_one_based_count(self):
        rt = engine(); rt.config["submission_budget"] = 2
        self.assertEqual(rt.submit(challenge(rt, "FIRST"))["submissions_total"], 1)
        self.assertEqual(rt.submit(challenge(rt, "SECOND"))["submissions_total"], 2)
        self.assertEqual(rt.submit(challenge(rt, "THIRD"))["event_status"], "BUDGET_EXHAUSTED")

    def test_track_a_single_item_profile_is_enforced(self):
        rt = engine(); rt.config["max_items"] = 1
        self.assertEqual(rt.submit(fixture_event(FIXTURES["cases"][3]))["event_status"], "ITEM_BUDGET_EXCEEDED")
        self.assertEqual(rt.state.challenge_counts, {})

    def test_no_feedback_time_forgetting(self):
        rt = engine(); rt.submit(challenge(rt)); before = copy.deepcopy(rt.state.challenge_counts)
        rt.submit(fixture_event({**FIXTURES["cases"][2], "event": {**FIXTURES["cases"][2]["event"], "snapshot_version": rt.state.version}}))
        self.assertEqual(rt.state.challenge_counts, before)

    def test_bounded_tasks_include_competitor_with_long_history(self):
        state = from_synthetic_ledger(LEDGER)
        for i in range(4, 25):
            state.raw["windows"][f"W{i}"] = {"start_time": i * 60, "end_time": (i + 1) * 60}
        for cid in ["C02", "C06"]:
            state.claims[cid]["window_ids"] = list(state.raw["windows"])
        state.immutable_hash = digest(state.raw)
        rt = FeedbackCoordinator(state, CONFIG); result = rt.submit(challenge(rt))
        self.assertLessEqual(len(result["tasks"]), CONFIG["max_tasks"])
        self.assertTrue(any(not t["direct_feedback"] for t in result["tasks"]))
        self.assertTrue(all(len(t["window_ids"]) <= CONFIG["max_windows_per_task"] for t in result["tasks"]))
        self.assertTrue(all(set(t["window_ids"]) <= set(state.raw["windows"]) for t in result["tasks"]))

    def test_different_concurrent_events_obey_snapshot_version(self):
        rt = engine(); first = challenge(rt, "ONE"); second = challenge(rt, "TWO")
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(rt.submit, [first, second]))
        self.assertEqual(sum(r["committed"] for r in results), 1)
        self.assertIn("STALE_SNAPSHOT", [r["event_status"] for r in results])
        self.assertEqual(rt.state.challenge_counts, {"C02": 1})

    def test_repeated_global_target_prioritizes_unreviewed_windows(self):
        state = from_synthetic_ledger(LEDGER)
        for i in range(4, 13):
            state.raw["windows"][f"W{i}"] = {"start_time": i * 60, "end_time": (i + 1) * 60}
        for cid in ["C02", "C06"]:
            state.claims[cid]["window_ids"] = list(state.raw["windows"])
        state.immutable_hash = digest(state.raw)
        rt = FeedbackCoordinator(state, CONFIG)
        first = rt.submit(challenge(rt, "FIRST")); second = rt.submit(challenge(rt, "SECOND"))
        def target_windows(result):
            return {w for t in result["tasks"] if t["claim"]["claim_id"] == "C02" for w in t["window_ids"]}
        self.assertTrue(target_windows(first))
        self.assertTrue(target_windows(second))
        self.assertFalse(target_windows(first) & target_windows(second))
        self.assertEqual(first["immutable_hash_after"], second["immutable_hash_after"])

    def test_persisted_derived_snapshot_mismatch_is_rejected(self):
        rt = engine(); artifact = rt.state.artifact(); artifact["claims"]["C02"]["text"] = "silently changed"
        with self.assertRaisesRegex(ValueError, "derived state"):
            FeedbackCoordinator(FeedbackState.restore(artifact), CONFIG)

    def test_steward_cannot_rewrite_relation_text(self):
        class BadSteward(OfflineEvidenceSteward):
            def reason(self, state, config):
                result = super().reason(state, config)
                result["derived_relations"][0]["text"] = "unsupported new causal fact"
                return result
        rt = engine(); rt.steward = BadSteward(); before = rt.state.diagnostic_digest()
        self.assertEqual(rt.submit(challenge(rt))["event_status"], "PROCESSING_FAILED")
        self.assertEqual(rt.state.diagnostic_digest(), before)

    def test_gate_rejects_cross_entity_and_window_predicate(self):
        rt = engine(); claim = copy.deepcopy(rt.state.claims["C01"])
        claim["predicate"] = {"type": "field_equals", "field": "queue_deviation", "value": True, "evidence_refs": ["E02"]}
        result = TypedEvidenceGate().evaluate(claim, rt.state.raw["evidence"], "TEST")
        self.assertEqual(result["status"], "UNRESOLVED")

    def test_external_clients_are_not_called_without_gate(self):
        class NeverCall:
            def complete(self, **kwargs):
                self.fail("External client must not be called")
        client = NeverCall()
        rt = FeedbackCoordinator(from_synthetic_ledger(LEDGER), CONFIG,
                                  investigator=JSONClientInvestigator(client), steward=JSONClientSteward(client))
        self.assertEqual(rt.submit(challenge(rt))["event_status"], "EXTERNAL_EXECUTION_NOT_AUTHORIZED")
        self.assertEqual(rt.state.challenge_counts, {})

    def test_stub_model_steward_receives_new_claim_state(self):
        class Stub:
            def complete(self, **kwargs):
                claims = {c["claim_id"]: c for c in kwargs["payload"]["derived_claims"]}
                if claims["C07"]["state"] != "contradicted":
                    raise AssertionError("Steward did not receive the newly validated claim state")
                response = {"ranked_entities": ["S-C", "S-P"], "rationales": [
                    {"entity_id": "S-C", "basis_claim_ids": ["C03"], "explanation": "Stub only"},
                    {"entity_id": "S-P", "basis_claim_ids": ["C01"], "explanation": "Stub only"}]}
                return kwargs["validator"](response), {"stub_only": True, "network_calls": 0}
        rt = engine(); rt.config["external_API_authorized"] = True  # In-memory stub, no actual API.
        rt.steward = JSONClientSteward(Stub())
        result = rt.submit(fixture_event(FIXTURES["cases"][10]))
        self.assertTrue(result["committed"])
        self.assertTrue(result["snapshot"]["synthesis"]["client_metadata"]["stub_only"])
        self.assertEqual(result["snapshot"]["synthesis"]["selected_entity"], "S-C")

    def test_stub_investigator_causal_proposal_remains_unresolved(self):
        class Stub:
            def complete(self, **kwargs):
                response = {"status": "SUPPORTED", "evidence_refs": list(kwargs["payload"]["evidence"])[:1], "limitations": []}
                return kwargs["validator"](response), {"stub_only": True, "network_calls": 0}
        rt = engine(); rt.config["external_API_authorized"] = True
        rt.investigator = JSONClientInvestigator(Stub())
        result = rt.submit(challenge(rt))
        self.assertTrue(result["committed"])
        self.assertTrue(all(r["status"] == "UNRESOLVED" for r in result["task_results"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
