from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from root_teller.module3.session import build_coordinator, refinement_config
from root_teller.module3.structured.adapters import from_synthetic_ledger
from root_teller.module3.structured.contracts import default_item, event_for
from root_teller.module3.structured.state import FeedbackState
from root_teller.module3.refinement.steward import OfflineContractClient
from root_teller.module3.evaluate_feedback import run_trajectory, summarize


def initial():
    ledger = json.loads((Path(__file__).parent / "fixtures/feedback_ledger.json").read_text(encoding="utf-8"))
    return from_synthetic_ledger(ledger)


def review(state, cid="C02", feedback_id="one", stance="challenge"):
    claim = state.claims[cid]
    return event_for(state, feedback_id, [default_item(item_id="item", target_claim_id=cid,
        entity_ids=claim["entity_ids"], window_ids=claim["window_ids"], stance=stance)])


class RecordedProposalClient:
    """An injected protocol fixture, never a paid model or an efficacy result."""
    requires_external_execution = False

    def __init__(self):
        self.calls = []

    def complete(self, *, role, payload, validator, **kwargs):
        self.calls.append((role, copy.deepcopy(payload)))
        if role == "feedback_window_investigator":
            value = {"status": "UNRESOLVED", "evidence_refs": [], "limitations": ["No independent causal evidence."]}
        else:
            value, _ = OfflineContractClient().complete(payload=payload, validator=lambda x: x)
            value["ranked_entities"] = list(reversed(value["ranked_entities"]))
        return validator(value), {"network_request": False, "execution_mode": "protocol_fixture"}


def test_rejection_runs_targeted_checks_and_fresh_global_synthesis():
    client = RecordedProposalClient()
    coordinator = build_coordinator(initial(), client=client)
    assert not client.calls  # T0 is frozen, not silently resynthesized.
    before = coordinator.state.immutable_hash
    result = coordinator.submit(review(coordinator.state))
    assert result["committed"], result
    roles = [role for role, _ in client.calls]
    assert roles.count("feedback_window_investigator") == 2
    assert roles.count("cp4_evidence_steward") == 1
    assert coordinator.state.immutable_hash == before
    assert result["snapshot"]["synthesis"]["selected_entity"] == "S-C"
    payment = next(r for r in result["snapshot"]["synthesis"]["ranked_candidates"] if r["entity_id"] == "S-P")
    assert payment["score"] == pytest.approx(0.8 * 0.9)
    assert coordinator.state.feedback_items[-1]["submitted_at"]
    assert result["report"]["snapshot_id"] == result["snapshot"]["snapshot_id"]


def test_duplicate_event_does_not_repeat_model_calls_or_decay():
    client = RecordedProposalClient()
    coordinator = build_coordinator(initial(), client=client)
    event = review(coordinator.state)
    first = coordinator.submit(event)
    calls = len(client.calls)
    second = coordinator.submit(event)
    assert first["committed"] and second["new_interactions"] == 0
    assert len(client.calls) == calls
    assert coordinator.state.challenge_counts["C02"] == 1


def test_supporting_observation_is_separate_from_final_acceptance():
    coordinator = build_coordinator(initial())
    result = coordinator.submit(review(coordinator.state, "C01", stance="support"))
    assert result["committed"] and not coordinator.state.session_closed
    assert not coordinator.state.challenge_counts


def test_failed_model_transaction_preserves_snapshot_and_source():
    class Failing(RecordedProposalClient):
        def complete(self, **kwargs):
            raise RuntimeError("Recorded model failure")
    coordinator = build_coordinator(initial(), client=Failing())
    before = coordinator.state.diagnostic_digest()
    result = coordinator.submit(review(coordinator.state))
    assert result["event_status"] == "PROCESSING_FAILED"
    assert not result["committed"] and coordinator.state.diagnostic_digest() == before


def test_serialized_session_retains_receipts_and_latest_report():
    coordinator = build_coordinator(initial())
    event = review(coordinator.state)
    assert coordinator.submit(event)["committed"]
    restored = build_coordinator(FeedbackState.restore(coordinator.state.artifact()))
    assert restored.submit(event)["new_interactions"] == 0
    assert restored.state.snapshots[-1]["version"] == restored.state.version


def test_first_correct_acceptance_counts_one_interaction():
    state = initial()
    target = max(state.base_scores, key=state.base_scores.get)
    for strategy in ["full", "flat", "stateful_single_agent", "top5", "downweighting"]:
        row = run_trajectory(state, target, strategy=strategy)
        assert row["success"] and row["interactions"] == 1
        assert not state.feedback_items


def test_top5_false_rejection_fails_without_revisiting():
    state = initial()
    target = max(state.base_scores, key=state.base_scores.get)
    row = run_trajectory(state, target, strategy="top5", false_rejections=1)
    assert not row["success"] and row["capped_interactions"] == 20


def test_offline_root_only_path_does_not_fabricate_model_gain():
    state = initial()
    target = max(state.base_scores, key=state.base_scores.get)
    full = run_trajectory(state, target, strategy="full", false_rejections=1)
    decay = run_trajectory(state, target, strategy="downweighting", false_rejections=1)
    assert full["success"] == decay["success"]
    assert full["capped_interactions"] == decay["capped_interactions"]


def test_summary_distinguishes_last_round_success_from_failure():
    rows = [{"strategy": "full", "K": 1, "success": True, "capped_interactions": 20},
            {"strategy": "full", "K": 1, "success": False, "capped_interactions": 20}]
    result = summarize(rows)[0]
    assert result["mean_capped_interactions"] == 20
    assert result["success_rate_within_budget"] == 0.5


def test_no_feedback_reference_is_independent_of_false_rejection_budget():
    state = initial()
    for k in [0, 1, 5]:
        assert run_trajectory(state, "S-P", strategy="no_feedback", false_rejections=k)["reference_only"]
