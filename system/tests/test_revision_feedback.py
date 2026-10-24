from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from root_teller.module3.structured.adapters import from_synthetic_ledger
from root_teller_system import engine


def fixture_state():
    root = Path(__file__).resolve().parents[2]
    return from_synthetic_ledger(json.loads((root / "tests/fixtures/feedback_ledger.json").read_text(encoding="utf-8")))


def payload():
    return {"job_id": "fixture", "live_llm": False, "result": {"incident_id": "SYN-INC-001"},
        "rmg": {"unchanged_source_marker": True},
        "ranking": [{"entity_id": "S-P", "base_score": 1., "feedback_confidence": 1.},
                    {"entity_id": "S-C", "base_score": 1., "feedback_confidence": 1.}],
        "feedback": {"feedback_events": []}, "report": {"report": {"root_cause_summary": "old report"}}}


def test_system_feedback_uses_coordinator_and_regenerates_report(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "RUNTIME_ROOT", tmp_path)
    monkeypatch.setattr(engine, "state_from_diagnosis", lambda *args: fixture_state())
    data = payload()
    source = copy.deepcopy(data["rmg"])
    update = engine.apply_feedback(data, "S-P", "REJECT", "Inspect this hypothesis", feedback_id="system-one")
    assert update["event"]["committed"]
    assert data["rmg"] == source
    assert update["event"]["tasks"]
    assert update["ranking"][0]["entity_id"] == "S-C"
    assert update["report"]["report"]["snapshot_id"] == update["report"]["snapshot_id"]
    assert update["feedback"]["execution_mode"] == "offline_contract_fixture_no_LLM"
    assert (tmp_path / "jobs/fixture/result.json").exists()


def test_system_idempotent_retry_and_stale_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "RUNTIME_ROOT", tmp_path)
    monkeypatch.setattr(engine, "state_from_diagnosis", lambda *args: fixture_state())
    data = payload()
    first = engine.apply_feedback(data, "S-P", "REJECT", "check", feedback_id="stable", snapshot_version=1)
    repeated = engine.apply_feedback(data, "S-P", "REJECT", "check", feedback_id="stable", snapshot_version=1)
    assert repeated["event"]["new_interactions"] == 0
    assert len(data["feedback"]["feedback_events"]) == 1
    assert first["ranking"] == repeated["ranking"]
    before = copy.deepcopy(data)
    with pytest.raises(ValueError, match="STALE_SNAPSHOT"):
        engine.apply_feedback(data, "S-C", "REJECT", "check", feedback_id="new", snapshot_version=1)
    assert data == before


def test_system_support_is_not_final_acceptance(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "RUNTIME_ROOT", tmp_path)
    monkeypatch.setattr(engine, "state_from_diagnosis", lambda *args: fixture_state())
    data = payload()
    update = engine.apply_feedback(data, "S-P", "SUPPORT", "Keep this observation", scope="observation", window_id="W3")
    assert update["event"]["committed"]
    assert not update["feedback"]["session_closed"]


def test_system_model_failure_does_not_partially_write_result(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "RUNTIME_ROOT", tmp_path)
    monkeypatch.setattr(engine, "state_from_diagnosis", lambda *args: fixture_state())
    data = payload()
    data["live_llm"] = True
    class Failing:
        requires_external_execution = False
        def complete(self, **kwargs):
            raise RuntimeError("fixture transport failure")
    monkeypatch.setattr(engine, "_client", lambda *args: Failing())
    before = copy.deepcopy(data)
    with pytest.raises(ValueError, match="PROCESSING_FAILED"):
        engine.apply_feedback(data, "S-P", "REJECT", "check")
    assert data == before
    assert not (tmp_path / "jobs/fixture/result.json").exists()
