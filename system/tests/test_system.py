from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

import pytest

from root_teller_system.engine import apply_feedback, inspect_case_path, safe_extract_zip


OB_CASE = Path(os.environ.get("ROOTTELLER_TEST_CASE", "__dataset_not_configured__"))


def test_re2_family_expands_only_numeric_repetitions() -> None:
    if not OB_CASE.exists():
        pytest.skip("local dataset not installed")
    result = inspect_case_path(str(OB_CASE))
    assert result["dataset"] == "re2_ob"
    assert [item["repetition"] for item in result["variants"]] == ["1", "2", "3"]


def test_zip_slip_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escape.txt", "unsafe")
    with pytest.raises(ValueError, match="Unsafe archive"):
        safe_extract_zip(archive, tmp_path / "output")


def test_uploaded_layout_is_detected_from_content(tmp_path: Path) -> None:
    incident = tmp_path / "checkoutservice_cpu" / "1"
    incident.mkdir(parents=True)
    (incident / "metrics.csv").write_text("time,checkoutservice_cpu\n1,0.5\n", encoding="utf-8")
    (incident / "logs.csv").write_text("time,service,message\n", encoding="utf-8")
    (incident / "traces.csv").write_text("time,service\n", encoding="utf-8")
    result = inspect_case_path(str(tmp_path / "checkoutservice_cpu"))
    assert result["dataset"] == "re2_ob"
    assert result["variants"][0]["id"] == "checkoutservice_cpu/1"


def test_feedback_rejects_unknown_entity() -> None:
    payload = {
        "job_id": "x",
        "result": {"incident_id": "incident"},
        "ranking": [
            {
                "entity_id": "checkoutservice",
                "base_score": 0.8,
                "reject_count": 0,
                "feedback_confidence": 0.8,
            }
        ],
        "feedback": {
            "current_ranking": [
                {
                    "entity_id": "checkoutservice",
                    "base_score": 0.8,
                    "reject_count": 0,
                    "feedback_confidence": 0.8,
                }
            ],
            "reject_counts": {},
            "feedback_events": [],
        },
        "report": {
            "report": {"root_cause_summary": "checkoutservice", "ranked_alternatives": []}
        },
    }
    with pytest.raises(ValueError, match="current RMG hypothesis"):
        apply_feedback(payload, "unknown", "REJECT", "not observed")
