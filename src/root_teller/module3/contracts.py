from __future__ import annotations

from typing import Any

from root_teller.module2.contracts import require_keys


def _string_list(value: object, field: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list or string")
    return [str(item) for item in value]


def validate_interpreter_response(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("interpreter response is not an object")
    require_keys(
        value,
        {
            "root_cause_summary",
            "ranked_alternatives",
            "propagation_chain",
            "evidence_ids",
            "limitations",
            "unresolved_issues",
            "recommended_next_checks",
        },
        "interpreter response",
    )
    return {
        "root_cause_summary": str(value["root_cause_summary"]),
        "ranked_alternatives": _string_list(
            value["ranked_alternatives"], "ranked_alternatives"
        ),
        "propagation_chain": _string_list(
            value["propagation_chain"], "propagation_chain"
        ),
        "evidence_ids": _string_list(value["evidence_ids"], "evidence_ids"),
        "limitations": _string_list(value["limitations"], "limitations"),
        "unresolved_issues": _string_list(
            value["unresolved_issues"], "unresolved_issues"
        ),
        "recommended_next_checks": _string_list(
            value["recommended_next_checks"], "recommended_next_checks"
        ),
    }


def validate_verifier_response(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("verifier response is not an object")
    require_keys(value, {"verdict", "discrepancies"}, "verifier response")
    verdict = str(value["verdict"]).upper()
    if verdict not in {"PASS", "REVISE", "EVIDENCE_GAP"}:
        raise ValueError(f"invalid verifier verdict: {verdict}")
    return {
        "verdict": verdict,
        "discrepancies": _string_list(value["discrepancies"], "discrepancies"),
    }
