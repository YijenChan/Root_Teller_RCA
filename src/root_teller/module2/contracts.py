from __future__ import annotations

import hashlib
import json
from typing import Any


VERDICTS = {"SUPPORTED", "CONTRADICTED", "INCONCLUSIVE"}
ACTIONS = {"CONCLUDE", "EXPAND", "ABSTAIN"}


def stable_id(prefix: str, *parts: object, length: int = 16) -> str:
    raw = "\0".join(str(part) for part in parts)
    return f"{prefix}-{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:length]}"


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def require_keys(value: dict[str, Any], required: set[str], context: str) -> None:
    missing = required - set(value)
    if missing:
        raise ValueError(f"{context} missing keys: {sorted(missing)}")


def validate_steward_response(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("steward response is not an object")
    require_keys(
        value,
        {"ranked_entities", "selected_entity", "claims", "limitations"},
        "steward response",
    )
    if not isinstance(value["ranked_entities"], list) or not value["ranked_entities"]:
        raise ValueError("ranked_entities must be a non-empty list")
    if not all(isinstance(item, str) and item for item in value["ranked_entities"]):
        raise ValueError("ranked_entities contains invalid entity")
    if not isinstance(value["selected_entity"], str):
        raise ValueError("selected_entity must be a string")
    if value["selected_entity"] not in value["ranked_entities"]:
        raise ValueError("selected_entity is absent from ranked_entities")
    if not isinstance(value["claims"], list):
        raise ValueError("claims must be a list")
    normalized_claims = []
    for index, claim in enumerate(value["claims"]):
        if not isinstance(claim, dict):
            raise ValueError(f"claim {index} is not an object")
        require_keys(claim, {"claim", "entity", "window_ids", "evidence_ids"}, f"claim {index}")
        if not all(isinstance(claim[key], list) for key in ("window_ids", "evidence_ids")):
            raise ValueError(f"claim {index} references must be lists")
        normalized_claims.append(
            {
                "claim": str(claim["claim"]),
                "entity": str(claim["entity"]),
                "window_ids": [str(item) for item in claim["window_ids"]],
                "evidence_ids": [str(item) for item in claim["evidence_ids"]],
            }
        )
    return {
        "ranked_entities": list(dict.fromkeys(value["ranked_entities"])),
        "selected_entity": value["selected_entity"],
        "claims": normalized_claims,
        "limitations": [str(item) for item in value["limitations"]],
    }


def validate_investigator_response(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("investigator response is not an object")
    require_keys(
        value,
        {
            "verdict",
            "supporting_evidence_ids",
            "contradicting_evidence_ids",
            "verified_relations",
            "limitations",
        },
        "investigator response",
    )
    verdict = str(value["verdict"]).upper()
    if verdict not in VERDICTS:
        raise ValueError(f"invalid investigator verdict: {verdict}")
    normalized_lists: dict[str, list[str]] = {}
    for field in (
        "supporting_evidence_ids",
        "contradicting_evidence_ids",
        "verified_relations",
        "limitations",
    ):
        raw = value[field]
        if raw is None:
            raw = []
        elif isinstance(raw, str):
            raw = [raw]
        elif field == "verified_relations" and isinstance(raw, dict):
            # Some OpenAI-compatible endpoints emit one relation object instead
            # of a one-element array. Preserve the relation rather than
            # discarding the otherwise valid Investigator response.
            raw = [raw]
        if not isinstance(raw, list):
            raise ValueError(f"{field} must be a list or string")
        normalized_lists[field] = [str(item) for item in raw]
    return {
        "verdict": verdict,
        **normalized_lists,
    }
