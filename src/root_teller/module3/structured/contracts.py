from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

SCHEMA_PATH = Path(__file__).with_name("feedback_schema.json")
SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
STANCE = {"support": 1, "challenge": -1, "unsure": 0}
EVIDENCE_ENCODING = {"SUPPORTED": (1, 1), "CONTRADICTED": (-1, 1), "UNRESOLVED": (0, 0)}


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def stable_id(prefix: str, *parts: Any) -> str:
    return f"{prefix}-{digest(list(parts))[:20]}"


def _type(value: Any, kind: str) -> bool:
    checks = {
        "object": isinstance(value, dict), "array": isinstance(value, list),
        "string": isinstance(value, str), "integer": isinstance(value, int) and not isinstance(value, bool),
        "boolean": isinstance(value, bool), "null": value is None,
    }
    return checks[kind]


def format_errors(value: Any, rule: dict | None = None, path: str = "event") -> list[str]:
    """Explicit subset validator for every keyword used by the CP1 schema."""
    if rule is None:
        rule = SCHEMA
    supported = {"$schema", "$id", "title", "description", "type", "const", "enum", "properties",
                 "required", "additionalProperties", "items", "minItems", "maxItems", "uniqueItems",
                 "minLength", "maxLength", "minimum"}
    if set(rule) - supported:
        raise ValueError("Unsupported schema keywords; use a full schema validator before extending")
    result = []
    if "type" in rule:
        kinds = rule["type"] if isinstance(rule["type"], list) else [rule["type"]]
        if not any(_type(value, kind) for kind in kinds):
            return [f"{path}: invalid type"]
    if "const" in rule and value != rule["const"]:
        result.append(f"{path}: invalid constant")
    if "enum" in rule and value not in rule["enum"]:
        result.append(f"{path}: invalid enum")
    if isinstance(value, dict):
        result += [f"{path}: missing {key}" for key in rule.get("required", []) if key not in value]
        if rule.get("additionalProperties") is False:
            result += [f"{path}: unknown {key}" for key in value if key not in rule.get("properties", {})]
        for key, subrule in rule.get("properties", {}).items():
            if key in value:
                result += format_errors(value[key], subrule, f"{path}.{key}")
    if isinstance(value, list):
        if not rule.get("minItems", 0) <= len(value) <= rule.get("maxItems", math.inf):
            result.append(f"{path}: invalid item count")
        if rule.get("uniqueItems") and len(set(map(canonical, value))) != len(value):
            result.append(f"{path}: repeated array entry")
        if "items" in rule:
            for index, item in enumerate(value):
                result += format_errors(item, rule["items"], f"{path}[{index}]")
    if isinstance(value, str) and not rule.get("minLength", 0) <= len(value) <= rule.get("maxLength", math.inf):
        result.append(f"{path}: invalid string length")
    if isinstance(value, int) and not isinstance(value, bool) and value < rule.get("minimum", -math.inf):
        result.append(f"{path}: below minimum")
    return result


def default_item(**fields: Any) -> dict:
    return {"item_id": "", "target_kind": "existing_claim", "target_claim_id": None,
            "entity_ids": [], "entity_labels": [], "window_ids": [], "stance": "unsure",
            "reason": "", "requested_check": "", "evidence_refs": [], "proposed_text": None, **fields}


def event_for(state: Any, feedback_id: str, items: list[dict], event_type: str = "comment") -> dict:
    return {"schema_version": "feedback-cp1-proposal-1.0", "feedback_id": feedback_id,
            "incident_id": state.raw["incident_id"], "snapshot_version": state.version,
            "source_mode": "structured", "binding_confirmed": True, "event_type": event_type, "items": items}


def gate_record(item_id: str, status: str, evidence_refs: list[str] | None = None, **extra: Any) -> dict:
    value, mask = EVIDENCE_ENCODING.get(status, (0, 0))
    return {"item_id": item_id, "status": status, "evidence_value": value, "evidence_mask": mask,
            "evidence_refs": evidence_refs or [], **extra}


def new_claim_key(incident: str, entities: list[str], windows: list[str], text: str) -> str:
    # IDs survive snapshot versions; normalized duplicate proposals resolve to one axis.
    normalized = " ".join(text.lower().split())
    return stable_id("PENDING", incident, sorted(entities), sorted(windows), normalized)
