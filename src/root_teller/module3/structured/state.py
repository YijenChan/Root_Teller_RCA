from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any

from .contracts import digest


@dataclass
class FeedbackState:
    """Two layers: immutable public source evidence, versioned derived diagnosis."""
    raw: dict[str, Any]
    claims: dict[str, dict]
    base_scores: dict[str, float]
    version: int = 1
    local_contribution_weights: dict[str, dict[str, float]] = field(default_factory=dict)
    challenge_counts: dict[str, int] = field(default_factory=dict)
    feedback_items: list[dict] = field(default_factory=list)
    validations: list[dict] = field(default_factory=list)
    external_clues: dict[str, dict] = field(default_factory=dict)
    receipts: dict[str, dict] = field(default_factory=dict)
    submissions: int = 0
    snapshots: list[dict] = field(default_factory=list)
    session_closed: bool = False
    operator_accepted_claim: str | None = None
    immutable_hash: str = ""

    def __post_init__(self):
        self.raw = copy.deepcopy(self.raw)
        self.claims = copy.deepcopy(self.claims)
        self.base_scores = copy.deepcopy(self.base_scores)
        if self.version < 1 or not self.base_scores:
            raise ValueError("nonempty ranking and positive snapshot version required")
        for entity, score in self.base_scores.items():
            if entity not in self.raw["entities"] or not math.isfinite(score) or score < 0:
                raise ValueError("invalid T0 score/entity")
        actual_hash = digest(self.raw)
        if self.immutable_hash and self.immutable_hash != actual_hash:
            raise ValueError("source evidence hash mismatch")
        self.immutable_hash = actual_hash
        for claim_id, claim in self.claims.items():
            claim.setdefault("claim_id", claim_id)
            claim.setdefault("active", True)
            claim.setdefault("state", "unresolved")
            claim.setdefault("evidence_refs", [])
            claim.setdefault("predicate", {"type": "unverifiable"})

    def assert_immutable(self):
        if digest(self.raw) != self.immutable_hash:
            raise ValueError("immutable source evidence mutated")

    def diagnostic_digest(self) -> str:
        return digest({"claims": self.claims, "challenge_counts": self.challenge_counts,
                       "external_clues": self.external_clues, "version": self.version,
                       "session_closed": self.session_closed, "operator_accepted_claim": self.operator_accepted_claim})

    def artifact(self) -> dict:
        self.assert_immutable()
        return {"schema_version": "claim-feedback-rmg-cp2-1.0", "raw": copy.deepcopy(self.raw),
                "immutable_hash": self.immutable_hash, "claims": copy.deepcopy(self.claims),
                "base_scores": self.base_scores.copy(), "version": self.version,
                "local_contribution_weights": copy.deepcopy(self.local_contribution_weights),
                "challenge_counts": self.challenge_counts.copy(), "feedback_items": copy.deepcopy(self.feedback_items),
                "validations": copy.deepcopy(self.validations), "external_clues": copy.deepcopy(self.external_clues),
                "receipts": copy.deepcopy(self.receipts), "submissions": self.submissions,
                "snapshots": copy.deepcopy(self.snapshots), "session_closed": self.session_closed,
                "operator_accepted_claim": self.operator_accepted_claim}

    @classmethod
    def restore(cls, artifact: dict) -> "FeedbackState":
        fields = {key: copy.deepcopy(value) for key, value in artifact.items() if key != "schema_version"}
        return cls(**fields)
