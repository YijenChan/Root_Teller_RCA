from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any

from root_teller.module2.contracts import stable_id


@dataclass
class FeedbackRMGOverlay:
    incident_id: str
    base_scores: dict[str, float]
    reject_counts: dict[str, int] = field(default_factory=dict)
    feedback_events: list[dict[str, Any]] = field(default_factory=list)
    confirmed_hypothesis_id: str | None = None

    def ranking(self, decay: float) -> list[dict[str, Any]]:
        ranked = []
        for entity, base_score in self.base_scores.items():
            count = self.reject_counts.get(entity, 0)
            adjusted = float(base_score) * math.pow(decay, count)
            ranked.append(
                {
                    "entity_id": entity,
                    "hypothesis_id": stable_id("H", self.incident_id, entity),
                    "base_score": float(base_score),
                    "reject_count": count,
                    "feedback_confidence": adjusted,
                }
            )
        return sorted(
            ranked,
            key=lambda item: (
                item["feedback_confidence"],
                item["base_score"],
                item["entity_id"],
            ),
            reverse=True,
        )

    def commit(
        self,
        *,
        entity: str,
        verdict: str,
        round_index: int,
    ) -> dict[str, Any]:
        verdict = verdict.upper()
        if verdict not in {"ACCEPT", "REJECT"}:
            raise ValueError(f"invalid feedback verdict: {verdict}")
        hypothesis_id = stable_id("H", self.incident_id, entity)
        feedback_id = stable_id(
            "FB", self.incident_id, hypothesis_id, round_index, verdict
        )
        if any(item["feedback_id"] == feedback_id for item in self.feedback_events):
            raise ValueError(f"duplicate feedback event: {feedback_id}")
        before = self.reject_counts.get(entity, 0)
        if verdict == "REJECT":
            self.reject_counts[entity] = before + 1
        else:
            self.confirmed_hypothesis_id = hypothesis_id
        event = {
            "feedback_id": feedback_id,
            "hypothesis_id": hypothesis_id,
            "entity_id": entity,
            "verdict": verdict,
            "round": round_index,
            "reject_count_before": before,
            "reject_count_after": self.reject_counts.get(entity, before),
            "structural_evidence_mutated": False,
        }
        self.feedback_events.append(event)
        return copy.deepcopy(event)

    def artifact(self, decay: float) -> dict[str, Any]:
        return {
            "schema_version": "feedback-rmg-overlay-1.0",
            "incident_id": self.incident_id,
            "feedback_score_decay": decay,
            "reject_counts": self.reject_counts,
            "feedback_events": self.feedback_events,
            "confirmed_hypothesis_id": self.confirmed_hypothesis_id,
            "current_ranking": self.ranking(decay),
            "immutable_structural_evidence": True,
        }


def final_steward_payload(case: dict[str, Any]) -> dict[str, Any]:
    cycles = case["hierarchical_rca_loop"]
    final_steward = cycles[-1]["steward_after_inspection"]
    return final_steward


def truthful_feedback_run(
    *,
    incident_id: str,
    target: str,
    base_scores: dict[str, float],
    decay: float,
    budget: int,
) -> dict[str, Any]:
    overlay = FeedbackRMGOverlay(incident_id, dict(base_scores))
    trace = []
    initial_ranking = overlay.ranking(decay)
    initial_top1 = initial_ranking[0]["entity_id"]
    if initial_top1 == target:
        return {
            "incident_id": incident_id,
            "initial_top1": initial_top1,
            "initial_correct": True,
            "success": True,
            "rejected_rounds_to_correct": 0,
            "final_top1": initial_top1,
            "trace": trace,
            "overlay": overlay.artifact(decay),
        }
    for round_index in range(1, budget + 1):
        before = overlay.ranking(decay)
        inspected = before[0]["entity_id"]
        event = overlay.commit(
            entity=inspected, verdict="REJECT", round_index=round_index
        )
        after = overlay.ranking(decay)
        trace.append(
            {
                "round": round_index,
                "ranking_before": [item["entity_id"] for item in before],
                "inspected_hypothesis_id": event["hypothesis_id"],
                "feedback_event": event,
                "ranking_after": [item["entity_id"] for item in after],
                "top1_after": after[0]["entity_id"],
            }
        )
        if after[0]["entity_id"] == target:
            overlay.commit(
                entity=target, verdict="ACCEPT", round_index=round_index + 1
            )
            return {
                "incident_id": incident_id,
                "initial_top1": initial_top1,
                "initial_correct": False,
                "success": True,
                "rejected_rounds_to_correct": round_index,
                "final_top1": target,
                "trace": trace,
                "overlay": overlay.artifact(decay),
            }
    final = overlay.ranking(decay)
    return {
        "incident_id": incident_id,
        "initial_top1": initial_top1,
        "initial_correct": False,
        "success": False,
        "rejected_rounds_to_correct": None,
        "final_top1": final[0]["entity_id"],
        "trace": trace,
        "overlay": overlay.artifact(decay),
    }


def false_feedback_run(
    *,
    incident_id: str,
    target: str,
    base_scores: dict[str, float],
    decay: float,
    false_rejection_budget: int,
    total_budget: int,
) -> dict[str, Any]:
    overlay = FeedbackRMGOverlay(incident_id, dict(base_scores))
    trace = []
    false_applied = 0
    for round_index in range(0, total_budget + 1):
        ranking = overlay.ranking(decay)
        current = ranking[0]["entity_id"]
        if current == target and false_applied >= false_rejection_budget:
            overlay.commit(
                entity=current, verdict="ACCEPT", round_index=round_index + 1
            )
            return {
                "incident_id": incident_id,
                "false_rejection_budget": false_rejection_budget,
                "false_rejections_applied": false_applied,
                "success": True,
                "feedback_rounds_to_accept": round_index,
                "capped_feedback_rounds": round_index,
                "final_top1": current,
                "trace": trace,
                "overlay": overlay.artifact(decay),
            }
        if round_index == total_budget:
            break
        is_false_rejection = current == target
        if is_false_rejection:
            false_applied += 1
        event = overlay.commit(
            entity=current, verdict="REJECT", round_index=round_index + 1
        )
        after = overlay.ranking(decay)
        trace.append(
            {
                "round": round_index + 1,
                "top1_before": current,
                "feedback_event": event,
                "feedback_was_false": is_false_rejection,
                "top1_after": after[0]["entity_id"],
            }
        )
    final = overlay.ranking(decay)
    return {
        "incident_id": incident_id,
        "false_rejection_budget": false_rejection_budget,
        "false_rejections_applied": false_applied,
        "success": False,
        "feedback_rounds_to_accept": None,
        "capped_feedback_rounds": total_budget,
        "final_top1": final[0]["entity_id"],
        "trace": trace,
        "overlay": overlay.artifact(decay),
    }
