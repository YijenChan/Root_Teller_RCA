"""Portable claim-targeted feedback entry point; construction never calls a model."""
from __future__ import annotations

import copy
import json

from .structured.adapters import from_legacy_artifacts
from .structured.coordinator import FeedbackCoordinator
from .structured.state import FeedbackState
from .refinement.investigator import AuditedWindowInvestigator
from .refinement.reporting import AuditedReportInterpreter, AuditedReportVerifier
from .refinement.steward import AuditedSteward, OfflineContractClient


def feedback_config(*, external: bool = False, budget: int = 20) -> dict:
    return {
        "execution_mode": "model_feedback_refinement" if external else "offline_typed_evidence_pipeline_only",
        "score_policy": "fresh_synthesis_then_hypothesis_soft_prior",
        "gamma": 0.9, "feedback_forgetting_rho": 1.0, "evidence_window_recency": 0.94,
        "max_items": 3, "max_tasks": 2, "max_windows_per_task": 2,
        "max_context_claims_per_task": 24, "max_context_evidence_per_task": 160,
        "submission_budget": budget, "external_API_authorized": external,
    }


def refinement_config(*, model_rank_weight: float = 0.2) -> dict:
    return {
        "premise_selection": {"maximum_factual_premises": 64, "maximum_per_entity": 6,
            "maximum_hypotheses_per_entity": 4, "maximum_feedback_items": 3, "maximum_validations": 6},
        "scoring": {"policy_id": "declared_t0_model_rank_soft_prior_v2", "model_rank_weight": model_rank_weight},
    }


class SynthesisClient:
    """Reuse the portable cached client and validate every proposal locally."""
    requires_external_execution = True

    def __init__(self, client):
        self.client = client
        self.requires_external_execution = getattr(client, "requires_external_execution", True)

    def complete(self, *, response_schema=None, **kwargs):
        if response_schema is not None:
            kwargs["system_prompt"] += " Follow this JSON schema: " + json.dumps(response_schema, separators=(",", ":"))
        return self.client.complete(**kwargs)


def build_coordinator(state: FeedbackState, *, client=None, config=None,
                      model_rank_weight: float = 0.2, representation: str = "rmg",
                      delegate: bool = True) -> FeedbackCoordinator:
    """Build model refinement or a labelled offline typed-evidence path.

    Offline scores use zero model weight and may equal fixed downweighting for
    binary root-only feedback. No model effectiveness is simulated.
    """
    if representation not in {"rmg", "flat"}:
        raise ValueError("representation must be rmg or flat")
    core = feedback_config(external=client is not None)
    if config is not None:
        core.update(copy.deepcopy(config))
    if client is None:
        synthesis_client, investigator, weight = OfflineContractClient(), None, 0.0
    else:
        synthesis_client = SynthesisClient(client)
        investigator = AuditedWindowInvestigator(synthesis_client)
        weight = model_rank_weight
    steward = AuditedSteward(synthesis_client, refinement_config(model_rank_weight=weight), representation)
    if delegate:
        cls = FeedbackCoordinator
    else:
        from .refinement.baselines import StewardOnlyCoordinator
        cls = StewardOnlyCoordinator
    return cls(state, core, investigator=investigator, steward=steward,
               interpreter=AuditedReportInterpreter(), verifier=AuditedReportVerifier())


def state_from_diagnosis(case: dict, rmg: dict) -> FeedbackState:
    """Adapt full-range or blind diagnostics without acquiring hidden windows."""
    public_case = copy.deepcopy(case)
    if not public_case.get("hierarchical_rca_loop"):
        stop = public_case.get("actual_stop", {})
        diagnosis = copy.deepcopy(stop.get("diagnosis_state", {}))
        ranking = diagnosis.get("ranked_entities") or stop.get("ranking") or public_case.get("default_exhaustive", {}).get("ranking", [])
        if not ranking:
            raise ValueError("Diagnosis has no current ranking")
        diagnosis.setdefault("ranked_entities", ranking)
        diagnosis.setdefault("fused_scores", {e: 1 / (i + 1) for i, e in enumerate(ranking)})
        public_case["hierarchical_rca_loop"] = [{"steward_after_inspection": diagnosis}]
    public_case.setdefault("snapshot", {})
    return from_legacy_artifacts(public_case, rmg)
