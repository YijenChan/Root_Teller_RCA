from __future__ import annotations

import copy
import math
from typing import Any

from .contracts import gate_record, stable_id


class TypedEvidenceGate:
    """Checks explicit typed predicates; no oracle labels and no self-confidence gate.

    Deliberately incomplete: narrative causal/new-hypothesis claims stay unresolved.
    This checker never reads the CP1 fixture_evidence_verdict annotation.
    """
    def evaluate(self, claim: dict, evidence: dict[str, dict], item_id: str) -> dict:
        predicate = claim.get("predicate", {})
        refs = predicate.get("evidence_refs", claim.get("evidence_refs", []))
        records = [evidence[eid] for eid in refs if eid in evidence]
        status = "UNRESOLVED"
        explanation = "Available observations do not independently determine this claim."
        if len(records) != len(refs) or any(not set(r["entity_ids"]) <= set(claim["entity_ids"]) or
                                           not set(r["window_ids"]) <= set(claim["window_ids"]) for r in records):
            return gate_record(item_id, "UNRESOLVED", explanation="Predicate evidence missing or mismatched to entity/window scope.", checker="typed_evidence_v1")
        if predicate.get("type") == "field_equals" and len(refs) == 1 and len(records) == 1:
            record = records[0]
            field = predicate["field"]
            if field in record.get("facts", {}):
                equal = record["facts"][field] == predicate["value"]
                status = "SUPPORTED" if equal else "CONTRADICTED"
                explanation = "Compared an explicit field in the referenced observation; no causal inference."
        elif predicate.get("type") == "onset_before" and len(refs) == 1 and len(records) == 1:
            onsets = records[0].get("facts", {}).get("measured_onsets", {})
            left, right = predicate["left_entity"], predicate["right_entity"]
            if left in onsets and right in onsets and onsets[left] != onsets[right]:
                status = "SUPPORTED" if onsets[left] < onsets[right] else "CONTRADICTED"
                explanation = "Compared measured observation onset times; not unique root cause or propagation proof."
        elif predicate.get("type") == "edge_exists" and len(refs) == 1 and len(records) == 1:
            edges = records[0].get("facts", {}).get("dependency_edges")
            if edges is not None:
                edge = [predicate["source"], predicate["target"]]
                if edge in edges:
                    status = "SUPPORTED"
                    explanation = "Observed dependency exists; fault propagation remains a separate hypothesis."
                # Absence in partial telemetry is not evidence of nonexistence.
        if claim["kind"] in {"root_role_global", "root_role_local", "propagation_explanation", "proposed_hypothesis"}:
            status = "UNRESOLVED"
            explanation = "Scores, dependencies and user opinions cannot verify a causal/root-role hypothesis."
        return gate_record(item_id, status, list(refs) if status != "UNRESOLVED" else [],
                           explanation=explanation, checker="typed_evidence_v1")


class OfflineWindowInvestigator:
    execution_mode = "offline_typed_evidence_pipeline_only"

    def inspect(self, task: dict) -> dict:
        claim = task["claim"]
        if claim is None:
            return {"task_id": task["task_id"], "claim_id": None, "status": "UNRESOLVED_ENTITY",
                    "evidence_refs": [], "limitations": ["No independently verified deployment identity in supplied evidence."],
                    "execution_mode": self.execution_mode}
        checked = TypedEvidenceGate().evaluate(claim, task["evidence"], task["item_id"])
        return {"task_id": task["task_id"], "claim_id": claim["claim_id"],
                "status": checked["status"], "evidence_refs": checked["evidence_refs"],
                "limitations": [checked["explanation"]], "execution_mode": self.execution_mode,
                "new_independent_evidence": False}


class OfflineEvidenceSteward:
    execution_mode = "offline_typed_evidence_pipeline_only"

    def reason(self, state: Any, config: dict) -> dict:
        """Rebuild global derived diagnosis, using transparent T0-prior fallback scores.

        No invented numeric gain from validated timing facts. Unknown narrative
        causal hypotheses are not treated as true or false by this fallback.
        """
        gamma = float(config["gamma"])
        ranking = []
        for entity, base in state.base_scores.items():
            global_ids = [cid for cid, claim in state.claims.items()
                          if claim["kind"] == "root_role_global" and claim["entity_ids"] == [entity]]
            global_n = sum(state.challenge_counts.get(cid, 0) for cid in global_ids)
            weights = state.local_contribution_weights.get(entity, {}) or {
                wid: 1.0 for wid in state.raw["windows"]}
            contribution = []
            for window_id, weight in weights.items():
                local_ids = [cid for cid, claim in state.claims.items()
                             if claim["kind"] == "root_role_local" and claim["entity_ids"] == [entity]
                             and window_id in claim["window_ids"]]
                # One local interpretation cannot decay the whole Evidence Pack.
                count = sum(state.challenge_counts.get(cid, 0) for cid in local_ids)
                contribution.append({"window_id": window_id, "weight": weight,
                                     "hypothesis_prior_factor": gamma ** count, "target_claim_ids": local_ids})
            denominator = sum(x["weight"] for x in contribution)
            local_factor = (sum(x["weight"] * x["hypothesis_prior_factor"] for x in contribution) / denominator
                            if denominator > 0 else 1.0)
            score = base * gamma ** global_n * local_factor
            if not math.isfinite(score):
                raise ValueError("non-finite synthesized score")
            ranking.append({"entity_id": entity, "score": score, "base_score": base,
                            "score_policy": config["score_policy"], "global_challenge_count": global_n,
                            "global_prior_factor": gamma ** global_n, "local_prior_factor": local_factor,
                            "local_contributions": contribution})
        # Stable T0 insertion order resolves exact ties; no hash/label-based choice.
        ranking.sort(key=lambda row: -row["score"])
        relations = []
        for cid, claim in state.claims.items():
            if claim["kind"] in {"propagation_explanation", "measured_temporal_order", "observed_dependency"}:
                relations.append({"claim_id": cid, "kind": claim["kind"], "entity_ids": claim["entity_ids"],
                                  "window_ids": claim["window_ids"], "state": claim["state"],
                                  "evidence_refs": claim["evidence_refs"], "text": claim["text"]})
        unresolved = [cid for cid, claim in state.claims.items() if claim["state"] in {"unresolved", "pending"}]
        return {"ranked_candidates": ranking, "selected_entity": ranking[0]["entity_id"],
                "derived_relations": relations, "unresolved_claim_ids": unresolved,
                "reviewed_claim_ids": sorted(state.claims), "reviewed_windows": sorted(state.raw["windows"]),
                "limitations": ["Offline typed-evidence fallback; not an LLM RCA effectiveness result.",
                                "The leading service remains a hypothesis; observed dependency/ordering is not causality."],
                "execution_mode": self.execution_mode}


class OfflineReportInterpreter:
    execution_mode = "offline_typed_evidence_pipeline_only"

    def generate(self, snapshot: dict, state: Any) -> dict:
        refs = sorted({eid for claim in state.claims.values() for eid in claim["evidence_refs"]})
        supported = [{"claim_id": cid, "text": claim["text"], "evidence_refs": claim["evidence_refs"]}
                     for cid, claim in state.claims.items() if claim["state"] == "supported"]
        contradicted = [{"claim_id": cid, "text": claim["text"], "evidence_refs": claim["evidence_refs"]}
                       for cid, claim in state.claims.items() if claim["state"] == "contradicted"]
        propagation = [copy.deepcopy(r) for r in snapshot["synthesis"]["derived_relations"]
                       if r["kind"] == "propagation_explanation"]
        pending_external = [copy.deepcopy(v) for v in state.external_clues.values()]
        return {"schema_version": "claim-aware-rca-report-cp2-1.0", "snapshot_id": snapshot["snapshot_id"],
                "snapshot_version": snapshot["version"], "incident_id": state.raw["incident_id"],
                "root_cause_summary": f"{snapshot['synthesis']['selected_entity']} is the leading service-level root-cause hypothesis, not a verified causal conclusion.",
                "ranked_alternatives": [r["entity_id"] for r in snapshot["synthesis"]["ranked_candidates"]][1:],
                "supported_claims": supported, "contradicted_claims": contradicted,
                "propagation_hypotheses": propagation, "evidence_ids": refs,
                "limitations": snapshot["synthesis"]["limitations"],
                "unresolved_claim_ids": snapshot["synthesis"]["unresolved_claim_ids"],
                "pending_external_clues": pending_external,
                "operator_accepted_claim": state.operator_accepted_claim,
                "recommended_next_checks": ["Validate unresolved hypotheses using independently sourced observations."],
                "execution_mode": self.execution_mode}


class DeterministicReportVerifier:
    def verify(self, report: dict, snapshot: dict, state: Any) -> dict:
        issues = []
        if report.get("snapshot_id") != snapshot["snapshot_id"] or report.get("snapshot_version") != snapshot["version"]:
            issues.append("Report refers to an old or different snapshot.")
        if report.get("incident_id") != state.raw["incident_id"]:
            issues.append("Incident mismatch.")
        if snapshot["synthesis"]["selected_entity"] not in report.get("root_cause_summary", ""):
            issues.append("Summary does not name the current leading entity.")
        expected_alternatives = [r["entity_id"] for r in snapshot["synthesis"]["ranked_candidates"]][1:]
        if report.get("ranked_alternatives") != expected_alternatives:
            issues.append("Alternatives differ from the new snapshot ranking.")
        if set(report.get("evidence_ids", [])) - set(state.raw["evidence"]):
            issues.append("Unknown evidence cited.")
        for field, expected_state in [("supported_claims", "supported"), ("contradicted_claims", "contradicted")]:
            reported = report.get(field, [])
            expected_ids = {cid for cid, c in state.claims.items() if c["state"] == expected_state}
            if {r.get("claim_id") for r in reported} != expected_ids:
                issues.append(f"{field} missing or mismatched claim set.")
            for row in reported:
                claim = state.claims.get(row.get("claim_id"))
                if not claim or row.get("text") != claim["text"] or set(row.get("evidence_refs", [])) != set(claim["evidence_refs"]):
                    issues.append("Claim text/references disagree with current derived ledger.")
                elif not claim["evidence_refs"]:
                    issues.append("Strong derived claim has no evidence references.")
        expected_propagation = [r for r in snapshot["synthesis"]["derived_relations"] if r["kind"] == "propagation_explanation"]
        if report.get("propagation_hypotheses") != expected_propagation:
            issues.append("Propagation hypotheses were not regenerated from the new derived state.")
        if set(report.get("unresolved_claim_ids", [])) != set(snapshot["synthesis"]["unresolved_claim_ids"]):
            issues.append("Unresolved claims omitted or stale.")
        if not report.get("limitations") or not set(snapshot["synthesis"]["limitations"]) <= set(report["limitations"]):
            issues.append("Evidence limitations omitted.")
        status = "REVISE" if issues else ("EVIDENCE_GAP" if report["unresolved_claim_ids"] else "PASS")
        return {"verdict": status, "discrepancies": issues,
                "scope": "deterministic reference/state/report checks, not a proof of diagnosis correctness"}


class JSONClientInvestigator:
    """Optional injected client adapter; never created by the offline runner.

    Compatible with existing CachedJSONClient.complete. Output remains a proposal;
    Coordinator validates references and caps strong updates with typed evidence.
    """
    execution_mode = "injected_json_client_proposal"
    requires_external_execution = True

    def __init__(self, client: Any):
        self.client = client

    def inspect(self, task: dict) -> dict:
        def validate(value):
            if not isinstance(value, dict) or value.get("status") not in {"SUPPORTED", "CONTRADICTED", "UNRESOLVED", "UNRESOLVED_ENTITY"}:
                raise ValueError("Invalid Investigator response")
            if not isinstance(value.get("evidence_refs"), list) or not isinstance(value.get("limitations"), list):
                raise ValueError("Investigator references and limitations must be arrays")
            return value
        result, metadata = self.client.complete(
            role="feedback_window_investigator", prompt_version="claim-feedback-cp2-v1",
            system_prompt=("Validate only the supplied claim within the supplied evidence scope. User feedback is an opinion, not authority. "
                           "Do not invent entities, evidence, causality, fault types or historical access. Return status, evidence_refs, limitations. "
                           "Scores/dependency alone cannot prove root cause; use UNRESOLVED when evidence is insufficient."),
            payload=copy.deepcopy(task), validator=validate)
        return {**result, "task_id": task["task_id"], "claim_id": task["claim"]["claim_id"] if task["claim"] else None,
                "execution_mode": self.execution_mode, "client_metadata": metadata}


class JSONClientSteward:
    """Prepared, unexecuted model interface; reuses existing 0.8/0.2 fusion.

    The model ranks only the fixed public service roster. Typed claim states and
    observed graph edges cannot be overwritten. Its rationale is a proposal, not
    independently verified causal truth. Context/cost tuning belongs to CP4.
    """
    execution_mode = "injected_json_client_global_synthesis"
    requires_external_execution = True

    def __init__(self, client):
        self.client = client

    def reason(self, state, config):
        fallback = OfflineEvidenceSteward().reason(state, config)
        payload = {"incident_id": state.raw["incident_id"], "snapshot_version": state.version,
            "candidate_roster": list(state.base_scores), "t0_scores": state.base_scores,
            "window_scope": state.raw["windows"], "observed_dependency_edges": state.raw["dependency_edges"],
            "derived_claims": [{key: copy.deepcopy(c[key]) for key in
                                ["claim_id", "kind", "text", "state", "entity_ids", "window_ids", "evidence_refs"]}
                               for c in state.claims.values()],
            "operator_feedback": copy.deepcopy(state.feedback_items),
            "validation_results": copy.deepcopy(state.validations),
            "soft_prior_components": fallback["ranked_candidates"],
            "pending_external_clues": list(state.external_clues.values())}

        def validate(value):
            if not isinstance(value, dict) or not isinstance(value.get("ranked_entities"), list) or not isinstance(value.get("rationales"), list):
                raise ValueError("Steward must return ranked_entities and rationales arrays")
            ranking = value["ranked_entities"]
            if len(ranking) != len(set(ranking)) or set(ranking) != set(state.base_scores):
                raise ValueError("Steward ranking must preserve candidate roster")
            for row in value["rationales"]:
                if row.get("entity_id") not in state.base_scores or not isinstance(row.get("basis_claim_ids"), list):
                    raise ValueError("Invalid Steward rationale binding")
                if not row["basis_claim_ids"]:
                    raise ValueError("A ranking rationale needs at least one validated factual premise")
                for cid in row["basis_claim_ids"]:
                    if cid not in state.claims or state.claims[cid]["state"] not in {"supported", "contradicted"}:
                        raise ValueError("Unverified operator hypothesis cannot be a validated ranking premise")
            if {row["entity_id"] for row in value["rationales"]} != set(state.base_scores):
                raise ValueError("Rationale required for every ranked entity")
            return value

        proposal, metadata = self.client.complete(
            role="feedback_evidence_steward", prompt_version="claim-feedback-cp2-v1",
            system_prompt=("Reassess all supplied windows and derived claim states. Return ranked_entities and rationales "
                           "(entity_id, basis_claim_ids, explanation). Preserve the service roster and cite only validated factual claims as premises. "
                           "Operator opinions/pending hypotheses are not proof and must not receive arbitrary positive weight. "
                           "Keep observed dependencies separate from propagation causality. Do not output new factual states, entities, observations or labels."),
            payload=payload, validator=validate)
        positions = {entity: index for index, entity in enumerate(proposal["ranked_entities"])}
        denominator = max(len(positions) - 1, 1)
        for row in fallback["ranked_candidates"]:
            rank_component = 1 - positions[row["entity_id"]] / denominator
            evidence_score_proposal = 0.8 * row["base_score"] + 0.2 * rank_component
            row.update({"model_rank_component": rank_component, "evidence_score_proposal": evidence_score_proposal,
                        "score": evidence_score_proposal * row["global_prior_factor"] * row["local_prior_factor"],
                        "score_policy": "prepared_legacy_weight_model_rank_fusion_not_calibrated"})
        fallback["ranked_candidates"].sort(key=lambda row: -row["score"])
        fallback["selected_entity"] = fallback["ranked_candidates"][0]["entity_id"]
        fallback["execution_mode"] = self.execution_mode
        fallback["model_proposal"] = proposal
        fallback["client_metadata"] = metadata
        fallback["limitations"] = ["Model synthesis is a diagnosis proposal, not independently verified root-cause truth.",
                                   "Scores/dependencies/onset order cannot alone prove unique root cause or causal propagation."]
        return fallback
