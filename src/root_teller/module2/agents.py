from __future__ import annotations

from typing import Any

from .config import Module2Config
from .contracts import (
    stable_id,
    validate_investigator_response,
    validate_steward_response,
)
from .llm import CachedJSONClient
from .rmg import RCAMemoryGraph


STEWARD_SYSTEM_PROMPT = """You are the Evidence Steward in a microservice RCA system.
You receive only typed structural evidence and must return one JSON object.
Rank service entities as root-cause hypotheses. Prefer repeated root-role evidence,
early/initiating anomalies, dependency-compatible propagation, and evidence that
survives across windows. When validation results are supplied, revise the global
hypothesis: retain supported claims, weaken explicitly contradicted claims, and keep
inconclusive gaps as limitations. Do not invent entities, evidence IDs, windows,
fault labels, timestamps, or telemetry. The selected entity must be in ranked_entities.
Return exactly: ranked_entities (array of service IDs), selected_entity (string),
claims (array of objects with claim, entity, window_ids, evidence_ids), and
limitations (array of strings). Scores are model evidence, not proof."""


INVESTIGATOR_SYSTEM_PROMPT = """You are a stateless Window Investigator.
Validate one targeted RCA claim using only the supplied Evidence Pack excerpts.
Do not infer ground truth and do not invent identifiers. Return exactly one JSON
object with verdict (SUPPORTED, CONTRADICTED, or INCONCLUSIVE),
supporting_evidence_ids, contradicting_evidence_ids, verified_relations, and
limitations. A high model score alone can support a candidate ranking claim but
cannot prove causality; contradictions require explicit typed evidence."""


class EvidenceSteward:
    def __init__(self, client: CachedJSONClient, config: Module2Config) -> None:
        self.client = client
        self.config = config

    def reason(self, rmg: RCAMemoryGraph) -> tuple[dict[str, Any], dict[str, Any]]:
        deterministic = rmg.deterministic_ranking(self.config.recency_decay)
        compact_windows = []
        for window_id, pack in sorted(
            rmg.windows.items(), key=lambda item: item[1]["window"]["activation_order"]
        ):
            compact_windows.append(
                {
                    "window_id": window_id,
                    "activation_order": pack["window"]["activation_order"],
                    "ranked_candidates": [
                        {
                            "entity_id": item["entity_id"],
                            "evidence_id": item["evidence_id"],
                            "localization_probability": item["localization_probability"],
                            "diagnostic_role": item["diagnostic_role"],
                            "root_role_probability": item["role_probabilities"]["root"],
                            "anomaly_score": item["anomaly_score"],
                            "onset_bin": item["onset_bin"],
                            "chains": item["candidate_chains"],
                        }
                        for item in pack["ranked_candidates"]
                    ],
                    "quality_notes": pack["quality_notes"],
                }
            )
        payload = {
            "incident_id": rmg.incident_id,
            "activated_windows": compact_windows,
            "deterministic_candidate_ledger": deterministic[: self.config.top_k],
            "validated_cross_window_relations": [
                relation
                for relation in rmg.relations
                if relation["type"] == "propagation-continuation"
            ][:20],
            "prior_validation_results": rmg.validation_results[-10:],
            "hypothesis_states": [
                {
                    "hypothesis_id": value["hypothesis_id"],
                    "entity_id": value["entity_id"],
                    "state": value["state"],
                    "validation_verdicts": value["validation_verdicts"],
                    "supporting_evidence_ids": value["supporting_evidence_ids"][-5:],
                    "contradicting_evidence_ids": value[
                        "contradicting_evidence_ids"
                    ][-5:],
                }
                for value in rmg.hypotheses.values()
            ],
            "allowed_entities": [item["entity_id"] for item in deterministic],
            # Expose only identifiers that the compact prompt actually shows.
            # The full RMG may contain one entity-level ID per service/window;
            # sending that unreferenced ledger makes large systems (e.g.,
            # RE2-TT) exceed practical gateway payload limits without adding
            # evidence the Steward can reason over. Grounding below still
            # validates returned IDs against the complete immutable RMG.
            "allowed_evidence_ids": sorted(
                {
                    evidence_id
                    for window in compact_windows
                    for candidate in window["ranked_candidates"]
                    for evidence_id in (
                        [candidate["evidence_id"]]
                        + [
                            chain["evidence_id"]
                            for chain in candidate["chains"]
                        ]
                    )
                }
            ),
        }
        try:
            response, metadata = self.client.complete(
                role="evidence_steward",
                prompt_version=self.config.prompt_version,
                system_prompt=STEWARD_SYSTEM_PROMPT,
                payload=payload,
                validator=validate_steward_response,
            )
        except RuntimeError:
            response = self._fallback(deterministic)
            metadata = {"cache_key": None, "cached": False, "fallback": True}
        response = self._ground(response, rmg, deterministic)
        return response, metadata

    def _fallback(self, deterministic: list[dict[str, Any]]) -> dict[str, Any]:
        leader = deterministic[0]
        return {
            "ranked_entities": [item["entity_id"] for item in deterministic],
            "selected_entity": leader["entity_id"],
            "claims": [
                {
                    "claim": "The entity is the leading model-grounded root candidate.",
                    "entity": leader["entity_id"],
                    "window_ids": [],
                    "evidence_ids": leader["supporting_evidence_ids"][:3],
                }
            ],
            "limitations": ["deterministic fallback used"],
        }

    def _ground(
        self,
        response: dict[str, Any],
        rmg: RCAMemoryGraph,
        deterministic: list[dict[str, Any]],
    ) -> dict[str, Any]:
        allowed_entities = {item["entity_id"] for item in deterministic}
        allowed_evidence = rmg.evidence_ids()
        allowed_windows = set(rmg.windows)
        llm_entities = [
            item for item in response["ranked_entities"] if item in allowed_entities
        ]
        for item in deterministic:
            if item["entity_id"] not in llm_entities:
                llm_entities.append(item["entity_id"])
        llm_position = {entity: index for index, entity in enumerate(llm_entities)}
        max_position = max(len(llm_entities) - 1, 1)
        fused = []
        for item in deterministic:
            llm_score = 1.0 - llm_position[item["entity_id"]] / max_position
            fused_score = (
                self.config.deterministic_weight * item["score"]
                + self.config.steward_weight * llm_score
            )
            fused.append((item["entity_id"], fused_score))
        fused.sort(key=lambda item: item[1], reverse=True)
        claims = []
        for claim in response["claims"]:
            if claim["entity"] not in allowed_entities:
                continue
            grounded = dict(claim)
            grounded["window_ids"] = [
                item for item in claim["window_ids"] if item in allowed_windows
            ]
            grounded["evidence_ids"] = [
                item for item in claim["evidence_ids"] if item in allowed_evidence
            ]
            claims.append(grounded)
        if not claims:
            leader = fused[0][0]
            ledger = next(item for item in deterministic if item["entity_id"] == leader)
            claims.append(
                {
                    "claim": "Leading model-grounded root candidate.",
                    "entity": leader,
                    "window_ids": [],
                    "evidence_ids": ledger["supporting_evidence_ids"][:3],
                }
            )
        return {
            "ranked_entities": [entity for entity, _ in fused],
            "fused_scores": {entity: round(score, 8) for entity, score in fused},
            "selected_entity": fused[0][0],
            "claims": claims,
            "limitations": response["limitations"],
        }


class WindowInvestigator:
    def __init__(self, client: CachedJSONClient, config: Module2Config) -> None:
        self.client = client
        self.config = config

    def inspect(
        self,
        rmg: RCAMemoryGraph,
        hypothesis_id: str,
        claim: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        requested_windows = claim["window_ids"] or sorted(rmg.windows)[-2:]
        excerpts = []
        for window_id in requested_windows[:2]:
            if window_id not in rmg.windows:
                continue
            pack = rmg.windows[window_id]
            excerpts.append(
                {
                    "window": pack["window"],
                    "ranked_candidates": pack["ranked_candidates"],
                    "dependency_edges": pack["dependency_edges"],
                    "quality_notes": pack["quality_notes"],
                }
            )
        payload = {
            "incident_id": rmg.incident_id,
            "hypothesis_id": hypothesis_id,
            "claim": claim,
            "evidence_pack_excerpts": excerpts,
        }
        try:
            response, metadata = self.client.complete(
                role="window_investigator",
                prompt_version=self.config.prompt_version,
                system_prompt=INVESTIGATOR_SYSTEM_PROMPT,
                payload=payload,
                validator=validate_investigator_response,
            )
        except RuntimeError:
            response = self._fallback(claim, excerpts)
            metadata = {"cache_key": None, "cached": False, "fallback": True}
        allowed = rmg.evidence_ids()
        response["supporting_evidence_ids"] = [
            item for item in response["supporting_evidence_ids"] if item in allowed
        ]
        response["contradicting_evidence_ids"] = [
            item for item in response["contradicting_evidence_ids"] if item in allowed
        ]
        if response["verdict"] == "SUPPORTED" and not response["supporting_evidence_ids"]:
            response["verdict"] = "INCONCLUSIVE"
            response["limitations"].append("No valid supporting evidence ID returned.")
        return response, metadata

    @staticmethod
    def _fallback(claim: dict[str, Any], excerpts: list[dict[str, Any]]) -> dict[str, Any]:
        available = {
            candidate["evidence_id"]
            for excerpt in excerpts
            for candidate in excerpt["ranked_candidates"]
        }
        supporting = [item for item in claim["evidence_ids"] if item in available]
        return {
            "verdict": "SUPPORTED" if supporting else "INCONCLUSIVE",
            "supporting_evidence_ids": supporting,
            "contradicting_evidence_ids": [],
            "verified_relations": [],
            "limitations": ["deterministic fallback used"],
        }
