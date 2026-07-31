from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from .contracts import stable_id


@dataclass
class RCAMemoryGraph:
    incident_id: str
    windows: dict[str, dict[str, Any]] = field(default_factory=dict)
    immutable_hashes: dict[str, str] = field(default_factory=dict)
    relations: list[dict[str, Any]] = field(default_factory=list)
    hypotheses: dict[str, dict[str, Any]] = field(default_factory=dict)
    validation_results: list[dict[str, Any]] = field(default_factory=list)
    diagnostic_abstracts: dict[str, str] = field(default_factory=dict)
    synthesis_history: list[dict[str, Any]] = field(default_factory=list)

    def add_window(self, pack: dict[str, Any]) -> None:
        if pack["incident_id"] != self.incident_id:
            raise ValueError("Evidence Pack belongs to another incident")
        window_id = pack["window"]["window_id"]
        canonical = json.dumps(pack, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if window_id in self.windows:
            if self.immutable_hashes[window_id] != digest:
                raise ValueError(f"immutable Evidence Pack changed for {window_id}")
            return
        if self.windows:
            previous = max(
                self.windows,
                key=lambda key: self.windows[key]["window"]["activation_order"],
            )
            expected = self.windows[previous]["window"]["activation_order"] + 1
            if pack["window"]["activation_order"] != expected:
                raise ValueError("only the adjacent earlier window may be activated")
        self.windows[window_id] = copy.deepcopy(pack)
        self.immutable_hashes[window_id] = digest
        local_candidates = ", ".join(
            f"{item['entity_id']}#{item['rank']}"
            for item in pack["ranked_candidates"][:3]
        )
        unavailable = ", ".join(pack["quality_notes"]["unavailable_modalities"]) or "none"
        self.diagnostic_abstracts[window_id] = (
            f"local candidates: {local_candidates}; unavailable modalities: {unavailable}"
        )
        self._add_adjacent_relations(window_id)
        self._refresh_hypotheses()

    def _add_adjacent_relations(self, window_id: str) -> None:
        order = self.windows[window_id]["window"]["activation_order"]
        if order == 0:
            return
        newer_id = f"W{order - 1:02d}"
        if newer_id not in self.windows:
            raise ValueError("adjacent newer window is missing")
        older = self.windows[window_id]
        newer = self.windows[newer_id]
        older_entities = {item["entity_id"] for item in older["entity_evidence"]}
        newer_entities = {item["entity_id"] for item in newer["entity_evidence"]}
        for entity in sorted(older_entities & newer_entities):
            self.relations.append(
                {
                    "relation_id": stable_id(
                        "REL", self.incident_id, "entity-continuity", window_id, newer_id, entity
                    ),
                    "type": "entity-continuity",
                    "source_window": window_id,
                    "destination_window": newer_id,
                    "entity_id": entity,
                    "state": "validated",
                }
            )
        older_top = {
            item["entity_id"]: item["evidence_id"]
            for item in older["ranked_candidates"][:3]
        }
        newer_top = {
            item["entity_id"]: item["evidence_id"]
            for item in newer["ranked_candidates"][:3]
        }
        shared_top = sorted(set(older_top) & set(newer_top))
        if shared_top:
            self.relations.append(
                {
                    "relation_id": stable_id(
                        "REL",
                        self.incident_id,
                        "semantic-association",
                        window_id,
                        newer_id,
                        *shared_top,
                    ),
                    "type": "semantic-association",
                    "source_window": window_id,
                    "destination_window": newer_id,
                    "shared_candidate_entities": shared_top,
                    "supporting_evidence_ids": [
                        evidence_id
                        for entity in shared_top
                        for evidence_id in (older_top[entity], newer_top[entity])
                    ],
                    "strength": "weak",
                    "state": "hypothesized",
                    "final_support_eligible": False,
                }
            )
        older_chains = [
            chain
            for candidate in older["ranked_candidates"]
            for chain in candidate["candidate_chains"]
        ]
        newer_chains = [
            chain
            for candidate in newer["ranked_candidates"]
            for chain in candidate["candidate_chains"]
        ]
        for left in older_chains:
            for right in newer_chains:
                shared = sorted(set(left["entities"]) & set(right["entities"]))
                if not shared:
                    continue
                self.relations.append(
                    {
                        "relation_id": stable_id(
                            "REL",
                            self.incident_id,
                            "propagation-continuation",
                            left["evidence_id"],
                            right["evidence_id"],
                        ),
                        "type": "propagation-continuation",
                        "source_window": window_id,
                        "destination_window": newer_id,
                        "shared_entities": shared,
                        "supporting_evidence_ids": [
                            left["evidence_id"],
                            right["evidence_id"],
                        ],
                        "state": (
                            "validated"
                            if left["temporal_status"] == right["temporal_status"] == "consistent"
                            else "hypothesized"
                        ),
                    }
                )

    def _refresh_hypotheses(self) -> None:
        for window_id, pack in self.windows.items():
            for candidate in pack["ranked_candidates"]:
                entity = candidate["entity_id"]
                hypothesis_id = stable_id("H", self.incident_id, entity)
                hypothesis = self.hypotheses.setdefault(
                    hypothesis_id,
                    {
                        "hypothesis_id": hypothesis_id,
                        "entity_id": entity,
                        "fault": "unspecified",
                        "state": "ACTIVE",
                        "supporting_evidence_ids": [],
                        "contradicting_evidence_ids": [],
                        "window_scores": {},
                        "validation_verdicts": [],
                    },
                )
                hypothesis["window_scores"][window_id] = candidate[
                    "localization_probability"
                ]
                if candidate["evidence_id"] not in hypothesis["supporting_evidence_ids"]:
                    hypothesis["supporting_evidence_ids"].append(candidate["evidence_id"])

    def evidence_ids(self) -> set[str]:
        values: set[str] = set()
        for pack in self.windows.values():
            for item in pack["entity_evidence"]:
                values.add(item["evidence_id"])
            for candidate in pack["ranked_candidates"]:
                values.add(candidate["evidence_id"])
                for chain in candidate["candidate_chains"]:
                    values.add(chain["evidence_id"])
        return values

    def entity_record(self, window_id: str, entity: str) -> dict[str, Any] | None:
        for record in self.windows[window_id]["entity_evidence"]:
            if record["entity_id"] == entity:
                return record
        return None

    def deterministic_ranking(self, recency_decay: float) -> list[dict[str, Any]]:
        scores: dict[str, list[tuple[int, float, float, float]]] = {}
        supports: dict[str, list[str]] = {}
        for window_id, pack in self.windows.items():
            order = int(pack["window"]["activation_order"])
            for record in pack["entity_evidence"]:
                entity = record["entity_id"]
                scores.setdefault(entity, []).append(
                    (
                        order,
                        float(record["localization_probability"]),
                        float(record["role_probabilities"]["root"]),
                        float(record["anomaly_score"]),
                    )
                )
            for candidate in pack["ranked_candidates"]:
                supports.setdefault(candidate["entity_id"], []).append(candidate["evidence_id"])
        ranking = []
        for entity, values in scores.items():
            weights = [recency_decay**order for order, _, _, _ in values]
            probability = sum(w * value[1] for w, value in zip(weights, values)) / sum(weights)
            root_role = max(value[2] for value in values)
            anomaly = max(value[3] for value in values)
            peak = max(value[1] for value in values)
            score = 0.45 * peak + 0.30 * probability + 0.15 * root_role + 0.10 * anomaly
            ranking.append(
                {
                    "entity_id": entity,
                    "score": round(score, 8),
                    "peak_probability": round(peak, 8),
                    "mean_probability": round(probability, 8),
                    "max_root_role": round(root_role, 8),
                    "max_anomaly": round(anomaly, 8),
                    "supporting_evidence_ids": supports.get(entity, []),
                }
            )
        return sorted(ranking, key=lambda item: item["score"], reverse=True)

    def register_validation(self, hypothesis_id: str, result: dict[str, Any]) -> None:
        record = copy.deepcopy(result)
        record["hypothesis_id"] = hypothesis_id
        self.validation_results.append(record)
        hypothesis = self.hypotheses[hypothesis_id]
        hypothesis["validation_verdicts"].append(result["verdict"])
        if result["verdict"] == "CONTRADICTED":
            hypothesis["state"] = "SUPPRESSED"
        elif result["verdict"] == "SUPPORTED" and hypothesis["state"] == "SUPPRESSED":
            hypothesis["state"] = "ACTIVE"
        for field in ("supporting_evidence_ids", "contradicting_evidence_ids"):
            for evidence_id in result[field]:
                if evidence_id not in hypothesis[field]:
                    hypothesis[field].append(evidence_id)
        verified = set(result["verified_relations"])
        contradicted = set(result["contradicting_evidence_ids"])
        for relation in self.relations:
            if relation["relation_id"] in verified:
                relation["state"] = "validated"
            elif contradicted & set(relation.get("supporting_evidence_ids", [])):
                relation["state"] = "contradicted"

    def record_synthesis(
        self, phase: str, round_index: int, steward: dict[str, Any]
    ) -> None:
        record = {
            "phase": phase,
            "round": round_index,
            "selected_entity": steward["selected_entity"],
            "ranked_entities": list(steward["ranked_entities"]),
            "claims": copy.deepcopy(steward["claims"]),
            "limitations": list(steward["limitations"]),
        }
        self.synthesis_history.append(record)
        selected = steward["selected_entity"]
        for window_id, pack in self.windows.items():
            local_rank = next(
                (
                    item["rank"]
                    for item in pack["ranked_candidates"]
                    if item["entity_id"] == selected
                ),
                None,
            )
            self.diagnostic_abstracts[window_id] = (
                f"global selected={selected}; local_rank="
                f"{local_rank if local_rank is not None else 'outside-top5'}; "
                f"synthesis_phase={phase}; synthesis_round={round_index}"
            )

    def to_artifact(self) -> dict[str, Any]:
        return {
            "schema_version": "rca-memory-graph-1.0",
            "incident_id": self.incident_id,
            "windows": self.windows,
            "immutable_hashes": self.immutable_hashes,
            "relations": self.relations,
            "hypotheses": self.hypotheses,
            "validation_results": self.validation_results,
            "diagnostic_abstracts": self.diagnostic_abstracts,
            "synthesis_history": self.synthesis_history,
        }
