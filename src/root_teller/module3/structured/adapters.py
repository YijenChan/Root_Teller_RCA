from __future__ import annotations

import copy
import hashlib
import json

from .contracts import digest, stable_id
from .state import FeedbackState


def from_synthetic_ledger(ledger: dict) -> FeedbackState:
    """Explicit engineering world; never uses fixture_evidence_verdict labels.

    log_contains_timeout and queue_deviation encode the observations declared in
    the CP1 synthetic descriptions, not a learned model or real benchmark record.
    """
    windows = {w: {"start_time": r["start_seconds"], "end_time": r["end_seconds"]}
               for w, r in ledger["available_windows"].items()}
    evidence = {}
    for eid, row in ledger["evidence"].items():
        facts = {}
        if eid == "E01":
            facts = {"log_contains_timeout": True}
        elif eid == "E02":
            facts = {"queue_deviation": True}
        elif eid == "E03":
            facts = {"measured_onsets": {"S-C": row["checkout_onset_seconds"], "S-P": row["payment_timeout_onset_seconds"]}}
        elif eid == "E04":
            facts = {"dependency_edges": [["S-C", "S-P"]]}
        evidence[eid] = {"entity_ids": row["entity_ids"], "window_ids": row["window_ids"],
                         "kind": row["kind"], "description": row["description"], "facts": facts}
    raw = {"incident_id": ledger["incident_id"], "windows": windows, "entities": copy.deepcopy(ledger["canonical_entities"]),
           "evidence": evidence, "dependency_edges": [["S-C", "S-P"]],
           "provenance": {"mode": "synthetic_engineering_only", "ground_truth_available": False}}
    claims = {}
    for cid, row in ledger["claims"].items():
        predicate = {"type": "unverifiable"}
        if cid == "C01":
            predicate = {"type": "field_equals", "field": "log_contains_timeout", "value": True, "evidence_refs": ["E01"]}
        elif cid == "C03":
            predicate = {"type": "field_equals", "field": "queue_deviation", "value": True, "evidence_refs": ["E02"]}
        elif cid in ["C04", "C07"]:
            left, right = ("S-C", "S-P") if cid == "C04" else ("S-P", "S-C")
            predicate = {"type": "onset_before", "left_entity": left, "right_entity": right, "evidence_refs": ["E03"]}
        claims[cid] = {"claim_id": cid, "kind": row["kind"], "entity_ids": row["entity_ids"], "window_ids": row["window_ids"],
                       "text": row["text"], "state": row["initial_state"], "predicate": predicate,
                       "evidence_refs": list(predicate.get("evidence_refs", [])) if row["initial_state"] == "supported" else []}
    # Equal illustrative weights, not real scores or fixture root-cause labels.
    weights = {entity: {w: 1.0 for w in windows} for entity in raw["entities"]}
    return FeedbackState(raw, claims, {"S-P": 1.0, "S-C": 1.0}, local_contribution_weights=weights)


def from_legacy_artifacts(case: dict, rmg: dict, recency: float = 0.94) -> FeedbackState:
    """Whitelist public Evidence Pack fields; do not read any evaluator file.

    Model outputs are labelled as model observations, never as raw truth.
    T0 fused scores are preserved; old feedback trajectories are not imported.
    """
    if case["incident_id"] != rmg["incident_id"]:
        raise ValueError("Legacy case/RMG incident mismatch")
    raw_windows = copy.deepcopy(rmg["windows"])
    for wid, pack in raw_windows.items():
        canonical_old = json.dumps(pack, sort_keys=True, separators=(",", ":"))
        expected = rmg.get("immutable_hashes", {}).get(wid)
        actual = hashlib.sha256(canonical_old.encode("utf-8")).hexdigest()
        if expected and expected != actual:
            raise ValueError(f"Legacy Evidence Pack hash mismatch: {wid}")
    last = case["hierarchical_rca_loop"][-1]["steward_after_inspection"]
    ordering = last["ranked_entities"]
    scores = {entity: float(last["fused_scores"][entity]) for entity in ordering}
    windows, entities, evidence, claims, edges, weights = {}, {}, {}, {}, set(), {}
    all_windows = sorted(raw_windows)
    for wid, pack in raw_windows.items():
        windows[wid] = {key: pack["window"][key] for key in ["start_time", "end_time", "activation_order"]}
        for entity_record in pack["entity_evidence"]:
            entity = entity_record["entity_id"]
            entities[entity] = {"label": entity, "aliases": []}
            eid = entity_record["evidence_id"]
            facts = {key: copy.deepcopy(entity_record[key]) for key in
                     ["diagnostic_role", "localization_probability", "anomaly_score", "role_probabilities", "onset_bin"] if key in entity_record}
            evidence[eid] = {"kind": "cached_model_observation", "entity_ids": [entity], "window_ids": [wid], "facts": facts,
                             "source_pack_hash": rmg.get("immutable_hashes", {}).get(wid)}
            oid = stable_id("OBS", case["incident_id"], entity, wid, "model_diagnostic_role")
            claims[oid] = {"claim_id": oid, "kind": "model_observation", "entity_ids": [entity], "window_ids": [wid],
                "text": f"Cached Perception output reports diagnostic_role={facts['diagnostic_role']} for {entity} in {wid}; this is a model output, not verified fault causality.",
                "state": "supported", "evidence_refs": [eid],
                "predicate": {"type": "field_equals", "field": "diagnostic_role", "value": facts["diagnostic_role"], "evidence_refs": [eid]}}
            lid = stable_id("LOCAL", case["incident_id"], entity, wid)
            claims[lid] = {"claim_id": lid, "kind": "root_role_local", "entity_ids": [entity], "window_ids": [wid],
                          "text": f"{entity} is an initiating root-role hypothesis in {wid}.", "state": "unresolved", "evidence_refs": []}
            weights.setdefault(entity, {})[wid] = recency ** int(pack["window"]["activation_order"])
        for candidate in pack["ranked_candidates"]:
            eid = candidate["evidence_id"]
            facts = {key: copy.deepcopy(candidate[key]) for key in ["diagnostic_role", "localization_probability", "rank", "onset_bin"] if key in candidate}
            evidence[eid] = {"kind": "cached_model_candidate", "entity_ids": [candidate["entity_id"]], "window_ids": [wid], "facts": facts}
            for chain in candidate["candidate_chains"]:
                evidence[chain["evidence_id"]] = {"kind": "model_propagation_candidate", "entity_ids": chain["entities"], "window_ids": [wid],
                                                  "facts": {"temporal_status": chain["temporal_status"], "entities": chain["entities"]}}
                if len(chain["entities"]) > 1:
                    pid = stable_id("PROP", case["incident_id"], wid, chain["entities"])
                    claims[pid] = {"claim_id": pid, "kind": "propagation_explanation", "entity_ids": chain["entities"], "window_ids": [wid],
                                   "text": "Model-proposed propagation " + "->".join(chain["entities"]) + f" in {wid}; not verified causality.",
                                   "state": "unresolved", "evidence_refs": [chain["evidence_id"]]}
        for source, target in pack["dependency_edges"]:
            edges.add((source, target))
            eid = stable_id("EDGE", case["incident_id"], wid, source, target)
            evidence[eid] = {"kind": "observed_dependency", "entity_ids": [source, target], "window_ids": [wid],
                             "facts": {"dependency_edges": [[source, target]]}}
            did = stable_id("DEP", case["incident_id"], wid, source, target)
            claims[did] = {"claim_id": did, "kind": "observed_dependency", "entity_ids": [source, target], "window_ids": [wid],
                          "text": f"Dependency {source}->{target} occurs in the exported {wid} graph; not causal fault propagation.",
                          "state": "supported", "evidence_refs": [eid],
                          "predicate": {"type": "edge_exists", "source": source, "target": target, "evidence_refs": [eid]}}
    for entity in scores:
        cid = stable_id("ROOT", case["incident_id"], entity)
        claims[cid] = {"claim_id": cid, "kind": "root_role_global", "entity_ids": [entity], "window_ids": all_windows,
                       "text": f"{entity} is a global root-cause hypothesis for this incident.", "state": "unresolved", "evidence_refs": []}
    raw = {"incident_id": case["incident_id"], "windows": windows, "entities": entities, "evidence": evidence,
           "dependency_edges": [list(edge) for edge in sorted(edges)], "source_evidence_packs": raw_windows,
           "cross_window_relations": copy.deepcopy(rmg.get("relations",[])),
           "provenance": {"mode": "legacy_public_artifact_adaptation_smoke", "public_pack_hashes": rmg["immutable_hashes"],
                          "initial_public_snapshot_hash": digest(case["snapshot"]), "ground_truth_available": False}}
    return FeedbackState(raw, claims, scores, local_contribution_weights=weights)
