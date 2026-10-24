from __future__ import annotations

import copy
import math
import threading
from datetime import datetime, timezone

from .agents import (DeterministicReportVerifier, OfflineEvidenceSteward,
                     OfflineReportInterpreter, OfflineWindowInvestigator, TypedEvidenceGate)
from .contracts import (EVIDENCE_ENCODING, STANCE, digest, format_errors,
                        gate_record, new_claim_key, stable_id)
from .state import FeedbackState


class FeedbackCoordinator:
    def __init__(self, state: FeedbackState, config: dict, investigator=None, steward=None, interpreter=None, verifier=None):
        self.state = state
        self.config = copy.deepcopy(config)
        if config["feedback_forgetting_rho"] != 1 or not 0 < config["gamma"] <= 1:
            raise ValueError("CP2 requires rho=1 and 0<gamma<=1")
        if config["max_tasks"] < 1 or config["max_windows_per_task"] < 1:
            raise ValueError("task budgets must be positive")
        if not 1 <= config["max_items"] <= 3:
            raise ValueError("item limit must respect the CP1 schema")
        self.investigator = investigator or OfflineWindowInvestigator()
        self.steward = steward or OfflineEvidenceSteward()
        self.interpreter = interpreter or OfflineReportInterpreter()
        self.verifier = verifier or DeterministicReportVerifier()
        self.gate = TypedEvidenceGate()
        self._lock = threading.RLock()
        state.assert_immutable()
        if not state.snapshots:
            # Preserve the common T0; never contact a model during construction.
            synthesis = OfflineEvidenceSteward().reason(copy.deepcopy(state), copy.deepcopy(self.config))
            self._check_synthesis(synthesis, state)
            initial_interpreter, initial_verifier = self.interpreter, self.verifier
            self.interpreter, self.verifier = OfflineReportInterpreter(), DeterministicReportVerifier()
            try:
                snapshot, report, verification = self._release(state, synthesis, "INITIAL")
            finally:
                self.interpreter, self.verifier = initial_interpreter, initial_verifier
            state.snapshots.append({**snapshot, "report": report, "verification": verification})
        else:
            previous = state.snapshots[-1]
            if previous["version"] != state.version or previous["derived_state_hash"] != state.diagnostic_digest():
                raise ValueError("Persisted snapshot and derived state disagree")
            self._check_synthesis(previous["synthesis"], state)
            if DeterministicReportVerifier().verify(previous["report"], previous, state)["verdict"] == "REVISE":
                raise ValueError("Persisted report differs from current snapshot")

    def _check_synthesis(self, synthesis, state):
        rows = synthesis["ranked_candidates"]
        entities = [r["entity_id"] for r in rows]
        if set(entities) != set(state.base_scores) or len(entities) != len(set(entities)):
            raise ValueError("Steward changed candidate roster or repeated an entity")
        if not entities or synthesis["selected_entity"] != entities[0]:
            raise ValueError("Steward selected entity differs from ranking")
        if any(not isinstance(r["score"], (float, int)) or not math.isfinite(r["score"]) or r["score"] < 0 for r in rows):
            raise ValueError("Steward invalid score")
        if set(synthesis["reviewed_claim_ids"]) != set(state.claims) or set(synthesis["reviewed_windows"]) != set(state.raw["windows"]):
            raise ValueError("Global synthesis must account for the full current derived RMG")
        # Derived factual state cannot be overwritten by a Steward proposal.
        for relation in synthesis["derived_relations"]:
            claim = state.claims[relation["claim_id"]]
            if any(relation[key] != claim[key] for key in ["state", "evidence_refs", "kind", "text", "entity_ids", "window_ids"]):
                raise ValueError("Steward attempted an unvalidated relation-state rewrite")
        expected_relations = {cid for cid, c in state.claims.items()
                              if c["kind"] in {"propagation_explanation", "measured_temporal_order", "observed_dependency"}}
        if {r["claim_id"] for r in synthesis["derived_relations"]} != expected_relations:
            raise ValueError("Steward omitted derived relations")
        expected_unresolved = {cid for cid, c in state.claims.items() if c["state"] in {"unresolved", "pending"}}
        if set(synthesis["unresolved_claim_ids"]) != expected_unresolved:
            raise ValueError("Steward omitted unresolved hypotheses")

    def _release(self, state, synthesis, feedback_id):
        snapshot = {"schema_version": "claim-feedback-snapshot-cp2-1.0", "incident_id": state.raw["incident_id"],
                    "version": state.version, "feedback_id": feedback_id, "immutable_source_hash": state.immutable_hash,
                    "derived_state_hash": state.diagnostic_digest(), "synthesis": copy.deepcopy(synthesis),
                    "claim_states": {cid: {"state": c["state"], "kind": c["kind"], "text": c["text"],
                                           "entity_ids": c["entity_ids"], "window_ids": c["window_ids"],
                                           "evidence_refs": c["evidence_refs"], "challenge_count": state.challenge_counts.get(cid, 0)}
                                     for cid, c in state.claims.items()},
                    "operator_accepted_claim": state.operator_accepted_claim, "session_closed": state.session_closed}
        snapshot["snapshot_id"] = stable_id("SNAP", snapshot)
        report = self.interpreter.generate(copy.deepcopy(snapshot), copy.deepcopy(state))
        verification = self.verifier.verify(report, snapshot, state)
        # A deterministic verifier is always enforced, even if a custom verifier passes.
        hard_verification = DeterministicReportVerifier().verify(report, snapshot, state)
        if verification["verdict"] == "REVISE" or hard_verification["verdict"] == "REVISE":
            raise ValueError("Report verification failed: " + str(verification.get("discrepancies", []) + hard_verification["discrepancies"]))
        state.assert_immutable()
        return snapshot, report, hard_verification

    def submit(self, event: dict) -> dict:
        # Single incident transaction lock; persistence is an explicit artifact step.
        with self._lock:
            return self._submit(copy.deepcopy(event))

    def _submit(self, event):
        self.state.assert_immutable()
        before_hash = self.state.diagnostic_digest()
        try:
            fingerprint = digest(event)
        except (ValueError, TypeError):
            return {"event_status": "INVALID_JSON", "new_interactions": 0, "committed": False}
        feedback_id = event.get("feedback_id") if isinstance(event, dict) else None
        previous = self.state.receipts.get(feedback_id) if isinstance(feedback_id, str) else None
        if previous:
            if previous["payload_hash"] != fingerprint:
                return {"event_status": "ID_CONFLICT", "new_interactions": 0, "committed": False,
                        "explanation": "Create a new feedback ID for a different payload; original receipt is retained."}
            response = copy.deepcopy(previous["response"])
            return {**response, "event_status": "IDEMPOTENT_REPLAY", "original_event_status": response["event_status"],
                    "gate": [], "review_windows": [], "investigator_windows": [], "soft_prior_targets": [],
                    "strong_state_changes": [], "tasks": [], "task_results": [], "state_diff": [],
                    "new_interactions": 0, "committed": False, "replayed_transaction": True,
                    "current_snapshot_version": self.state.version,
                    "pipeline_trace": [{"stage": "IDEMPOTENT_RETURN", "original_feedback_id": feedback_id,
                                        "original_snapshot_id": response.get("snapshot", {}).get("snapshot_id")}]}
        if self.state.submissions >= self.config["submission_budget"]:
            return {"event_status": "BUDGET_EXHAUSTED", "new_interactions": 0, "committed": False}
        response = {"feedback_id": feedback_id, "event_status": "PENDING", "vector_sparse": [], "gate": [],
                    "review_windows": [], "investigator_windows": [], "soft_prior_targets": [],
                    "strong_state_changes": [], "tasks": [], "task_results": [], "state_diff": [],
                    "new_interactions": 1, "end_session": False, "committed": False,
                    "before_version": self.state.version, "after_version": self.state.version,
                    "immutable_hash_before": self.state.immutable_hash, "diagnostic_hash_before": before_hash,
                    "pipeline_trace": [], "execution_mode": self.config["execution_mode"]}

        def finish(status):
            response["event_status"] = status
            self.state.submissions += 1
            self.state.assert_immutable()
            response["immutable_hash_after"] = self.state.immutable_hash
            response["diagnostic_hash_after"] = self.state.diagnostic_digest()
            response["submissions_total"] = self.state.submissions
            if isinstance(feedback_id, str) and feedback_id:
                self.state.receipts[feedback_id] = {"payload_hash": fingerprint, "response": copy.deepcopy(response)}
            return copy.deepcopy(response)

        errors = format_errors(event)
        if errors:
            response["errors"] = errors
            return finish("INVALID_FORMAT")
        response["pipeline_trace"].append({"stage": "FORMAT_CHECK", "status": "PASS"})
        if len(event["items"]) > self.config["max_items"]:
            return finish("ITEM_BUDGET_EXCEEDED")
        event_agents = [self.interpreter, self.verifier] + ([self.investigator, self.steward] if event["event_type"] == "comment" else [])
        if not self.config["external_API_authorized"] and any(getattr(agent, "requires_external_execution", False) for agent in event_agents):
            return finish("EXTERNAL_EXECUTION_NOT_AUTHORIZED")
        if event["incident_id"] != self.state.raw["incident_id"]:
            return finish("INCIDENT_MISMATCH")
        if not event["binding_confirmed"]:
            return finish("NEEDS_CONFIRMATION")
        if event["snapshot_version"] != self.state.version:
            return finish("STALE_SNAPSHOT")
        if self.state.session_closed:
            return finish("SESSION_CLOSED")
        if len({i["item_id"] for i in event["items"]}) != len(event["items"]):
            return finish("DUPLICATE_ITEM")
        working = copy.deepcopy(self.state)
        bindings = []
        seen_targets = set()
        errors = []
        for item in event["items"]:
            claim = None
            status = None
            scope = set(item["window_ids"])
            entity_scope = set(item["entity_ids"])
            if not scope or not scope <= set(working.raw["windows"]):
                status = "INVALID_SCOPE"
            elif not entity_scope <= set(working.raw["entities"]):
                status = "UNKNOWN_ENTITY_ID"
            elif item["target_kind"] == "existing_claim":
                claim = working.claims.get(item["target_claim_id"])
                if claim is None or not claim["active"]:
                    status = "INVALID_TARGET"
                elif entity_scope != set(claim["entity_ids"]) or scope != set(claim["window_ids"]):
                    status = "INVALID_SCOPE"
            elif item["target_kind"] == "new_hypothesis":
                if item["target_claim_id"] is not None or not entity_scope or not item["proposed_text"] or not item["proposed_text"].strip():
                    status = "INVALID_NEW_HYPOTHESIS"
                else:
                    axis = new_claim_key(event["incident_id"], item["entity_ids"], item["window_ids"], item["proposed_text"])
                    claim = working.claims.get(axis) or {"claim_id": axis, "kind": "proposed_hypothesis", "active": True,
                            "state": "pending", "text": item["proposed_text"], "entity_ids": sorted(entity_scope),
                            "window_ids": sorted(scope), "evidence_refs": [], "predicate": {"type": "unverifiable"}}
            else:
                if item["target_claim_id"] is not None or not item["entity_labels"] or not item["proposed_text"] or entity_scope:
                    status = "INVALID_EXTERNAL_CLUE"
            if status:
                errors.append(status)
                response["gate"].append(gate_record(item["item_id"], status))
                continue
            target_key = claim["claim_id"] if claim else (tuple(sorted(item["entity_labels"])), tuple(sorted(scope)))
            if target_key in seen_targets:
                errors.append("DUPLICATE_TARGET")
                response["gate"].append(gate_record(item["item_id"], "DUPLICATE_TARGET"))
                continue
            seen_targets.add(target_key)
            if claim:
                response["vector_sparse"].append({"axis": claim["claim_id"], "mask": 1, "stance_value": STANCE[item["stance"]]})
            if not set(item["evidence_refs"]) <= set(working.raw["evidence"]):
                errors.append("INVALID_REFERENCE")
                response["gate"].append(gate_record(item["item_id"], "INVALID_REFERENCE"))
                continue
            if any(not set(working.raw["evidence"][eid]["window_ids"]) & scope or
                   (entity_scope and not set(working.raw["evidence"][eid]["entity_ids"]) <= entity_scope)
                   for eid in item["evidence_refs"]):
                errors.append("REFERENCE_SCOPE_MISMATCH")
                response["gate"].append(gate_record(item["item_id"], "REFERENCE_SCOPE_MISMATCH"))
                continue
            gate = (self.gate.evaluate(claim, working.raw["evidence"], item["item_id"]) if claim else
                    gate_record(item["item_id"], "UNRESOLVED_ENTITY"))
            response["gate"].append(gate)
            bindings.append({"item": item, "claim": claim, "gate": gate})
        response["pipeline_trace"].append({"stage": "TARGET_BINDING_AND_EVIDENCE_GATE", "errors": errors})
        if errors:
            response["errors"] = errors
            return finish("NEEDS_CLARIFICATION")
        if event["event_type"] == "final_accept":
            b = bindings[0]
            leader = self.state.snapshots[-1]["synthesis"]["selected_entity"]
            if len(bindings) != 1 or not b["claim"] or b["claim"]["kind"] != "root_role_global" or b["item"]["stance"] != "support" or b["claim"]["entity_ids"] != [leader]:
                return finish("INVALID_FINAL_ACCEPT")

        try:
            for binding in bindings:
                item, claim, gate = binding["item"], binding["claim"], binding["gate"]
                if claim:
                    working.claims.setdefault(claim["claim_id"], copy.deepcopy(claim))
                    if gate["evidence_mask"]:
                        target = working.claims[claim["claim_id"]]
                        new_state = gate["status"].lower()
                        if target["state"] != new_state:
                            response["strong_state_changes"].append({"claim_id": target["claim_id"], "from": target["state"],
                                                                     "to": new_state, "evidence_refs": gate["evidence_refs"]})
                            target["state"] = new_state
                            target["evidence_refs"] = list(gate["evidence_refs"])
                    elif item["stance"] == "challenge" and claim["kind"] in {"root_role_global", "root_role_local"}:
                        cid = claim["claim_id"]
                        working.challenge_counts[cid] = working.challenge_counts.get(cid, 0) + 1
                        response["soft_prior_targets"].append(cid)
                else:
                    clue_id = stable_id("CLUE", working.raw["incident_id"], sorted(item["entity_labels"]), sorted(item["window_ids"]), item["proposed_text"])
                    working.external_clues[clue_id] = {"clue_id": clue_id, "entity_labels": item["entity_labels"],
                        "window_ids": item["window_ids"], "text": item["proposed_text"], "state": "pending_identity_resolution",
                        "source": "operator_reported", "last_feedback_id": feedback_id}
                working.feedback_items.append({"feedback_id": feedback_id, "snapshot_version": self.state.version,
                    "submitted_at": datetime.now(timezone.utc).isoformat(),
                    "item": copy.deepcopy(item), "target_claim_id": claim["claim_id"] if claim else None,
                    "operator_stance_value": STANCE[item["stance"]], "gate": copy.deepcopy(gate)})
                working.validations.append({"feedback_id": feedback_id, "claim_id": claim["claim_id"] if claim else None,
                                            "stage": "evidence_gate", **copy.deepcopy(gate)})
            response["review_windows"] = ([] if event["event_type"] == "final_accept" else
                                          sorted({w for b in bindings for w in b["item"]["window_ids"]}))
            tasks = [] if event["event_type"] == "final_accept" else self._plan(bindings, working, feedback_id)
            response["tasks"] = tasks
            response["investigator_windows"] = sorted({w for t in tasks for w in t["window_ids"]})
            response["pipeline_trace"].append({"stage": "TARGETED_TASK_PLANNING", "task_ids": [t["task_id"] for t in tasks],
                "selected_windows": response["investigator_windows"], "available_history_expanded": False})
            for task in tasks:
                proposal = self.investigator.inspect(copy.deepcopy(task))
                grounded = self._ground_finding(task, proposal)
                response["task_results"].append(grounded)
                working.validations.append({"feedback_id": feedback_id, "stage": "window_investigation", **copy.deepcopy(grounded)})
                # Only independently checked typed facts can change strong state.
                cid = grounded["claim_id"]
                if cid and grounded["status"] in {"SUPPORTED", "CONTRADICTED"}:
                    target = working.claims[cid]
                    if target["state"] != grounded["status"].lower():
                        response["strong_state_changes"].append({"claim_id": cid, "from": target["state"],
                            "to": grounded["status"].lower(), "evidence_refs": grounded["evidence_refs"]})
                        target["state"] = grounded["status"].lower()
                        target["evidence_refs"] = grounded["evidence_refs"]
            response["pipeline_trace"].append({"stage": "INVESTIGATOR_FINDINGS", "result_count": len(response["task_results"])})
            if event["event_type"] == "final_accept":
                working.session_closed = True
                working.operator_accepted_claim = bindings[0]["claim"]["claim_id"]
                response["end_session"] = True
            working.version += 1
            # Final acceptance applies to the diagnosis the operator inspected;
            # do not silently rerun the model and change that diagnosis.
            synthesis = (copy.deepcopy(self.state.snapshots[-1]["synthesis"]) if event["event_type"] == "final_accept" else
                         self.steward.reason(copy.deepcopy(working), copy.deepcopy(self.config)))
            self._check_synthesis(synthesis, working)
            response["pipeline_trace"].append({"stage": "REUSE_ACCEPTED_DIAGNOSIS" if event["event_type"] == "final_accept" else "GLOBAL_SYNTHESIS", "reviewed_claim_count": len(synthesis["reviewed_claim_ids"]),
                "reviewed_window_count": len(synthesis["reviewed_windows"]), "execution_mode": synthesis["execution_mode"],
                "score_policy": sorted({row["score_policy"] for row in synthesis["ranked_candidates"]})})
            snapshot, report, verification = self._release(working, synthesis, feedback_id)
            response.update({"snapshot": snapshot, "report": report, "verification": verification,
                             "after_version": working.version, "committed": True})
            response["pipeline_trace"].append({"stage": "NEW_SNAPSHOT_REPORT_VERIFY", "snapshot_id": snapshot["snapshot_id"],
                                               "verdict": verification["verdict"]})
            response["state_diff"] = self._diff(self.state, working)
            working.snapshots.append({**snapshot, "report": report, "verification": verification})
            working.assert_immutable()
            self.state = working
        except Exception as error:
            response["error"] = {"type": type(error).__name__, "message": str(error)}
            response["attempted_soft_prior_targets"] = response["soft_prior_targets"]
            response["attempted_strong_state_changes"] = response["strong_state_changes"]
            response["soft_prior_targets"] = []
            response["strong_state_changes"] = []
            response["end_session"] = False
            response["committed"] = False
            response["pipeline_trace"].append({"stage": "TRANSACTION_ROLLBACK", "status": "FAILED"})
            return finish("PROCESSING_FAILED")
        status = ("OPERATOR_ACCEPTED" if event["event_type"] == "final_accept" else
                  "PENDING_ENTITY_RESOLUTION" if any(b["claim"] is None for b in bindings) else
                  "ACCEPTED_AS_PENDING_HYPOTHESIS" if any(b["item"]["target_kind"] == "new_hypothesis" for b in bindings) else
                  "ACCEPTED_FOR_INVESTIGATION")
        return finish(status)

    def _plan(self, bindings, state, feedback_id):
        requests = []
        for b in bindings:
            item, claim, gate = b["item"], b["claim"], b["gate"]
            disagreement = bool(gate["evidence_mask"] and STANCE[item["stance"]] and STANCE[item["stance"]] != gate["evidence_value"])
            needs_task = gate["evidence_mask"] == 0 or disagreement or bool(item["requested_check"])
            if needs_task:
                requests.append((claim, item, True))
        # Root-role challenges also inspect a competing explanation, without
        # assigning that competitor a fabricated operator verdict.
        for b in bindings:
            claim, item = b["claim"], b["item"]
            if claim and claim["kind"] in {"root_role_global", "root_role_local"} and item["stance"] == "challenge":
                for row in state.snapshots[-1]["synthesis"]["ranked_candidates"]:
                    if row["entity_id"] in claim["entity_ids"]:
                        continue
                    competitors = [c for c in state.claims.values() if c["kind"] == "root_role_global" and c["entity_ids"] == [row["entity_id"]]]
                    if competitors:
                        competitor = competitors[0]
                        proxy = {**item, "window_ids": sorted(set(item["window_ids"]) & set(competitor["window_ids"]))}
                        if proxy["window_ids"]:
                            requests.append((competitor, proxy, False))
                        break
        tasks = []
        cap = self.config["max_windows_per_task"]
        ordered_requests = []
        for claim, item, direct in requests:
            covered = {w for result in state.validations if result.get("stage") == "window_investigation" and claim
                       and result.get("claim_id") == claim["claim_id"] for w in result.get("window_ids", [])}
            windows = sorted(item["window_ids"], key=lambda w: (w in covered, -float(state.raw["windows"][w].get("start_time", 0))))
            ordered_requests.append((claim, item, direct, windows))
        requests = ordered_requests
        # Round-robin pages avoid one 24-window target consuming every task slot
        # and starving the competing-hypothesis validation.
        schedule = [(claim, item, direct, windows[offset:offset + cap])
                    for offset in range(0, max((len(r[3]) for r in requests), default=0), cap)
                    for claim, item, direct, windows in requests if offset < len(windows)]
        for claim, item, direct, selected in schedule[:self.config["max_tasks"]]:
            anchors = set(claim["entity_ids"] if claim else [])
            entities = set(anchors)
            for edge in state.raw["dependency_edges"]:
                if set(edge) & anchors:
                    entities.update(edge)
            evidence = {eid: copy.deepcopy(record) for eid, record in state.raw["evidence"].items()
                        if set(record["window_ids"]) & set(selected) and
                        (not entities or set(record["entity_ids"]) & entities)}
            priority = claim.get("predicate", {}).get("evidence_refs", []) if claim else []
            ordered = sorted(evidence, key=lambda eid: (eid not in priority, eid))[:self.config["max_context_evidence_per_task"]]
            evidence = {eid: evidence[eid] for eid in ordered}
            context = [copy.deepcopy(c) for c in state.claims.values() if set(c["window_ids"]) & set(selected) and
                       (not entities or set(c["entity_ids"]) & entities)][:self.config["max_context_claims_per_task"]]
            tasks.append({"task_id": stable_id("TASK", feedback_id, len(tasks), claim["claim_id"] if claim else None, selected),
                "incident_id": state.raw["incident_id"], "snapshot_version": state.version,
                "item_id": item["item_id"], "claim": copy.deepcopy(claim), "window_ids": selected,
                "requested_check": item["requested_check"], "operator_reason": item["reason"],
                "operator_stance": item["stance"] if direct else None, "direct_feedback": direct,
                "context_claims": context, "evidence": evidence,
                "dependency_edges": [edge for edge in state.raw["dependency_edges"] if set(edge) <= entities],
                "history_access": "existing_authorized_windows_only"})
        return tasks

    def _ground_finding(self, task, proposal):
        if proposal.get("task_id") != task["task_id"] or proposal.get("claim_id") != (task["claim"]["claim_id"] if task["claim"] else None):
            raise ValueError("Investigator changed task/claim binding")
        if proposal.get("status") not in {*EVIDENCE_ENCODING, "UNRESOLVED_ENTITY"}:
            raise ValueError("Unknown Investigator verdict")
        refs = proposal.get("evidence_refs")
        if not isinstance(refs, list) or not set(refs) <= set(task["evidence"]):
            raise ValueError("Investigator returned evidence outside supplied task scope")
        checked = (self.gate.evaluate(task["claim"], task["evidence"], task["item_id"]) if task["claim"] else
                   gate_record(task["item_id"], "UNRESOLVED_ENTITY"))
        # An LLM unsupported verdict cannot bypass independent grounding.
        return {**copy.deepcopy(proposal), "proposed_status": proposal["status"], "status": checked["status"],
                "evidence_refs": checked["evidence_refs"], "grounding_checker": "typed_evidence_v1",
                "window_ids": list(task["window_ids"]),
                "new_independent_evidence": False, "grounding_override": proposal["status"] != checked["status"]}

    @staticmethod
    def _diff(before, after):
        changes = []
        for cid in sorted(set(before.claims) | set(after.claims)):
            if before.claims.get(cid) != after.claims.get(cid):
                changes.append({"path": f"claims/{cid}", "before": copy.deepcopy(before.claims.get(cid)), "after": copy.deepcopy(after.claims.get(cid))})
        for key in ["challenge_counts", "external_clues", "version", "session_closed", "operator_accepted_claim"]:
            if getattr(before, key) != getattr(after, key):
                changes.append({"path": key, "before": copy.deepcopy(getattr(before, key)), "after": copy.deepcopy(getattr(after, key))})
        return changes
