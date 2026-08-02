from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from root_teller.module1.baseline import metrics

from .agents import EvidenceSteward, WindowInvestigator
from .config import Module2Config, Module2Paths
from .contracts import stable_id
from .llm import CachedJSONClient, load_api_settings
from .rmg import RCAMemoryGraph


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--case-limit", type=int)
    parser.add_argument("--case-id")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an interrupted run by loading completed case artifacts.",
    )
    parser.add_argument(
        "--protocol", choices=["default", "blind"], default="default"
    )
    return parser.parse_args()


def _load_pack(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _hypothesis_id(rmg: RCAMemoryGraph, entity: str) -> str:
    candidate = stable_id("H", rmg.incident_id, entity)
    if candidate not in rmg.hypotheses:
        rmg.hypotheses[candidate] = {
            "hypothesis_id": candidate,
            "entity_id": entity,
            "fault": "unspecified",
            "state": "ACTIVE",
            "supporting_evidence_ids": [],
            "contradicting_evidence_ids": [],
            "window_scores": {},
            "validation_verdicts": [],
        }
    return candidate


def _diagnosis_state(
    rmg: RCAMemoryGraph,
    steward: dict[str, Any],
    validation: dict[str, Any],
    config: Module2Config,
    total_windows: int,
) -> dict[str, Any]:
    selected = steward["selected_entity"]
    ranking = steward["ranked_entities"]
    scores = steward["fused_scores"]
    oldest = max(
        rmg.windows.values(), key=lambda pack: pack["window"]["activation_order"]
    )
    oldest_record = rmg.entity_record(oldest["window"]["window_id"], selected)
    history_exhausted = len(rmg.windows) >= total_windows
    normal_baseline = bool(
        oldest_record
        and oldest_record["anomaly_score"] < config.normal_anomaly_threshold
        and oldest_record["role_probabilities"]["normal"] >= 0.50
    )
    boundary_resolved = normal_baseline or history_exhausted
    allowed_evidence = rmg.evidence_ids()
    selected_claims = [
        claim for claim in steward["claims"] if claim["entity"] == selected
    ]
    grounded = bool(
        selected_claims
        and any(
            evidence_id in allowed_evidence
            for claim in selected_claims
            for evidence_id in claim["evidence_ids"]
        )
        and validation["verdict"] == "SUPPORTED"
    )
    complete = False
    propagation = []
    propagation_evidence_id = None
    for pack in rmg.windows.values():
        for candidate in pack["ranked_candidates"]:
            if candidate["entity_id"] != selected or not candidate["candidate_chains"]:
                continue
            chain = max(candidate["candidate_chains"], key=lambda item: item["score"])
            if not propagation or len(chain["entities"]) > len(propagation):
                propagation = chain["entities"]
                propagation_evidence_id = chain["evidence_id"]
                complete = chain["temporal_status"] in {"consistent", "uncertain"}
    leading_score = float(scores[selected])
    second_score = float(scores[ranking[1]]) if len(ranking) > 1 else 0.0
    disambiguated = (
        leading_score >= config.min_leading_score
        and leading_score - second_score >= config.ambiguity_margin
    )
    has_earlier_window = len(rmg.windows) < total_windows
    needs_earlier = (not boundary_resolved or not disambiguated) and has_earlier_window
    return {
        "schema_version": "diagnosis-state-1.0",
        "selected_entity": selected,
        "ranked_entities": ranking,
        "fused_scores": scores,
        "supporting_evidence_ids": sorted(
            {
                evidence_id
                for claim in selected_claims
                for evidence_id in claim["evidence_ids"]
                if evidence_id in allowed_evidence
            }
            | set(validation["supporting_evidence_ids"])
        ),
        "contradicting_evidence_ids": validation["contradicting_evidence_ids"],
        "propagation": propagation,
        "propagation_evidence_id": propagation_evidence_id,
        "grounded": grounded,
        "complete": complete,
        "boundary_resolved": boundary_resolved,
        "boundary_resolution": (
            "observed-normal-baseline"
            if normal_baseline
            else "history-exhausted"
            if history_exhausted
            else "unresolved"
        ),
        "disambiguated": disambiguated,
        "needs_earlier_evidence": needs_earlier,
        "has_earlier_window": has_earlier_window,
        "history_exhausted": history_exhausted,
        "limitations": steward["limitations"] + validation["limitations"],
    }


def _control(
    state: dict[str, Any], expansion_count: int, config: Module2Config
) -> tuple[str, str]:
    if (
        state["grounded"]
        and state["complete"]
        and state["boundary_resolved"]
        and state["disambiguated"]
    ):
        return "CONCLUDE", "all four deterministic stopping predicates passed"
    if (
        state["needs_earlier_evidence"]
        and state["has_earlier_window"]
        and expansion_count < config.max_expansions
    ):
        return "EXPAND", "past-resolvable boundary or ambiguity gap remains"
    return "ABSTAIN", "evidence predicates failed or expansion/history budget exhausted"


def _offline_steward(rmg: RCAMemoryGraph, config: Module2Config) -> dict[str, Any]:
    ranking = rmg.deterministic_ranking(config.recency_decay)
    return {
        "ranked_entities": [item["entity_id"] for item in ranking],
        "fused_scores": {item["entity_id"]: item["score"] for item in ranking},
        "selected_entity": ranking[0]["entity_id"],
        "claims": [
            {
                "claim": "Leading model-grounded root candidate.",
                "entity": ranking[0]["entity_id"],
                "window_ids": list(rmg.windows),
                "evidence_ids": ranking[0]["supporting_evidence_ids"][:3],
            }
        ],
        "limitations": ["offline deterministic mode"],
    }


def run_case_blind(
    case_root: Path,
    config: Module2Config,
    steward_agent: EvidenceSteward | None,
    investigator: WindowInvestigator | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    pack_files = sorted(case_root.glob("W*.json"))
    if not pack_files:
        raise ValueError(f"no window packs in {case_root}")
    first = _load_pack(pack_files[0])
    rmg = RCAMemoryGraph(first["incident_id"])
    rounds = []
    final_state: dict[str, Any] | None = None
    action = "ABSTAIN"
    for round_index, pack_file in enumerate(pack_files):
        pack = _load_pack(pack_file)
        rmg.add_window(pack)
        # New evidence comes from the frozen graph model.  Use the deterministic
        # Steward projection while the controller has an obvious earlier-window
        # gap, and invoke the remote roles only at a candidate stopping point.
        steward = _offline_steward(rmg, config)
        steward_meta = {"fallback": True, "offline": True, "phase": "expansion-gate"}
        hypothesis_id = _hypothesis_id(rmg, steward["selected_entity"])
        selected_claim = next(
            (
                claim
                for claim in steward["claims"]
                if claim["entity"] == steward["selected_entity"]
            ),
            steward["claims"][0],
        )
        validation = {
            "verdict": "SUPPORTED" if selected_claim["evidence_ids"] else "INCONCLUSIVE",
            "supporting_evidence_ids": selected_claim["evidence_ids"],
            "contradicting_evidence_ids": [],
            "verified_relations": [],
            "limitations": ["deterministic expansion gate"],
        }
        investigator_meta = {"fallback": True, "offline": True, "phase": "expansion-gate"}
        provisional_state = _diagnosis_state(
            rmg, steward, validation, config, total_windows=len(pack_files)
        )
        provisional_action, provisional_reason = _control(
            provisional_state, round_index, config
        )
        if provisional_action != "EXPAND" and steward_agent is not None:
            steward, steward_meta = steward_agent.reason(rmg)
            hypothesis_id = _hypothesis_id(rmg, steward["selected_entity"])
            selected_claim = next(
                (
                    claim
                    for claim in steward["claims"]
                    if claim["entity"] == steward["selected_entity"]
                ),
                steward["claims"][0],
            )
            assert investigator is not None
            validation, investigator_meta = investigator.inspect(
                rmg, hypothesis_id, selected_claim
            )
        rmg.register_validation(hypothesis_id, validation)
        final_state = _diagnosis_state(rmg, steward, validation, config, len(pack_files))
        action, reason = _control(final_state, round_index, config)
        if provisional_action == "EXPAND":
            reason = provisional_reason
        rounds.append(
            {
                "round": round_index,
                "activated_window": pack["window"]["window_id"],
                "steward": steward,
                "steward_call": steward_meta,
                "inspection_task": selected_claim,
                "investigator": validation,
                "investigator_call": investigator_meta,
                "diagnosis_state": final_state,
                "control_action": action,
                "control_reason": reason,
            }
        )
        if action != "EXPAND":
            break
    assert final_state is not None

    full_rmg = RCAMemoryGraph(first["incident_id"])
    for pack_file in pack_files:
        full_rmg.add_window(_load_pack(pack_file))
    if steward_agent is None:
        full_steward = _offline_steward(full_rmg, config)
        full_meta = {"fallback": True, "offline": True}
    else:
        full_steward, full_meta = steward_agent.reason(full_rmg)

    snapshot = {
        "schema_version": "rca-snapshot-1.0",
        "incident_id": rmg.incident_id,
        "status": action.lower(),
        "selected_hypothesis_id": _hypothesis_id(rmg, final_state["selected_entity"]),
        "root_cause": {
            "entity": final_state["selected_entity"],
            "fault": "unspecified",
        },
        "ranked_root_causes": final_state["ranked_entities"][: config.top_k],
        "propagation": final_state["propagation"],
        "supporting_evidence_ids": final_state["supporting_evidence_ids"],
        "contradicting_evidence_ids": final_state["contradicting_evidence_ids"],
        "limitations": final_state["limitations"],
        "activated_windows": len(rmg.windows),
        "available_windows": len(pack_files),
        "control_action": action,
    }
    result = {
        "incident_id": rmg.incident_id,
        "actual_stop": {
            "action": action,
            "ranking": final_state["ranked_entities"],
            "activated_windows": len(rmg.windows),
            "available_windows": len(pack_files),
            "diagnosis_state": final_state,
        },
        "full_history": {
            "ranking": full_steward["ranked_entities"],
            "steward_call": full_meta,
        },
        "rounds": rounds,
        "snapshot": snapshot,
    }
    return result, rmg.to_artifact()


def _inspection_claims(
    steward: dict[str, Any], limit: int
) -> list[dict[str, Any]]:
    selected = steward["selected_entity"]
    ordered = [
        claim for claim in steward["claims"] if claim["entity"] == selected
    ] + [claim for claim in steward["claims"] if claim["entity"] != selected]
    unique = []
    seen = set()
    for claim in ordered:
        identity = (
            claim["entity"],
            tuple(claim["window_ids"]),
            tuple(claim["evidence_ids"]),
        )
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(claim)
        if len(unique) >= limit:
            break
    return unique


def _offline_validation(
    claim: dict[str, Any],
) -> dict[str, Any]:
    return {
        "verdict": "SUPPORTED" if claim["evidence_ids"] else "INCONCLUSIVE",
        "supporting_evidence_ids": claim["evidence_ids"],
        "contradicting_evidence_ids": [],
        "verified_relations": [],
        "limitations": ["offline deterministic mode"],
    }


def run_case_default(
    case_root: Path,
    config: Module2Config,
    steward_agent: EvidenceSteward | None,
    investigator: WindowInvestigator | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Exhaust all windows, then execute the hierarchical RCA collaboration."""
    pack_files = sorted(case_root.glob("W*.json"))
    if not pack_files:
        raise ValueError(f"no window packs in {case_root}")
    first = _load_pack(pack_files[0])
    rmg = RCAMemoryGraph(first["incident_id"])
    ingestion_trace = []
    for pack_file in pack_files:
        pack = _load_pack(pack_file)
        rmg.add_window(pack)
        ingestion_trace.append(
            {
                "activation_order": pack["window"]["activation_order"],
                "window_id": pack["window"]["window_id"],
                "immutable_hash": rmg.immutable_hashes[pack["window"]["window_id"]],
                "window_count": len(rmg.windows),
                "hypothesis_count": len(rmg.hypotheses),
                "relation_count": len(rmg.relations),
            }
        )

    if steward_agent is None:
        current = _offline_steward(rmg, config)
        current_meta = {"fallback": True, "offline": True}
    else:
        current, current_meta = steward_agent.reason(rmg)
    rmg.record_synthesis("initial-global-reasoning", 0, current)
    cycles = []
    validated_entities: set[str] = set()
    for hierarchical_round in range(config.max_hierarchical_rounds):
        tasks = _inspection_claims(current, config.max_validation_tasks)
        inspections = []
        for task in tasks:
            hypothesis_id = _hypothesis_id(rmg, task["entity"])
            if investigator is None:
                validation = _offline_validation(task)
                call_meta = {"fallback": True, "offline": True}
            else:
                validation, call_meta = investigator.inspect(rmg, hypothesis_id, task)
            rmg.register_validation(hypothesis_id, validation)
            validated_entities.add(task["entity"])
            inspections.append(
                {
                    "task": task,
                    "hypothesis_id": hypothesis_id,
                    "result": validation,
                    "investigator_call": call_meta,
                }
            )
        before = current
        before_meta = current_meta
        if steward_agent is None:
            revised = _offline_steward(rmg, config)
            revised_meta = {"fallback": True, "offline": True}
        else:
            revised, revised_meta = steward_agent.reason(rmg)
        rmg.record_synthesis(
            "post-investigator-global-synthesis", hierarchical_round, revised
        )
        cycles.append(
            {
                "hierarchical_round": hierarchical_round,
                "steward_before_inspection": before,
                "steward_before_call": before_meta,
                "inspection_tasks": inspections,
                "steward_after_inspection": revised,
                "steward_after_call": revised_meta,
                "selected_entity_changed": (
                    before["selected_entity"] != revised["selected_entity"]
                ),
            }
        )
        current, current_meta = revised, revised_meta
        selected_supported = any(
            item["task"]["entity"] == current["selected_entity"]
            and item["result"]["verdict"] == "SUPPORTED"
            for item in inspections
        )
        if not cycles[-1]["selected_entity_changed"] and selected_supported:
            break

    # If synthesis selected an uninspected competitor at the round budget,
    # validate it once and synthesize a final bounded revision.
    if current["selected_entity"] not in validated_entities:
        tasks = [
            claim
            for claim in current["claims"]
            if claim["entity"] == current["selected_entity"]
        ][:1]
        if tasks:
            task = tasks[0]
            hypothesis_id = _hypothesis_id(rmg, task["entity"])
            if investigator is None:
                validation = _offline_validation(task)
                call_meta = {"fallback": True, "offline": True}
            else:
                validation, call_meta = investigator.inspect(rmg, hypothesis_id, task)
            rmg.register_validation(hypothesis_id, validation)
            if steward_agent is None:
                revised = _offline_steward(rmg, config)
                revised_meta = {"fallback": True, "offline": True}
            else:
                revised, revised_meta = steward_agent.reason(rmg)
            rmg.record_synthesis(
                "bounded-final-global-synthesis", len(cycles), revised
            )
            cycles.append(
                {
                    "hierarchical_round": len(cycles),
                    "steward_before_inspection": current,
                    "steward_before_call": current_meta,
                    "inspection_tasks": [
                        {
                            "task": task,
                            "hypothesis_id": hypothesis_id,
                            "result": validation,
                            "investigator_call": call_meta,
                        }
                    ],
                    "steward_after_inspection": revised,
                    "steward_after_call": revised_meta,
                    "selected_entity_changed": (
                        current["selected_entity"] != revised["selected_entity"]
                    ),
                    "reason": "validate newly selected synthesis candidate",
                }
            )
            current, current_meta = revised, revised_meta

    selected = current["selected_entity"]
    selected_claims = [
        claim for claim in current["claims"] if claim["entity"] == selected
    ]
    supporting = sorted(
        {
            evidence_id
            for claim in selected_claims
            for evidence_id in claim["evidence_ids"]
            if evidence_id in rmg.evidence_ids()
        }
    )
    selected_hypothesis = rmg.hypotheses[_hypothesis_id(rmg, selected)]
    contradicting = list(selected_hypothesis["contradicting_evidence_ids"])
    unresolved_issues = [
        limitation
        for validation in rmg.validation_results
        if validation["verdict"] == "INCONCLUSIVE"
        for limitation in validation["limitations"]
    ]
    propagation = []
    for pack in rmg.windows.values():
        for candidate in pack["ranked_candidates"]:
            if candidate["entity_id"] != selected:
                continue
            for chain in candidate["candidate_chains"]:
                if len(chain["entities"]) > len(propagation):
                    propagation = chain["entities"]
    snapshot = {
        "schema_version": "rca-snapshot-2.0",
        "incident_id": rmg.incident_id,
        "status": "full-history-synthesized",
        "protocol": "default-exhaustive",
        "selected_hypothesis_id": _hypothesis_id(rmg, selected),
        "root_cause": {"entity": selected, "fault": "unspecified"},
        "ranked_root_causes": current["ranked_entities"][: config.top_k],
        "propagation": propagation,
        "supporting_evidence_ids": supporting,
        "contradicting_evidence_ids": contradicting,
        "limitations": current["limitations"],
        "unresolved_issues": unresolved_issues,
        "activated_windows": len(rmg.windows),
    }
    result = {
        "incident_id": rmg.incident_id,
        "protocol": "default-exhaustive",
        "default_exhaustive": {
            "ranking": current["ranked_entities"],
            "activated_windows": len(rmg.windows),
            "available_windows": len(pack_files),
            "hierarchical_rounds": len(cycles),
        },
        "window_ingestion_trace": ingestion_trace,
        "hierarchical_rca_loop": cycles,
        "snapshot": snapshot,
    }
    return result, rmg.to_artifact()


def run(
    split: str,
    run_id: str,
    case_limit: int | None = None,
    case_id: str | None = None,
    offline: bool = False,
    workers: int = 1,
    protocol: str = "default",
    resume: bool = False,
) -> dict[str, Any]:
    paths = Module2Paths()
    config = Module2Config()
    source_root = paths.window_pack_root / split
    case_roots = sorted(
        path for path in source_root.iterdir() if path.is_dir() and path.name.startswith("re2ob-")
    )
    if case_id:
        case_roots = [path for path in case_roots if path.name == case_id]
    if case_limit is not None:
        case_roots = case_roots[:case_limit]
    if not case_roots:
        raise ValueError("no cases selected")

    client = None
    steward_agent = None
    investigator = None
    if not offline:
        client = CachedJSONClient(
            load_api_settings(paths.api_config),
            paths.response_cache,
            config.model,
            config.temperature,
            config.request_timeout_seconds,
            config.max_retries,
        )
        steward_agent = EvidenceSteward(client, config)
        investigator = WindowInvestigator(client, config)

    output_root = paths.run_root / run_id
    if output_root.exists() and any(output_root.iterdir()) and not resume:
        raise ValueError(f"run directory is not empty: {output_root}")
    case_output_root = output_root / "cases"
    rmg_output_root = output_root / "rmg"
    case_output_root.mkdir(parents=True, exist_ok=True)
    rmg_output_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    case_results = []
    failures = []
    completed_case_ids: set[str] = set()
    total_selected_cases = len(case_roots)
    if resume:
        for completed_path in sorted(case_output_root.glob("*.json")):
            completed = json.loads(completed_path.read_text(encoding="utf-8"))
            case_results.append(completed)
            completed_case_ids.add(str(completed["incident_id"]))
        case_roots = [
            case_root
            for case_root in case_roots
            if case_root.name not in completed_case_ids
        ]
    def execute(case_root: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
        if protocol == "default":
            result, rmg = run_case_default(
                case_root, config, steward_agent, investigator
            )
        else:
            result, rmg = run_case_blind(
                case_root, config, steward_agent, investigator
            )
        return case_root, result, rmg

    def save_completed(
        index: int, case_root: Path, result: dict[str, Any], rmg: dict[str, Any]
    ) -> None:
        case_results.append(result)
        (case_output_root / f"{case_root.name}.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        (rmg_output_root / f"{case_root.name}.json").write_text(
            json.dumps(rmg, indent=2) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                    {
                        "progress": (
                            f"{len(completed_case_ids) + index}/"
                            f"{len(completed_case_ids) + len(case_roots)}"
                        ),
                        "case_id": case_root.name,
                        "protocol": protocol,
                        "action": (
                            result["actual_stop"]["action"]
                            if protocol == "blind"
                            else "FULL_SCAN"
                        ),
                        "windows": (
                            result["actual_stop"]["activated_windows"]
                            if protocol == "blind"
                            else result["default_exhaustive"]["activated_windows"]
                        ),
                }
            ),
            flush=True,
        )

    if workers <= 1:
        scheduled = [(case_root, None) for case_root in case_roots]
        for index, (case_root, _) in enumerate(scheduled, start=1):
            try:
                _, result, rmg = execute(case_root)
                save_completed(index, case_root, result, rmg)
            except Exception as error:
                failures.append(
                    {
                        "incident_id": case_root.name,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(execute, case_root): case_root for case_root in case_roots}
            for index, future in enumerate(as_completed(futures), start=1):
                case_root = futures[future]
                try:
                    _, result, rmg = future.result()
                    save_completed(index, case_root, result, rmg)
                except Exception as error:
                    failures.append(
                        {
                            "incident_id": case_root.name,
                            "error_type": type(error).__name__,
                            "error": str(error),
                        }
                    )
    case_results.sort(key=lambda item: item["incident_id"])
    private_labels = json.loads(
        (
            paths.project
            / "cache"
            / "module2_re2ob"
            / "private_evaluator"
            / f"{split}_labels.json"
        ).read_text(encoding="utf-8")
    )
    actual_ranks = []
    full_ranks = []
    evaluated = []
    for result in case_results:
        target = private_labels[result["incident_id"]]["root_cause_service"]
        if protocol == "default":
            rank = result["default_exhaustive"]["ranking"].index(target) + 1
            actual_ranks.append(rank)
            evaluated.append(
                {
                    "incident_id": result["incident_id"],
                    "target": target,
                    "default_exhaustive_rank": rank,
                }
            )
        else:
            actual = result["actual_stop"]["ranking"].index(target) + 1
            full = result["full_history"]["ranking"].index(target) + 1
            actual_ranks.append(actual)
            full_ranks.append(full)
            evaluated.append(
                {
                    "incident_id": result["incident_id"],
                    "target": target,
                    "actual_stop_rank": actual,
                    "full_history_rank": full,
                }
            )
    summary = {
        "schema_version": "module2-run-summary-2.0",
        "run_id": run_id,
        "split": split,
        "offline": offline,
        "workers": workers,
        "protocol": protocol,
        "config": config.to_dict(),
        "config_sha256": hashlib.sha256(
            json.dumps(config.to_dict(), sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "cases_requested": total_selected_cases,
        "cases_completed": len(case_results),
        "cases_resumed": len(completed_case_ids),
        "failures": failures,
        "default_exhaustive_metrics": (
            metrics(actual_ranks) if protocol == "default" and actual_ranks else {}
        ),
        "actual_stop_metrics": (
            metrics(actual_ranks) if protocol == "blind" and actual_ranks else {}
        ),
        "full_history_metrics": (
            metrics(full_ranks) if protocol == "blind" and full_ranks else {}
        ),
        "abstention_rate": (
            sum(result["actual_stop"]["action"] == "ABSTAIN" for result in case_results)
            / len(case_results)
            if case_results and protocol == "blind"
            else None
        ),
        "mean_activated_windows": (
            sum(
                (
                    result["actual_stop"]["activated_windows"]
                    if protocol == "blind"
                    else result["default_exhaustive"]["activated_windows"]
                )
                for result in case_results
            )
            / len(case_results)
            if case_results
            else None
        ),
        "mean_window_saving": (
            sum(
                1
                - (
                    result["actual_stop"]["activated_windows"]
                    / result["actual_stop"]["available_windows"]
                    if protocol == "blind"
                    else result["default_exhaustive"]["activated_windows"]
                    / result["default_exhaustive"]["available_windows"]
                )
                for result in case_results
            )
            / len(case_results)
            if case_results
            else None
        ),
        "llm_stats": client.stats if client else {},
        "elapsed_seconds": round(time.time() - started, 3),
    }
    (output_root / "evaluation_private.json").write_text(
        json.dumps(evaluated, indent=2) + "\n", encoding="utf-8"
    )
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (output_root / "config.json").write_text(
        json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit(1)
    return summary


def main() -> None:
    args = _args()
    run(
        args.split,
        args.run_id,
        args.case_limit,
        args.case_id,
        args.offline,
        args.workers,
        args.protocol,
        args.resume,
    )


if __name__ == "__main__":
    main()
