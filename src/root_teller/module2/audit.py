from __future__ import annotations

import argparse
from collections import Counter
import json
import re
from pathlib import Path

from root_teller.module1.config import Paths
from root_teller.module1.features import load_case_specs

from .config import Module2Paths


def _metrics(ranks: list[int], denominator: int) -> dict[str, float]:
    if denominator == 0:
        return {}
    values = {
        f"A@{k}": sum(rank <= k for rank in ranks) / denominator for k in range(1, 6)
    }
    values["Avg@5"] = sum(values.values()) / 5
    return {key: round(value, 6) for key, value in values.items()}


def audit(run_id: str) -> dict[str, object]:
    paths = Module2Paths()
    run_root = paths.run_root / run_id
    evaluations = {
        item["incident_id"]: item
        for item in json.loads((run_root / "evaluation_private.json").read_text())
    }
    cases = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((run_root / "cases").glob("*.json"))
    ]
    protocol = cases[0].get("protocol", "blind") if cases else "unknown"
    actions: dict[str, int] = {}
    boundary: dict[str, int] = {}
    strict_ranks = []
    conclude_ranks = []
    early_concluded = 0
    premature_proxy = 0
    full_history_activated = 0
    stable_top1_full_expansion = 0
    fallback_counts = {
        "steward": 0,
        "investigator": 0,
        "full_history_steward": 0,
    }
    hierarchical_rounds = 0
    inspection_tasks = 0
    selected_entity_revisions = 0
    initial_steward_ranks: list[int] = []
    final_synthesis_ranks: list[int] = []
    ranking_changed_cases = 0
    target_rank_improved_cases = 0
    target_rank_worsened_cases = 0
    hierarchical_round_distribution: Counter[int] = Counter()
    inspection_verdict_distribution: Counter[str] = Counter()
    for case in cases:
        evaluation = evaluations[case["incident_id"]]
        if protocol == "default-exhaustive":
            rank = int(evaluation["default_exhaustive_rank"])
            strict_ranks.append(rank)
            target = evaluation["target"]
            initial_ranking = case["hierarchical_rca_loop"][0][
                "steward_before_inspection"
            ]["ranked_entities"]
            final_ranking = case["default_exhaustive"]["ranking"]
            initial_rank = initial_ranking.index(target) + 1
            final_rank = final_ranking.index(target) + 1
            initial_steward_ranks.append(initial_rank)
            final_synthesis_ranks.append(final_rank)
            ranking_changed_cases += int(initial_ranking != final_ranking)
            target_rank_improved_cases += int(final_rank < initial_rank)
            target_rank_worsened_cases += int(final_rank > initial_rank)
            hierarchical_rounds += len(case["hierarchical_rca_loop"])
            hierarchical_round_distribution[len(case["hierarchical_rca_loop"])] += 1
            for cycle in case["hierarchical_rca_loop"]:
                inspection_tasks += len(cycle["inspection_tasks"])
                selected_entity_revisions += int(cycle["selected_entity_changed"])
                if cycle["steward_before_call"].get("fallback") and not cycle[
                    "steward_before_call"
                ].get("offline"):
                    fallback_counts["steward"] += 1
                if cycle["steward_after_call"].get("fallback") and not cycle[
                    "steward_after_call"
                ].get("offline"):
                    fallback_counts["steward"] += 1
                for inspection in cycle["inspection_tasks"]:
                    inspection_verdict_distribution[
                        inspection["result"]["verdict"]
                    ] += 1
                    if inspection["investigator_call"].get("fallback") and not inspection[
                        "investigator_call"
                    ].get("offline"):
                        fallback_counts["investigator"] += 1
            continue
        rank = int(evaluation["actual_stop_rank"])
        action = case["actual_stop"]["action"]
        actions[action] = actions.get(action, 0) + 1
        boundary_key = case["actual_stop"]["diagnosis_state"]["boundary_resolution"]
        boundary[boundary_key] = boundary.get(boundary_key, 0) + 1
        strict_ranks.append(rank if action == "CONCLUDE" else 999)
        if action == "CONCLUDE":
            conclude_ranks.append(rank)
        activated = int(case["actual_stop"]["activated_windows"])
        if action == "CONCLUDE" and activated < 24:
            early_concluded += 1
            if evaluation["actual_stop_rank"] > evaluation["full_history_rank"]:
                premature_proxy += 1
        if activated == 24:
            full_history_activated += 1
            top_entities = [
                round_payload["diagnosis_state"]["selected_entity"]
                for round_payload in case["rounds"]
            ]
            if len(set(top_entities)) == 1:
                stable_top1_full_expansion += 1
        for round_payload in case["rounds"]:
            if round_payload["steward_call"].get("fallback") and not round_payload[
                "steward_call"
            ].get("offline"):
                fallback_counts["steward"] += 1
            if round_payload["investigator_call"].get("fallback") and not round_payload[
                "investigator_call"
            ].get("offline"):
                fallback_counts["investigator"] += 1
        if case["full_history"]["steward_call"].get("fallback"):
            fallback_counts["full_history_steward"] += 1

    raw_ids = [
        spec.incident_id
        for spec in load_case_specs(Paths())
        if spec.split == "test" and spec.eligible
    ]
    public_files = (
        list((paths.window_pack_root / "test").rglob("*.json"))
        + list((run_root / "cases").glob("*.json"))
        + list((run_root / "rmg").glob("*.json"))
    )
    raw_hits = []
    label_key_hits = []
    for path in public_files:
        text = path.read_text(encoding="utf-8")
        if any(raw_id in text for raw_id in raw_ids):
            raw_hits.append(str(path))
        if re.search(r'"(?:root_cause_service|fault_type|inject_time)"\s*:', text):
            label_key_hits.append(str(path))

    total = len(cases)
    result = {
        "schema_version": "module2-audit-1.0",
        "run_id": run_id,
        "protocol": protocol,
        "cases": total,
        "actions": actions,
        "boundary_resolution": boundary,
        "ranking_metrics_note": "Primary A@k scores evaluate the internal ranking even when action=ABSTAIN.",
        "strict_metrics_abstain_as_miss": _metrics(strict_ranks, total),
        "conditional_conclude_metrics": _metrics(conclude_ranks, len(conclude_ranks)),
        "premature_stop_proxy": {
            "definition": "early CONCLUDE whose target rank is worse than under full history",
            "early_concluded_cases": early_concluded,
            "count": premature_proxy,
            "rate_all_cases": round(premature_proxy / total, 6),
        },
        "over_expansion_proxy": {
            "definition": "all 24 windows activated while the selected Top-1 never changed",
            "full_history_activated_cases": full_history_activated,
            "stable_top1_cases": stable_top1_full_expansion,
            "rate_all_cases": round(stable_top1_full_expansion / total, 6),
        },
        "fallback_counts": fallback_counts,
        "leakage_audit": {
            "public_json_files_scanned": len(public_files),
            "raw_incident_id_hits": len(raw_hits),
            "private_label_key_hits": len(label_key_hits),
        },
    }
    if protocol == "default-exhaustive":
        result = {
            "schema_version": "module2-audit-2.0",
            "run_id": run_id,
            "protocol": protocol,
            "cases": total,
            "default_exhaustive_metrics": _metrics(strict_ranks, total),
            "all_windows_consumed": all(
                case["default_exhaustive"]["activated_windows"]
                == case["default_exhaustive"]["available_windows"]
                == 24
                for case in cases
            ),
            "mean_hierarchical_rounds": round(hierarchical_rounds / total, 6),
            "hierarchical_round_distribution": {
                str(key): value
                for key, value in sorted(hierarchical_round_distribution.items())
            },
            "total_inspection_tasks": inspection_tasks,
            "mean_inspection_tasks": round(inspection_tasks / total, 6),
            "inspection_verdict_distribution": dict(
                inspection_verdict_distribution
            ),
            "selected_entity_revisions": selected_entity_revisions,
            "collaboration_effect": {
                "initial_steward_metrics": _metrics(
                    initial_steward_ranks, total
                ),
                "post_investigator_synthesis_metrics": _metrics(
                    final_synthesis_ranks, total
                ),
                "ranking_changed_cases": ranking_changed_cases,
                "target_rank_improved_cases": target_rank_improved_cases,
                "target_rank_worsened_cases": target_rank_worsened_cases,
            },
            "fallback_counts": fallback_counts,
            "leakage_audit": {
                "public_json_files_scanned": len(public_files),
                "raw_incident_id_hits": len(raw_hits),
                "private_label_key_hits": len(label_key_hits),
            },
        }
    (run_root / "audit_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.run_id), indent=2))


if __name__ == "__main__":
    main()
