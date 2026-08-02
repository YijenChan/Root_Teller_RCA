from __future__ import annotations

from typing import Any

from root_teller.module2.llm import CachedJSONClient

from .config import Module3Config
from .contracts import validate_interpreter_response, validate_verifier_response


INTERPRETER_SYSTEM_PROMPT = """You are the RCA Report Interpreter.
Generate one concise structured JSON report using only the supplied RCA Snapshot
and canonical evidence ledger. Do not invent services, fault types, causal edges,
timestamps, or evidence IDs. The fault is unspecified unless the input explicitly
states otherwise. Preserve evidence limitations. Return exactly:
root_cause_summary (string), ranked_alternatives (array of service IDs),
propagation_chain (array of service IDs), evidence_ids (array), limitations (array),
unresolved_issues (array), and recommended_next_checks (array). If discrepancy
notes are supplied, revise only the affected fields."""


VERIFIER_SYSTEM_PROMPT = """You are the RCA Report Verifier.
Do not redo root-cause diagnosis. Check the supplied report against the canonical
ledger for root entity, alternative entities, propagation edges, evidence IDs,
contradictions, and limitations. Return exactly one JSON object with verdict
(PASS, REVISE, or EVIDENCE_GAP) and discrepancies (array of concise strings).
Use EVIDENCE_GAP only when the ledger itself lacks evidence needed for a report."""


def build_ledger(case: dict[str, Any], rmg: dict[str, Any]) -> dict[str, Any]:
    snapshot = case["snapshot"]
    evidence_ids: set[str] = set()
    evidence_records = []
    edges: set[tuple[str, str]] = set()
    for pack in rmg["windows"].values():
        edges.update(tuple(edge) for edge in pack["dependency_edges"])
        for record in pack["entity_evidence"]:
            evidence_ids.add(record["evidence_id"])
        for candidate in pack["ranked_candidates"]:
            evidence_ids.add(candidate["evidence_id"])
            if candidate["evidence_id"] in snapshot["supporting_evidence_ids"]:
                evidence_records.append(
                    {
                        "window_id": pack["window"]["window_id"],
                        "entity_id": candidate["entity_id"],
                        "evidence_id": candidate["evidence_id"],
                        "diagnostic_role": candidate["diagnostic_role"],
                        "localization_probability": candidate[
                            "localization_probability"
                        ],
                        "anomaly_score": candidate["anomaly_score"],
                    }
                )
            for chain in candidate["candidate_chains"]:
                evidence_ids.add(chain["evidence_id"])
    return {
        "incident_id": case["incident_id"],
        "selected_entity": snapshot["root_cause"]["entity"],
        "fault": snapshot["root_cause"]["fault"],
        "ranked_entities": case["default_exhaustive"]["ranking"],
        "propagation_chain": snapshot["propagation"],
        "allowed_entities": sorted(
            {
                entity
                for pack in rmg["windows"].values()
                for entity in (
                    item["entity_id"] for item in pack["entity_evidence"]
                )
            }
        ),
        "allowed_dependency_edges": [list(edge) for edge in sorted(edges)],
        "allowed_evidence_ids": sorted(evidence_ids),
        "referenced_evidence_records": evidence_records[:12],
        "required_limitations": snapshot["limitations"],
        "contradicting_evidence_ids": snapshot["contradicting_evidence_ids"],
        "unresolved_issues": snapshot["unresolved_issues"],
    }


def deterministic_discrepancies(
    report: dict[str, Any], ledger: dict[str, Any]
) -> list[str]:
    discrepancies = []
    selected = ledger["selected_entity"]
    if selected not in report["root_cause_summary"]:
        discrepancies.append("root_cause_summary does not name the selected entity")
    unknown_alternatives = sorted(
        set(report["ranked_alternatives"]) - set(ledger["allowed_entities"])
    )
    if unknown_alternatives:
        discrepancies.append(
            "unknown ranked alternatives: " + ", ".join(unknown_alternatives)
        )
    unknown_chain = sorted(
        set(report["propagation_chain"]) - set(ledger["allowed_entities"])
    )
    if unknown_chain:
        discrepancies.append(
            "unknown propagation entities: " + ", ".join(unknown_chain)
        )
    allowed_edges = {tuple(edge) for edge in ledger["allowed_dependency_edges"]}
    invalid_edges = [
        f"{source}->{target}"
        for source, target in zip(
            report["propagation_chain"][:-1], report["propagation_chain"][1:]
        )
        if (source, target) not in allowed_edges
    ]
    if invalid_edges:
        discrepancies.append(
            "unsupported propagation edges: " + ", ".join(invalid_edges)
        )
    invalid_evidence = sorted(
        set(report["evidence_ids"]) - set(ledger["allowed_evidence_ids"])
    )
    if invalid_evidence:
        discrepancies.append(
            "invalid evidence IDs: " + ", ".join(invalid_evidence[:5])
        )
    if not report["evidence_ids"]:
        discrepancies.append("report contains no evidence ID")
    if ledger["required_limitations"] and not report["limitations"]:
        discrepancies.append("report omits required evidence limitations")
    return discrepancies


def canonicalize_report(
    report: dict[str, Any], ledger: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """Selectively repair typed fields without changing the frozen diagnosis."""
    repaired = dict(report)
    repairs = []
    selected = ledger["selected_entity"]
    allowed_entities = set(ledger["allowed_entities"])
    allowed_evidence = set(ledger["allowed_evidence_ids"])
    allowed_edges = {tuple(edge) for edge in ledger["allowed_dependency_edges"]}
    if selected not in repaired["root_cause_summary"]:
        repaired["root_cause_summary"] = (
            f"{selected} is the leading service-level root-cause hypothesis; "
            f"the fault type remains {ledger['fault']}."
        )
        repairs.append("restored selected entity in root_cause_summary")
    alternatives = [
        entity
        for entity in repaired["ranked_alternatives"]
        if entity in allowed_entities and entity != selected
    ]
    if alternatives != repaired["ranked_alternatives"]:
        repairs.append("removed invalid or selected entities from ranked_alternatives")
    if not alternatives:
        alternatives = [
            entity
            for entity in ledger["ranked_entities"]
            if entity != selected
        ][:4]
        repairs.append("restored ranked alternatives from canonical ranking")
    repaired["ranked_alternatives"] = alternatives
    chain = repaired["propagation_chain"]
    chain_valid = (
        all(entity in allowed_entities for entity in chain)
        and all(
            (source, target) in allowed_edges
            for source, target in zip(chain[:-1], chain[1:])
        )
    )
    if not chain_valid:
        repaired["propagation_chain"] = ledger["propagation_chain"]
        repairs.append("restored canonical propagation chain")
    evidence = [
        evidence_id
        for evidence_id in repaired["evidence_ids"]
        if evidence_id in allowed_evidence
    ]
    if evidence != repaired["evidence_ids"]:
        repairs.append("removed invalid evidence IDs")
    if not evidence:
        evidence = [
            item["evidence_id"]
            for item in ledger["referenced_evidence_records"][:5]
        ]
        repairs.append("restored canonical evidence IDs")
    repaired["evidence_ids"] = evidence
    required_limitations = list(ledger["required_limitations"])
    for limitation in required_limitations:
        if limitation not in repaired["limitations"]:
            repaired["limitations"].append(limitation)
            repairs.append("restored required evidence limitation")
    for issue in ledger["unresolved_issues"]:
        if issue not in repaired["unresolved_issues"]:
            repaired["unresolved_issues"].append(issue)
            repairs.append("restored unresolved issue")
    if not repaired["recommended_next_checks"]:
        repaired["recommended_next_checks"] = [
            "Inspect the cited telemetry and dependency evidence before mitigation."
        ]
        repairs.append("restored advisory next check")
    return repaired, list(dict.fromkeys(repairs))


def _fallback_report(ledger: dict[str, Any]) -> dict[str, Any]:
    return {
        "root_cause_summary": (
            f"{ledger['selected_entity']} is the leading service-level root-cause "
            "hypothesis; the fault type remains unspecified."
        ),
        "ranked_alternatives": [
            entity
            for entity in ledger["ranked_entities"]
            if entity != ledger["selected_entity"]
        ][:4],
        "propagation_chain": ledger["propagation_chain"],
        "evidence_ids": [
            item["evidence_id"]
            for item in ledger["referenced_evidence_records"][:5]
        ],
        "limitations": ledger["required_limitations"],
        "unresolved_issues": ledger["unresolved_issues"],
        "recommended_next_checks": [
            "Inspect the cited telemetry and dependency evidence before mitigation."
        ],
    }


def generate_verified_report(
    *,
    case: dict[str, Any],
    rmg: dict[str, Any],
    client: CachedJSONClient | None,
    config: Module3Config,
) -> dict[str, Any]:
    ledger = build_ledger(case, rmg)
    discrepancies: list[str] = []
    rounds = []
    report: dict[str, Any] | None = None
    for round_index in range(1, config.verifier_budget + 1):
        if client is None:
            report = _fallback_report(ledger)
            interpreter_meta = {"fallback": True, "offline": True}
        else:
            try:
                report, interpreter_meta = client.complete(
                    role="rca_interpreter",
                    prompt_version=config.prompt_version,
                    system_prompt=INTERPRETER_SYSTEM_PROMPT,
                    payload={
                        "snapshot": case["snapshot"],
                        "evidence_ledger": {
                            key: value
                            for key, value in ledger.items()
                            if key != "allowed_evidence_ids"
                        },
                        "allowed_evidence_ids": ledger["allowed_evidence_ids"],
                        "revision_discrepancies": discrepancies,
                    },
                    validator=validate_interpreter_response,
                )
            except RuntimeError:
                report = _fallback_report(ledger)
                interpreter_meta = {"fallback": True, "offline": False}

        report, canonical_repairs = canonicalize_report(report, ledger)
        deterministic = deterministic_discrepancies(report, ledger)
        if client is None:
            verifier = {
                "verdict": "PASS" if not deterministic else "REVISE",
                "discrepancies": deterministic,
            }
            verifier_meta = {"fallback": True, "offline": True}
        else:
            try:
                verifier, verifier_meta = client.complete(
                    role="rca_verifier",
                    prompt_version=config.prompt_version,
                    system_prompt=VERIFIER_SYSTEM_PROMPT,
                    payload={
                        "report": report,
                        "canonical_ledger": ledger,
                    },
                    validator=validate_verifier_response,
                )
            except RuntimeError:
                verifier = {
                    "verdict": "PASS" if not deterministic else "REVISE",
                    "discrepancies": deterministic,
                }
                verifier_meta = {"fallback": True, "offline": False}
        combined = list(dict.fromkeys(deterministic))
        if not ledger["referenced_evidence_records"]:
            verdict = "EVIDENCE_GAP"
        elif deterministic:
            verdict = "REVISE"
        else:
            verdict = "PASS"
        rounds.append(
            {
                "round": round_index,
                "report": report,
                "interpreter_call": interpreter_meta,
                "canonical_repairs": canonical_repairs,
                "deterministic_discrepancies": deterministic,
                "verifier": {
                    "verdict": verdict,
                    "discrepancies": combined,
                    "llm_advisory_verdict": verifier["verdict"],
                    "llm_advisory_notes": verifier["discrepancies"],
                },
                "verifier_call": verifier_meta,
            }
        )
        if verdict in {"PASS", "EVIDENCE_GAP"}:
            return {
                "schema_version": "rca-report-result-1.0",
                "incident_id": case["incident_id"],
                "status": "verified" if verdict == "PASS" else "evidence-gap",
                "verifier_verdict": verdict,
                "report": report,
                "rounds": rounds,
            }
        discrepancies = combined
    assert report is not None
    return {
        "schema_version": "rca-report-result-1.0",
        "incident_id": case["incident_id"],
        "status": "unverified",
        "verifier_verdict": "REVISE",
        "report": report,
        "rounds": rounds,
        "remaining_discrepancies": discrepancies,
    }


def report_markdown(result: dict[str, Any]) -> str:
    report = result["report"]
    alternatives = "\n".join(f"- {item}" for item in report["ranked_alternatives"])
    evidence = "\n".join(f"- `{item}`" for item in report["evidence_ids"])
    limitations = "\n".join(f"- {item}" for item in report["limitations"]) or "- None recorded"
    unresolved = (
        "\n".join(f"- {item}" for item in report["unresolved_issues"])
        or "- None recorded"
    )
    checks = "\n".join(f"- {item}" for item in report["recommended_next_checks"])
    chain = " -> ".join(report["propagation_chain"]) or "(zero-hop/local root)"
    return f"""# RCA Report: {result['incident_id']}

Status: **{result['status']}**  
Verifier verdict: **{result['verifier_verdict']}**

## Root-cause summary

{report['root_cause_summary']}

## Ranked alternatives

{alternatives}

## Propagation chain

{chain}

## Evidence references

{evidence}

## Limitations

{limitations}

## Unresolved issues

{unresolved}

## Recommended next checks

{checks}
"""
