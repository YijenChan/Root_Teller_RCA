# Revision implementation notes

## Source provenance

The revised library was ported from the existing local development source:
the claim-feedback coordinator (checkpoint 2), evidence-audited refinement
(checkpoint 4), non-adjacent cross-window linking, and the two added baseline
reproductions. Personal drive paths, sibling-directory import tricks and
credential-file parsing are not required by the new packages.

Source evidence and historical experiment directories were not overwritten.
The fixed-score feedback overlay remains an explicitly labelled control.

## What changed

- Typed, persistent claim/hypothesis feedback with timestamped notes, target
  binding and separate operator final acceptance.
- Targeted Investigator tasks followed by fresh evidence-grounded Steward
  synthesis, a new Snapshot, regenerated report, and consistency validation.
- Bounded non-adjacent RMG context from activated windows only.
- Portable Top-k and candidate-downweighting controls, with exact acceptance
  counting and separate within-budget success rates.
- Portable RCACopilot/RCAgent adaptations, using explicit runtime model
  configuration and environment credentials.
- Local UI/API feedback routed through the new coordinator, including support,
  challenge and check-only actions.
- One-second interval preset and multi-interval RCA window export, while
  preserving historical 60-second checkpoint configurations.

## Empirical boundaries to resolve in the manuscript

1. Local added-baseline development records describe a GPT-5-nano adaptation,
   whereas the supplied manuscript names GPT-4/Vicuna native configurations.
   This release does not alter model names to disguise that difference. Confirm
   which actual run manifests support each table before submission.
2. The legacy released weights use 60-second feature aggregation. Supporting a
   one-second input contract in code does not retrain those weights or validate
   the manuscript's temporal-resolution claim.
3. New cross-window links and revised feedback prompts require appropriately
   versioned executions. Historical scores are not silently reassigned.
4. The illustrative interface trace, synthetic fixtures and offline fallback
   are not new human studies, production evidence or formal RQ4 measurements.

No paid model evaluation, training, dataset download, or human data collection
is performed by the alignment tests.

## Configuration and use

The new configuration is `configs/revision2_method.json`.
It declares the temporal contract, cross-window policy, evidence budgets,
feedback representation and score ordering. Historical frozen JSON files retain
their original meanings.

From Python:

```python
from root_teller.module3.session import state_from_diagnosis, build_coordinator
from root_teller.module3.structured.contracts import default_item, event_for

state = state_from_diagnosis(public_diagnosis, public_rmg)
coordinator = build_coordinator(state, client=configured_cached_json_client)
# Resolve the service/window to an actual claim ID; ambiguous inputs need clarification.
claim = state.claims[claim_id]
item = default_item(
    item_id="review-1", target_claim_id=claim_id,
    entity_ids=claim["entity_ids"], window_ids=claim["window_ids"],
    stance="challenge", reason="Operator assessment", requested_check="Check the retained evidence",
)
result = coordinator.submit(event_for(state, "unique-review-id", [item]))
```

Omit `client` for the explicit offline typed-evidence path. Its output is not
a model efficacy result. Final acceptance uses a separate `final_accept`
event targeting the current global root hypothesis.
