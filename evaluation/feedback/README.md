# Revised feedback evaluation

The executable entry point is `python -m root_teller.module3.evaluate_feedback`.
It consumes a frozen public FeedbackState manifest and a separate private
incident-to-service evaluator map. No ground-truth reasons are supplied to
diagnostic policies. Every method starts from a deep copy of the same T0.

```powershell
python -m root_teller.module3.evaluate_feedback --manifest public_states.json --labels private_labels.json --output runs/feedback-check --mode offline --K 0 1 3 5
```

The manifest is a JSON list of objects with `incident_id` and `state`
(the path to a serialized public FeedbackState). Relative paths resolve from the
manifest directory. The private label map is never passed to the coordinator.

Strategies: `full`, `flat`, `stateful_single_agent`, `top5`,
`downweighting`, `revisit`, and `no_feedback`.
Flat/stateful variants are explicit engineering controls, not retroactive
reproductions of paper ablation numbers. Every accepted inspection, including
the first one, consumes one interaction. An unsuccessful trajectory contributes
the cap to the cost metric and zero to the separate success-rate metric.

Offline output is labelled `completed_pipeline_only`. Explicit `--mode live`
uses the configured model and may incur API cost; no live run is part of the
release checks. Preserve model/seed/protocol manifests with any formal results.
Do not derive a success rate from capped means.
