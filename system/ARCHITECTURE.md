# System architecture

The local application is intentionally a thin product layer over the paper implementation.

```text
Case path / ZIP
      |
      v
Dataset adapter (RE2-OB / RE2-TT / Eadro-SN)
      |
      v
Perception Agent -> 60-second provenance-linked Evidence Packs
      |
      v
Evidence Steward <-> Window Investigator
      |
      v
Cross-window RCA Memory Graph -> snapshot-linked verified RCA report
      ^                                      |
      |                                      v
Fresh Steward synthesis <- targeted checks <- claim-targeted SRE feedback
```

`root_teller_system/engine.py` calls the implementation in `src/root_teller`; it does not reimplement the model or agent prompts. Runtime files are isolated under `system/runtime/`, while small frozen perception checkpoints are bundled under `system/checkpoints/`.

The frontend is dependency-free HTML/CSS/JavaScript. It polls background jobs,
renders progressive access and ranking traces, exposes the RMG, and submits
schema-bounded feedback events. Targets bind to an incident, claim, window and
snapshot version. Support, challenge and check-only comments are distinct from
the terminal final-acceptance event.

The engine calls `module3.session.build_coordinator`, not the historical
fixed-score overlay. Accepted comments can schedule bounded Investigator work,
invoke fresh Steward synthesis, apply the cumulative hypothesis prior once,
and persist a new snapshot with a regenerated verified report. Duplicate IDs
do not repeat work; stale versions and failed transactions cannot partially
replace persisted diagnostic state. Original evidence and relations remain
immutable.

Cross-window associations use activated windows only. A non-adjacent link is a
provenance-bearing context association, not a verified causal edge. The bundled
Perception Agent weights still use historical 60-second bins; fine-interval
training is not performed by launching this application.

With no model client, refinement follows the explicitly labelled offline
typed-evidence path. It is a functional fallback, not a simulated LLM result.
