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
RCA Memory Graph -> verified RCA report -> structured SRE feedback overlay
```

`root_teller_system/engine.py` calls the implementation in `src/root_teller`; it does not reimplement the model or agent prompts. Runtime files are isolated under `system/runtime/`, while small frozen perception checkpoints are bundled under `system/checkpoints/`.

The frontend is dependency-free HTML/CSS/JavaScript. It polls background jobs, renders progressive access and ranking traces, exposes the RMG, and submits only schema-bounded feedback events. A rejected hypothesis is confidence-decayed but its original evidence and relations remain immutable.
