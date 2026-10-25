# RCAgent (CIKM 2024) benchmark adaptation

Entry point: `python -m root_teller.baselines.run --method rcagent --help`.
Implementation: `src/root_teller/baselines/methods.py` (`RCAgent`).

This is a paper-based reproduction. It retains the controller/tool loop,
Observation Snapshot Keys with actual pagination, log-expert clustering,
JSON regeneration, error feedback and trajectory-level self-consistency.
Unavailable Flink source-code and advisor databases are reported unavailable,
not simulated. Text output is adapted to a distinct allowed-service Top-5.
No private target is used to repair a malformed ranking.

The runtime model and TSC sample count are explicit and recorded with each run.
