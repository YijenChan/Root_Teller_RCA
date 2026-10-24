# Paper-to-code alignment (revision implementation)

This release exposes the revised method, not a claim that historical experiment
outputs were regenerated. See [revision notes](REVISION2_ALIGNMENT.md) for
provenance, configuration differences, and the exact verification boundary.

| Manuscript mechanism | Executable implementation | Contract |
|---|---|---|
| Availability-aware perception | `module1/model.py`, `features.py` | Preserve channel masks; bypass unavailable embeddings |
| Within-window temporal input | `FeatureConfig.paper_intervals()`, `module2/window_export.py` | Optional 1-second features grouped into 60-second RCA windows |
| Window evidence | `module2/window_export.py` | Typed Evidence Packs and absolute anomaly-onset timestamps |
| Cross-window memory | `module2/rmg.py`, `cross_window.py` | Immutable source records; bounded non-adjacent evidence associations |
| Progressive investigation | `module2/run.py`, `agents.py` | Acquire only activated history; conclude/expand/abstain |
| Structured SRE feedback | `module3/structured/`, `module3/session.py` | Bind claims, entities, windows, incident and Snapshot version |
| Targeted checks | `module3/structured/coordinator.py`, `refinement/investigator.py` | Bounded tasks; typed evidence validation; retain unresolved findings |
| Fresh diagnosis synthesis | `module3/refinement/steward.py`, `scoring.py` | Reassess selected evidence and feedback; apply cumulative prior once to a fresh score |
| Report regeneration | `module3/refinement/reporting.py` | New report tied to the new Snapshot and deterministic consistency checks |
| Feedback controls | `module3/refinement/baselines.py` | Fixed Top-k, revisit inspection, fixed candidate downweighting |
| Paired evaluation | `module3/evaluate_feedback.py` | Common T0; private oracle; acceptance counts as an interaction; success and capped cost |
| Added RCA baselines | `root_teller/baselines/` | Paper-based, task-adapted implementations; record actual model and sampling settings |
| Interactive feedback | `system/root_teller_system/engine.py` | Calls the revised coordinator rather than editing an old report summary |

## Historical versus revised configurations

Historical configuration files dated July 2026, bundled model checkpoints,
and legacy results are retained unchanged. They encode 60-second feature bins.
The new `paper_intervals()` preset and `--aggregation-seconds 1` support the
revised temporal input; fine-grained features and corresponding trained weights
must be generated before claiming matching experimental results. The SN/TT
dataset-specific historical runners retain their recorded preprocessing; verify
their sampling contract before using them for a new temporal-resolution study.

Cross-window linking changes model context and cache identities. It does not
make an old API response a new-model result. Historical tables and UI recordings
cannot validate the revised refinement workflow automatically.

`module3/feedback.py` and `run_feedback.py` are explicitly historical
fixed-score controls. The active feedback entry point is `module3.session`.
Without an injected model client, the revised pipeline is labelled offline and
uses zero model-ranking weight; binary global rejections may then equal fixed
candidate downweighting. No comparative effectiveness is fabricated.

## Public evaluation records

`evaluation/rq1/results/paper_table.csv` is a transcription of the supplied
October manuscript table, not a fresh execution log. The prior transcription is
preserved as `paper_table_legacy_r1.csv`. The added baseline source reproduces
the locally developed algorithms, not an official author implementation.
Model/backbone equivalence, seeds, and result provenance require the matching
experiment manifests; this update does not certify those empirical claims.

## Release verification

Run `python -m pytest -q` and
`root-teller audit-release --release-root . --output-dir docs/audit`.
Tests cover source preservation, version binding, duplicate feedback, rollback,
independent final acceptance, model-client orchestration, and interaction
accounting. Synthetic protocol tests are engineering checks, not benchmark or
human-subject results.

Raw datasets, private labels, credentials, API caches, session databases and
large run directories are not part of this release. The independently maintained
`office_mini_storm_dataset/` remains outside the method update.
