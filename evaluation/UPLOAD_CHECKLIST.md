# Public-release checklist

This checklist covers the repository package rather than private experiment
workspaces. Complete it from a clean clone before creating a public tag.

## Packaged material

- Root-Teller modules for perception, graph evidence, collaborative diagnosis,
  reporting, and feedback refinement;
- frozen public configurations and grouped split manifests;
- compact manuscript result table and common evaluator definitions;
- portable compatibility adapters and provenance records for all six
  baselines;
- GMO/IAMI dataset-view generator and semantic validator;
- local interactive system, small fold-specific perception checkpoints, and
  self-contained tests;
- paper-to-code mapping and release-integrity audit.

## Maintainer actions

1. Confirm that `evaluation/rq1/results/paper_table.csv` matches the submitted
   manuscript exactly.
2. Select and add an appropriate repository-level `LICENSE` or clearly state
   the applicable distribution terms. Do not infer or overwrite third-party
   licenses.
3. Review `evaluation/rq1/baseline_adapters/THIRD_PARTY_STATUS.md` before
   redistributing upstream snapshots, especially Eadro.
4. Confirm that no raw benchmark telemetry, private labels, API responses,
   credentials, or author-machine paths are tracked.
5. Run the commands below from a fresh clone and archive their console output
   with the release record.
6. Create a release tag and record the public commit hash/tag in the response
   letter or artifact appendix.

## Fresh-clone commands

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e ".[logs,test,system]"

.\.venv\Scripts\python evaluation\rq1\scripts\verify_artifact.py
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\root-teller audit-release --release-root . --output-dir docs\audit
```

