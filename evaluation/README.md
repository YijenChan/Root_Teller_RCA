# Evaluation artifact

This directory contains the compact, paper-aligned evaluation material that
can be distributed without redistributing benchmark telemetry, private labels,
or experiment-host API responses.

| Path | Contents |
|---|---|
| `rq1/` | Frozen split manifests, compact paper table, portable baseline adapters, and integrity scripts |
| `rq2/` | The path contract connecting the GMO/IAMI generator to evaluation consumers |

The raw RCAEval and Eadro-SN datasets are not included. Experiment-host
checkpoints, private labels, response caches, per-case rankings, and LLM
transcripts also remain outside the repository. Nothing in this directory is
used as an inference feature.

Run the lightweight artifact audit from the repository root:

```powershell
python evaluation/rq1/scripts/verify_artifact.py
```

Before a public release, complete the maintainer checks in
[`UPLOAD_CHECKLIST.md`](UPLOAD_CHECKLIST.md), including the repository license,
release tag, relative-link audit, and fresh-clone test.
