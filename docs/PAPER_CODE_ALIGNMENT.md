# Paper-to-code alignment

This release was audited against the latest manuscript PDF available during
artifact preparation. The table below maps the manuscript's implementation
claims to executable code and frozen public configuration. The automated
contract is in `tests/test_paper_alignment.py`.

## Methodology mapping

| Manuscript element | Public implementation | Frozen setting |
|---|---|---|
| 60-second adjacent windows | `module1/features.py`, `module2/window_export.py` | `bin_seconds=60`, zero overlap |
| Ten metric groups | `module1/config.py`, `module1/features.py` | CPU, memory, disk, network, socket, request, error, latency, bytes, other |
| Log representation | `module1/features.py` | all-MiniLM-L6-v2 (384 dimensions) plus three count features |
| Trace representation | `module1/features.py` | Six structural, error, and latency features |
| Availability-aware fusion | `module1/model.py` | Availability mask, adaptive fusion, lambda 0.6 |
| Relational graph reasoning | `module1/model.py` | 48-dimensional states, two R-GAT layers, dropout 0.2 |
| Training | `module1/train.py`, `module1/refit.py` | AdamW, LR 1e-3, weight decay 1e-4, gradient clipping 5.0 |
| Missing-modality training | `module1/data.py` | Complete- and suffix-modality dropout, each with probability 0.25 |
| Graph Evidence Excavator | `module2/window_export.py` | Provenance-linked window Evidence Packs |
| RCA Memory Graph | `module2/rmg.py` | Immutable evidence, mutable hypothesis state, typed relations |
| Hierarchical RCA loop | `module2/run.py`, `module2/agents.py` | Stateful Steward, stateless Investigator, two tasks and two rounds |
| Progressive control | `module2/run.py` | Conclude, Expand, or Abstain from validated diagnosis state |
| Cross-window synthesis | `module2/rmg.py`, `module2/agents.py` | Recency 0.94; deterministic/Steward weights 0.80/0.20 |
| Evidence-grounded reporting | `module3/reporting.py` | Canonical evidence ledger and at most three verifier rounds |
| SRE feedback | `module3/feedback.py` | Score multiplied by 0.9 per rejection; evidence is not deleted |

## Evaluation boundary

The repository packages only the RQ1 reproduction path and the compact RQ1
paper table. RQ2--RQ6 result workbooks, plots, response caches, and intermediate
artifacts are intentionally excluded. This is a packaging boundary, not a
different implementation: the same core modules power the local system and the
RQ1 runner.

RQ1 uses complete telemetry, grouped outer folds, seeds 41/42/43, full-range
window access, no SRE feedback, and service-level A@1, A@3, and Avg@5. The
private evaluator alone accesses the test label. Case directory names are
replaced by opaque identifiers before evidence reaches an LLM.

## Intentional public-release boundaries

- Raw benchmark data and the 18 materialized TUM dataset views are not
  redistributed. `tools/telemetry_unavailability/prepare.py` regenerates the
  views from local benchmark copies.
- LLM response caches, run logs, predictions, and credentials are excluded.
- Small perception checkpoints required by the interactive local system are
  included under `system/checkpoints/`; they contain model parameters and fold
  manifests, not benchmark telemetry.
- The six baseline directories contain upstream implementation code only.
  Root-Teller-specific adapters and reproduced result files are not packaged.

## Verification

```powershell
python -m pytest -q
root-teller audit-release --release-root . --output-dir docs/audit
```

The second command verifies key paper values, the repository boundary, English-
only text, absence of credentials and local author paths, and emits a checksum
manifest.
