# RQ1: overall root-cause localization

RQ1 evaluates complete telemetry, full-range access, no SRE feedback, and an
ordered service-level ranking. All methods use the same outer-test cases and
the same A@1, A@3, and Avg@5 evaluator.

## Dataset layout

Set `ROOTTELLER_WORKSPACE` to a directory containing the repository and the
locally downloaded datasets:

```text
<workspace>/
  evaluation/rq1/manifests/
  dataset/
    RCAEval RE/RE2/RE2-OB/RE2-OB/<service_fault>/<repeat>/
      metrics.csv
      logs.csv
      traces.csv
    RCAEval RE/RE2/RE2-TT/RE2-TT/<service_fault>/<repeat>/
      metrics.csv
      logs.csv
      traces.csv
    Eadro-SN/SN Dataset/SN Dataset/
      data/
        SN.fault-*.json
        SN.*/metrics/*.csv
        SN.*/logs.json
        SN.*/spans.json
      no fault/
        SN.*.tar.xz
```

The code also consumes RCAEval's released derived files such as
`simple_metrics.csv`, `logts.csv`, and trace time-series files when a native
baseline interface requires them.

## Frozen split protocol

- RE2-OB and RE2-TT each contain 90 incidents in 30 service--fault families.
  The three repeated injections of a family never cross train, validation, or
  test roles within an outer fold. Each of three folds has 45 training, 15
  validation, and 30 test incidents.
- Eadro-SN uses grouped four-fold leave-one-capture-out evaluation. Each fold
  has 18 training, nine validation, and nine test incidents; a continuous
  capture never crosses roles.

The exact assignments are in `manifests/`. `case_catalog.csv` supplies case
metadata to the feature loaders but does **not** define an evaluation split.
The nested fold JSON files are the authoritative split definitions.

## Root-Teller

Run every dataset with seeds 41, 42, and 43 as documented in the repository
README. Each run writes its fold manifest, private evaluation file, summary,
and checksum under the configured workspace. Use
`scripts/aggregate_seed_results.py` to export a compact mean/std table from
completed summaries.

## Baselines

`baseline_adapters/` contains the compatibility code used to normalize each
baseline to the benchmark telemetry and an ordered service ranking. For a RE2
fold, first materialize the active split table:

```powershell
$env:ROOTTELLER_WORKSPACE = (Get-Location).Path
python evaluation/rq1/scripts/prepare_fold.py --dataset re2_ob --fold 0
$env:ROOTTELLER_ACTIVE_SPLIT_MANIFEST = `
  "$env:ROOTTELLER_WORKSPACE/evaluation/rq1/manifests/active_split_manifest.csv"
```

Then invoke the selected adapter. Repeat for all outer folds and aggregate
predictions only after every incident has been evaluated exactly once. See
`baseline_adapters/README.md` for method-specific commands and adaptation
boundaries.

## Results and provenance

`results/paper_table.csv` is the machine-readable transcription of the RQ1
table. It is never read by the training or inference code and should not be
interpreted as a bundled execution log. The released manifests, evaluator,
adapters, and orchestration code make the experimental boundary inspectable;
rerunning it requires the cited public datasets and the configured model
backends. Raw datasets, private evaluator labels, API response caches, and
per-case predictions are deliberately excluded.
