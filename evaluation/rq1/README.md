# RQ1: overall localization accuracy

This is the only evaluation component packaged in the public repository. It
uses complete telemetry, full-range window access, no SRE feedback,
service-level ground truth, and grouped outer folds. Root-Teller incorporates
all available 60-second constituent windows before returning one ordered,
duplicate-free service ranking.

## Protocol

- RCAEval RE2-OB and RE2-TT: three grouped outer folds, each with 60
  development and 30 test cases. Repeated injections from the same
  service/fault group remain in the same fold.
- Eadro-SN: grouped four-fold leave-one-capture-out evaluation, with all nine
  injections from one continuous capture kept together.
- Perception Agent seeds: 41, 42, and 43.
- Reported metrics: A@1, A@3, and Avg@5.
- Test labels are read only by the private evaluator. Incident names, root
  labels, fault types, and injection timestamps are not exposed to LLM prompts.

## Entry points

The complete runner is `root_teller.multidataset.rq1_three_seed`. Each
dataset/seed pair receives isolated checkpoints, Evidence Packs, LLM caches,
and output directories.

```powershell
python -m root_teller.multidataset.rq1_three_seed --stage all --dataset re2_ob --seed 41
python -m root_teller.multidataset.rq1_three_seed --stage all --dataset re2_tt --seed 41
python -m root_teller.multidataset.rq1_three_seed --stage all --dataset eadro_sn --seed 41
```

Repeat for seeds 42 and 43, then run:

```powershell
python -m root_teller.multidataset.rq1_three_seed --stage aggregate
python -m root_teller.multidataset.verify_rq1_three_seed
```

Generated checkpoints, response caches, per-case predictions, and console logs
are not stored in this repository. `results/paper_table.csv` is a compact
transcription of the paper table and is never used as an inference input.
