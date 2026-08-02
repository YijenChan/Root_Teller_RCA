# Baseline compatibility adapters

These adapters preserve the released diagnostic procedure where executable
and provide only the minimum interface conversion required for the public
benchmarks and service-level evaluation. Labels are read only by the private
evaluator after a ranking has been produced.

| Adapter | Public adaptation boundary |
|---|---|
| `eadro.py` | Align KPI/log/trace tensors, reconstruct the dependency graph, and retrain the released TCN/GAT-style model per system |
| `nezha.py` | Construct fault-free/fault-period event representations and normalize the service ranking |
| `multisource_rcd.py` | Invoke the RCAEval Multi-source RCD implementation and aggregate indicator ranks to services |
| `torai.py` | Invoke the released TORAI implementation and normalize its service ranking |
| `thinkfl.py` | Implement the missing telemetry-tool inference interface; no dataset-specific fine-tuning |
| `rclagent.py` | Build the released per-service evidence, recursive topology, and global synthesis interfaces |

Multi-source RCD's released function consumes metric inputs in this evaluated
configuration. ThinkFL's evaluated tool interface consumes metrics and traces.
These modality boundaries are retained rather than silently adding inputs that
the evaluated implementation does not use.

## RE2 commands

Prepare a fold before running an adapter:

```powershell
python evaluation/rq1/scripts/prepare_fold.py --dataset re2_ob --fold 0
```

Example commands:

```powershell
python evaluation/rq1/baseline_adapters/eadro.py --phase frozen_test --run-id re2ob_fold0
python evaluation/rq1/baseline_adapters/nezha.py --split test --output runs/baselines/nezha/re2ob_fold0
python evaluation/rq1/baseline_adapters/multisource_rcd.py --dataset re2ob --split test --output runs/baselines/multisource_rcd/re2ob_fold0
python evaluation/rq1/baseline_adapters/torai.py --split test --output runs/baselines/torai/re2ob_fold0
python evaluation/rq1/baseline_adapters/thinkfl.py --split test --output runs/baselines/thinkfl/re2ob_fold0
python evaluation/rq1/baseline_adapters/rclagent.py --split test --output runs/baselines/rclagent/re2ob_fold0
```

For RE2-TT, use `run_re2_tt.py <method>` after preparing the corresponding
fold. For Eadro-SN, use `run_eadro_sn.py <method>`; the runner derives the
four leave-one-capture-out folds from the released annotations.

LLM-based adapters read `ROOTTELLER_API_KEY` and `ROOTTELLER_API_BASE` from the
environment. Credentials are never read from or written to repository files.

The upstream implementations remain under `baselines/` and are governed by
their original licenses. These small adapters are Root-Teller artifact code;
they do not replace or relicense upstream projects.

