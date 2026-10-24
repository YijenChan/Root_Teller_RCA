# Baseline compatibility adapters

These adapters normalize eight methods to benchmark telemetry and service-level
rankings. Their boundaries differ: some wrap upstream implementations, whereas
others are task/model adaptations. They must not all be described as native
author reproductions. Test-case labels are kept outside diagnostic inputs and
used by the private evaluator after a ranking is produced; the RCACopilot
adaptation also uses labelled development-history cases from permitted splits.

| Adapter | Public adaptation boundary |
|---|---|
| `eadro.py` | Align KPI/log/trace tensors, reconstruct the dependency graph, and retrain the released TCN/GAT-style model per system |
| `nezha.py` | Construct fault-free/fault-period event representations and normalize the service ranking |
| `multisource_rcd.py` | Invoke the RCAEval Multi-source RCD implementation and aggregate indicator ranks to services |
| `torai.py` | Invoke the released TORAI implementation and normalize its service ranking |
| `thinkfl.py` | Implement the missing telemetry-tool inference interface; no dataset-specific fine-tuning |
| `rclagent.py` | Build the released per-service evidence, recursive topology, and global synthesis interfaces |
| `rcacopilot.py` | Paper-based retrieval/summary adaptation; labelled history uses only the declared development split |
| `rcagent.py` | Paper-based tool, observation-memory, log-clustering, and trajectory-sampling adaptation |

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

Those two historical dataset wrappers support the original six methods. For
RCACopilot/RCAgent on **any** of the three datasets, use their dedicated runner
instead (do not route them through `run_re2_tt.py` or `run_eadro_sn.py`):

```powershell
python -m pip install -e ".[baselines]"
python -m root_teller.baselines.run --method rcacopilot --dataset re2_ob --fold 0 --split validation --mode prepare --output runs/rcacopilot-prepare
python -m root_teller.baselines.run --method rcagent --dataset eadro_sn --fold 0 --split validation --mode prepare --output runs/rcagent-prepare
```

Preparation does not call an LLM. Formal runs require explicit `--mode live`,
model/sampling budgets, datasets and a new output directory. See
[added-method protocol](../../../baselines/REPRODUCTIONS.md) for the recorded
GPT-5-nano adaptation, rather than claiming native GPT-4/Vicuna parity.

LLM-based adapters read `ROOTTELLER_API_KEY` and `ROOTTELLER_API_BASE` from the
environment. Credentials are never read from or written to repository files.

Available upstream snapshots remain under repository-level `baselines/` and
retain their original licenses. Eadro implementation code is not redistributed.
The two new algorithm adaptations live in `src/root_teller/baselines/` and are
Root-Teller-authored code. These adapters do not replace or relicense upstream
projects. Historical ThinkFL/RCLAgent compatibility model settings also need
matching execution manifests before any native-model parity claim.

