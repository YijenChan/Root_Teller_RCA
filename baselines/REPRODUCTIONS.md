# Added baseline reproduction protocol

Install the optional dependencies with `pip install -e ".[baselines]"`.
The runner defaults to telemetry preparation, not a model call:

```powershell
python -m root_teller.baselines.run --method rcagent --dataset re2_ob --mode prepare --workspace /path/to/workspace --output /path/to/new-run
```

Use `--manifests` for an explicit frozen split directory. The default is the
packaged `evaluation/rq1/manifests/`. Live inference requires explicit
`--mode live`, a model ID, and environment credentials. Replay never invokes
a model on a cache miss. Failed model calls do not produce substitute rankings.

The source default `gpt-5-nano-2025-08-07` records the local adaptation, not a
claim that the native GPT-4/Vicuna settings were reproduced. Set `--model`,
`--samples`, `--embedding-model` and budgets explicitly for the intended
experiment. GPT-5 deployments that omit temperature are recorded as such.
The paper's model labels, three-seed aggregation and provenance must be checked
against actual execution manifests before claiming a direct reproduction.

The original six upstream snapshots are retained untouched. The two new methods
are distributed as Root-Teller's paper-based adaptation code under
`root_teller.baselines`; no third-party demonstration data is used as a ranker.
