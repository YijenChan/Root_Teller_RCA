# RQ2 telemetry-unavailability path contract

The 18 materialized corrupted datasets are not redistributed. Generate them
from local benchmark copies with `tools/telemetry_unavailability/prepare.py`.

Use the following canonical output root:

```text
<workspace>/dataset/dataset_corrupted/rq2_three_datasets_outerfold_v1/
```

The generator creates six condition directories for each of RE2-OB, RE2-TT,
and Eadro-SN. It writes `_manifests/private_manifest.csv`,
`_manifests/condition_manifest.csv`, and `_manifests/provenance.json`, validates
the staged tree, and only then atomically promotes it to the requested output
path. Evaluation consumers must receive this exact root through
`ROOTTELLER_TUM_ROOT`; they must not infer an older checkpoint-specific path.

```powershell
$env:ROOTTELLER_TUM_ROOT = `
  "$env:ROOTTELLER_WORKSPACE/dataset/dataset_corrupted/rq2_three_datasets_outerfold_v1"

python tools/telemetry_unavailability/prepare.py `
  --re2-root "D:/datasets/RE2" `
  --eadro-sn-root "D:/datasets/SN Dataset" `
  --output-dir $env:ROOTTELLER_TUM_ROOT

python tools/telemetry_unavailability/validate_semantics.py `
  --root $env:ROOTTELLER_TUM_ROOT
```

The fault-injection timestamp is used only to materialize IAMI and never enters
a model feature or LLM prompt. GMO-Trace historical call maps must be built
from development-fold traces and frozen before held-out evaluation.
