# GMO/IAMI telemetry-view generator

The raw and corrupted datasets are intentionally not distributed here. This
tool materializes six conditions for each of three local systems:

- GMO-Metric, GMO-Log, GMO-Trace;
- IAMI-Metric, IAMI-Log, IAMI-Trace.

Together, RCAEval RE2-OB, RCAEval RE2-TT, and Eadro-SN produce 18
dataset-condition groups. The generator writes only the damaged modality and
uses opaque case identifiers in public view directories.

## Inputs

`--re2-root` must contain:

```text
RE2-OB/RE2-OB/
RE2-TT/RE2-TT/
```

`--eadro-sn-root` must contain Eadro-SN's `data/` directory. The bundled small
manifest fixes RCAEval common observation windows and grouped outer folds; it
does not contain telemetry records.

## Run

```powershell
python prepare.py `
  --re2-root "D:\datasets\RE2" `
  --eadro-sn-root "D:\datasets\SN Dataset" `
  --output-dir "D:\generated\rq2_three_datasets_outerfold_v1" `
  --dry-run

python prepare.py `
  --re2-root "D:\datasets\RE2" `
  --eadro-sn-root "D:\datasets\SN Dataset" `
  --output-dir "D:\generated\rq2_three_datasets_outerfold_v1"

python validate_semantics.py `
  --root "D:\generated\rq2_three_datasets_outerfold_v1"
```

Use `--resume` only to continue an interrupted `.incomplete` staging output.
The generator refuses to overwrite an existing validated output directory.
