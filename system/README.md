# Root-Teller local system

The local system is a thin FastAPI and browser layer over the paper
implementation in `src/root_teller`. It does not reimplement the model or
prompts. A case supplied by path or ZIP archive passes through multimodal
feature extraction, the Perception Agent, Evidence Pack generation, the
hierarchical RCA loop, verified reporting, and optional structured feedback.

## Install

From the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[logs,test]"
.\.venv\Scripts\python -m pip install -r system\requirements.txt
```

## Configure

`ROOTTELLER_WORKSPACE` points to the directory that contains local datasets,
caches, and experiment runs. API credentials are optional and are read from
environment variables:

```powershell
$env:ROOTTELLER_WORKSPACE = "D:\path\to\your\workspace"
$env:ROOTTELLER_API_KEY = "your-openai-compatible-key"
$env:ROOTTELLER_API_BASE = "https://your-endpoint.example/v1"
```

For compatibility with private experiment workspaces, the backend can also
read an untracked `config/API_KEY.txt`; environment variables take precedence.
Never commit that file.

## Start

```powershell
.\system\start.ps1 -Workspace $env:ROOTTELLER_WORKSPACE -Port 4315
```

Open <http://127.0.0.1:4315/>. Omitting API credentials activates the
deterministic evidence path and labels the run as a fallback in the UI.

## Supported inputs

The path inspector accepts:

- an RCAEval RE2-OB fault-family directory or numeric repetition directory;
- an RCAEval RE2-TT fault-family directory or numeric repetition directory;
- an Eadro-SN data directory, capture directory, or `SN.fault-*.json` file;
- a ZIP archive containing one of those layouts.

Example layout:

```text
workspace/
  dataset/
    RCAEval RE/
      RE2/
        RE2-OB/RE2-OB/<service_fault>/<repetition>/
        RE2-TT/RE2-TT/<service_fault>/<repetition>/
    Eadro-SN/SN Dataset/SN Dataset/data/
```

Uploaded archives are checked for traversal and size limits before extraction.
Runtime files are isolated under `system/runtime/` and ignored by Git.

## Interface workflow

1. **Overview** — inspect a path or upload a ZIP, choose an incident, protocol,
   and whether to use live LLM agents.
2. **Investigation** — observe window activation and the progressive decision.
3. **Memory Graph** — inspect provenance-linked evidence, typed relations, and
   hypothesis state.
4. **Feedback** — submit a schema-bounded Accept or Reject verdict against an
   active hypothesis. Rejection reduces confidence but preserves evidence.
5. **Settings** — review runtime, model, and security boundaries.

The `default` protocol incorporates the complete predefined observation range,
matching RQ1. The `blind` protocol exposes earlier windows only when requested
by progressive control, matching the RQ3 setting.

## Checkpoints

Small fold-specific Perception Agent checkpoints are bundled in
`system/checkpoints/` so the UI can run without training first. They contain
model parameters, reference statistics, and fold manifests; raw benchmark
telemetry is not included.

## Tests

```powershell
python -m pytest system\tests -q
```

Set `ROOTTELLER_TEST_CASE` to a locally installed RE2-OB family directory to
enable the dataset-dependent path-inspection test. The remaining tests are
self-contained.

The validation record is in [`VALIDATION.md`](VALIDATION.md), and the product
layer boundary is documented in [`ARCHITECTURE.md`](ARCHITECTURE.md).
