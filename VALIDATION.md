# Validation record

Validated on Windows 11 with Python 3.12, PyTorch 2.5.1, CUDA, and locally installed benchmark datasets.

## Target case

`RCAEval RE2-OB/checkoutservice_cpu/1` was executed in blind-progressive mode with live `gpt-4o-mini` agents:

- 24 Evidence Packs produced by the Perception Agent;
- progressive controller concluded after 13 activated windows;
- Top-1: `checkoutservice`;
- 5 LLM requests, 0 retries, 0 API errors, and 0 schema errors;
- verified report verdict: `PASS`.

## Additional RE2-OB regression cases

| Case | Top-1 | Control | Verified report |
|---|---|---|---|
| `emailservice_delay/1` | `checkoutservice` | CONCLUDE | yes |
| `currencyservice_loss/1` | `currencyservice` | CONCLUDE | yes |
| `productcatalogservice_cpu/1` | `productcatalogservice` | CONCLUDE | yes |

The first row is intentionally retained: the UI displays unsuccessful localization and permits structured feedback instead of filtering the case.

## Cross-dataset smoke checks

| Dataset | Case | Result |
|---|---|---|
| RCAEval RE2-TT | `ts-auth-service_cpu/1` | completed, report verified |
| Eadro-SN | `sn-c00-f00` | completed, report verified |

## UI regression

The browser workflow was exercised at 1600x900: inspect path, select case, run diagnosis, switch among Investigation/Memory Graph/Feedback, submit a structured rejection, and export the updated state. No JavaScript or browser-console errors were observed, and the document width remained within the viewport.
