"""
preprocessing_traces.py
=======================
Scan all root spans (``parent_span`` is NaN) in ``trace_jaeger-span.csv``
and emit ``error_traces.txt`` under ``<data_root>`` — a tab-separated file
listing anomalous root spans (either status-code errors or duration above
the threshold). Each listed root span is analysed by the RCLAgent
coordinator as one fault case.

Usage
-----
  python preprocess/preprocessing_traces.py [data_root] [duration_threshold_us]

  data_root             dataset directory containing
                        trace/all/trace_jaeger-span.csv
                        (default: sample_data)
  duration_threshold_us microsecond threshold above which a span is "slow"
                        (default: 10_000_000 = 10 s)
"""

import os
import sys
import pandas as pd

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)


# ── helpers ──────────────────────────────────────────────────────────────────

def _is_error_status(value) -> bool:
    """Return True when status_code indicates a non-success response."""
    ok_values = {"ok", "0", "200", 0, 200}
    if pd.isna(value):
        return False
    try:
        return str(value).strip().lower() not in ok_values
    except Exception:
        return False


def find_error_traces(
    data_root: str,
    duration_threshold_us: int = 10_000_000,
) -> pd.DataFrame:
    """Return a DataFrame of anomalous root spans."""
    trace_path = os.path.join(data_root, "trace", "all", "trace_jaeger-span.csv")
    if not os.path.exists(trace_path):
        raise FileNotFoundError(f"Trace file not found: {trace_path}")

    df = pd.read_csv(trace_path)

    # Root spans have no parent.
    root_spans = df[df["parent_span"].isna()].copy()

    # Two anomaly criteria.
    status_errors   = root_spans[root_spans["status_code"].apply(_is_error_status)]
    duration_errors = root_spans[root_spans["duration"] > duration_threshold_us]

    combined = pd.concat([status_errors, duration_errors]).drop_duplicates(
        subset=["span_id"]
    )
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    return combined


def write_error_traces(
    data_root: str,
    duration_threshold_us: int = 10_000_000,
) -> str:
    """Write error_traces.txt and return the output path."""
    combined = find_error_traces(data_root, duration_threshold_us)

    out_path = os.path.join(data_root, "error_traces.txt")
    with open(out_path, "w") as fout:
        # Write a human-readable header that the coordinator skips (line 0).
        fout.write(combined.to_string())
        fout.write("\n")

    print(
        f"[preprocessing_traces] Written {len(combined)} error traces → {out_path}"
    )
    return out_path


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    data_root_arg   = sys.argv[1] if len(sys.argv) > 1 else "sample_data"
    threshold_arg   = int(sys.argv[2]) if len(sys.argv) > 2 else 10_000_000

    write_error_traces(data_root_arg, threshold_arg)
