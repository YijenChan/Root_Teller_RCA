"""
preprocess_re2ob.py
===================
Convert RCAEval RE2-OB case data into the canonical AIOPS-2022 format.

Can process a single case or all 90 cases, merging them into one dataset
(since tool_server loads a single dataset dir at a time, each case becomes
a single entry in error_traces.txt + groundtruth.csv).

Input layout (one RE2-OB case):
  RE2-OB/{service}_{fault}/{instance}/
    traces.csv     – columns: time,traceID,spanID,serviceName,...,duration,statusCode,parentSpanID
    logs.csv       – columns: time,timestamp,container_name,message,level,...
    metrics.csv    – 418 wide-format columns
    inject_time.txt

Output layout (merged):
  {output_dir}/
    trace/all/trace_jaeger-span.csv   – all cases concatenated
    metric/all/metrics.csv
    log/all/logs.csv
    groundtruth.csv                   – one row per case
    error_traces.txt

Usage:
  # Process all 90 cases into one dataset directory
  python preprocess/preprocess_re2ob.py \\
      --input  datasets/RE2-OB \\
      --output data/re2ob
"""

import argparse
import os
import re
import pickle
import glob

import pandas as pd
import numpy as np

pd.set_option("display.max_colwidth", None)


# ── trace conversion ──────────────────────────────────────────────────────────

def _convert_one_trace(case_dir: str, case_id: str) -> pd.DataFrame:
    """Convert one case's traces.csv to canonical format."""
    trace_path = os.path.join(case_dir, "traces.csv")
    if not os.path.exists(trace_path):
        return pd.DataFrame()

    raw = pd.read_csv(trace_path, low_memory=False)
    if len(raw) == 0:
        return pd.DataFrame()

    # Determine timestamp column (startTimeMillis or startTime)
    if "startTimeMillis" in raw.columns:
        ts_col = raw["startTimeMillis"]
    elif "startTime" in raw.columns:
        ts_col = raw["startTime"]
    else:
        ts_col = pd.Series([0] * len(raw))

    out = pd.DataFrame({
        "timestamp":      ts_col.astype(int),
        "cmdb_id":        raw.get("serviceName", "unknown"),
        "span_id":        raw["spanID"],
        "trace_id":       raw["traceID"],
        "duration":       raw["duration"].astype(int),
        "type":           "rpc",
        "status_code":    raw.get("statusCode", 0).fillna(0).astype(int).astype(str),
        "operation_name": raw.get("operationName", "unknown"),
        "parent_span":    raw.get("parentSpanID", np.nan).replace("", np.nan),
    })
    # Tag with case_id for later separation
    out["_case_id"] = case_id
    return out


# ── metric conversion ─────────────────────────────────────────────────────────

def _convert_one_metric(case_dir: str, case_id: str) -> pd.DataFrame:
    """Convert one case's metrics.csv (wide) to long format."""
    metric_path = os.path.join(case_dir, "metrics.csv")
    if not os.path.exists(metric_path):
        return pd.DataFrame()

    raw = pd.read_csv(metric_path, low_memory=False)
    if len(raw) == 0:
        return pd.DataFrame()

    time_col = raw.columns[0]  # "time" usually
    rows = []

    for col in raw.columns[1:]:  # skip time column
        col_lower = col.lower()
        # Parse column name: {entity}_{metric-type-...}
        # Container metrics: checkoutservice_container-cpu-...
        # Node metrics: gke-...node-cpu-...
        service_name = ""
        node_id = ""
        kpi_name = col

        # Try container metric pattern
        m = re.match(r"^(.+?)_(container-.+)$", col)
        if m:
            service_name = m.group(1)
            kpi_name = m.group(2)
        else:
            # Try node metric pattern
            m = re.match(r"^(.+?)_(node-.+)$", col)
            if m:
                node_id = m.group(1)
                kpi_name = m.group(2)

        for _, row in raw.iterrows():
            val = row[col]
            if pd.notna(val):
                rows.append({
                    "timestamp": int(row[time_col]),
                    "node_id": node_id,
                    "service_name": service_name,
                    "kpi_name": kpi_name,
                    "value": val,
                })

    df = pd.DataFrame(rows)
    df["_case_id"] = case_id
    return df


def _convert_one_metric_simple(case_dir: str, case_id: str) -> pd.DataFrame:
    """Use simple_metrics.csv if available (much smaller, 76 cols vs 418)."""
    path = os.path.join(case_dir, "simple_metrics.csv")
    if not os.path.exists(path):
        return _convert_one_metric(case_dir, case_id)

    raw = pd.read_csv(path, low_memory=False)
    if len(raw) == 0:
        return _convert_one_metric(case_dir, case_id)

    time_col = raw.columns[0]
    raw = raw.dropna(subset=[time_col])
    raw[time_col] = raw[time_col].astype(int)

    # Vectorised melt: wide → long
    id_vars = [time_col]
    value_vars = [c for c in raw.columns if c != time_col]
    melted = raw.melt(id_vars=id_vars, value_vars=value_vars,
                      var_name="_col", value_name="value")
    melted = melted.dropna(subset=["value"])

    # Parse service_name and kpi_name from column name
    splits = melted["_col"].str.split("_", n=1, expand=True)
    melted["service_name"] = splits[0].fillna("")
    melted["kpi_name"] = splits[1].fillna(melted["_col"])
    melted["timestamp"] = melted[time_col]
    melted["node_id"] = ""
    melted["_case_id"] = case_id

    return melted[["timestamp", "node_id", "service_name", "kpi_name", "value", "_case_id"]]


# ── log conversion ────────────────────────────────────────────────────────────

def _convert_one_log(case_dir: str, case_id: str) -> pd.DataFrame:
    """Convert one case's logs.csv to canonical format."""
    log_path = os.path.join(case_dir, "logs.csv")
    if not os.path.exists(log_path):
        return pd.DataFrame()

    raw = pd.read_csv(log_path, low_memory=False)
    if len(raw) == 0:
        return pd.DataFrame()

    # Timestamp: the 'timestamp' column is nanosecond unix
    ts_col = raw.get("timestamp", pd.Series([0] * len(raw)))
    ts_seconds = (ts_col // 1_000_000_000).astype(int)

    level = raw.get("level", "unknown").fillna("unknown")
    message = raw.get("message", "").fillna("")

    out = pd.DataFrame({
        "log_id":    [f"re2ob_{case_id}_{i}" for i in range(len(raw))],
        "timestamp": ts_seconds,
        "cmdb_id":   raw.get("container_name", "unknown"),
        "log_name":  "log_" + raw.get("container_name", "unknown").astype(str),
        "value":     "severity: " + level.astype(str) + ", message: " + message.astype(str),
    })
    out["_case_id"] = case_id
    return out


# ── main processing ───────────────────────────────────────────────────────────

def preprocess_re2ob(input_dir: str, output_dir: str, use_simple_metrics: bool = True):
    """Process all RE2-OB cases into one merged dataset."""
    print(f"[preprocess_re2ob] {input_dir} → {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Discover all cases
    cases = []
    for svc_fault_dir in sorted(os.listdir(input_dir)):
        full = os.path.join(input_dir, svc_fault_dir)
        if not os.path.isdir(full):
            continue
        # Parse service and fault type from directory name
        parts = svc_fault_dir.rsplit("_", 1)
        if len(parts) != 2:
            continue
        service, fault_type = parts

        for inst in sorted(os.listdir(full)):
            inst_path = os.path.join(full, inst)
            if not os.path.isdir(inst_path) or not inst.isdigit():
                continue
            inject_path = os.path.join(inst_path, "inject_time.txt")
            if not os.path.exists(inject_path):
                continue
            with open(inject_path) as f:
                inject_ts = int(f.read().strip())

            case_id = f"{svc_fault_dir}_{inst}"
            cases.append({
                "case_id": case_id,
                "path": inst_path,
                "service": service,
                "fault_type": fault_type,
                "inject_ts": inject_ts,
            })

    print(f"  Found {len(cases)} cases")

    # ── Groundtruth ──
    gt_rows = []
    for c in cases:
        gt_rows.append({
            "timestamp": c["inject_ts"],
            "level": "service",
            "cmdb_id": c["service"],
            "failure_type": c["fault_type"],
        })
    gt_df = pd.DataFrame(gt_rows)
    gt_df.to_csv(os.path.join(output_dir, "groundtruth.csv"), index=False)
    print(f"  [groundtruth] {len(gt_df)} rows")

    # ── Process each case ──
    all_traces = []
    all_metrics = []
    all_logs = []
    error_trace_rows = []

    for i, c in enumerate(cases):
        print(f"  [{i+1}/{len(cases)}] {c['case_id']}")

        # Traces
        tdf = _convert_one_trace(c["path"], c["case_id"])
        if len(tdf) > 0:
            all_traces.append(tdf)
            # Find root spans near inject time for error_traces.txt
            inject_ms = c["inject_ts"] * 1000
            root_spans = tdf[tdf["parent_span"].isna()].copy()
            window = root_spans[
                root_spans["timestamp"].between(inject_ms - 60_000, inject_ms + 300_000)
            ]
            if len(window) > 0:
                # Take top-5 by duration
                top = window.nlargest(min(5, len(window)), "duration")
                error_trace_rows.append(top)
            elif len(root_spans) > 0:
                # Fallback: highest duration root spans overall
                top = root_spans.nlargest(min(3, len(root_spans)), "duration")
                error_trace_rows.append(top)

        # Metrics
        if use_simple_metrics:
            mdf = _convert_one_metric_simple(c["path"], c["case_id"])
        else:
            mdf = _convert_one_metric(c["path"], c["case_id"])
        if len(mdf) > 0:
            all_metrics.append(mdf)

        # Logs
        ldf = _convert_one_log(c["path"], c["case_id"])
        if len(ldf) > 0:
            all_logs.append(ldf)

    # ── Write merged outputs ──
    # Traces
    os.makedirs(os.path.join(output_dir, "trace", "all"), exist_ok=True)
    if all_traces:
        trace_merged = pd.concat(all_traces, ignore_index=True)
        trace_merged.drop(columns=["_case_id"], errors="ignore").to_csv(
            os.path.join(output_dir, "trace", "all", "trace_jaeger-span.csv"), index=False
        )
        print(f"  [trace] {len(trace_merged)} total spans")

    # Metrics
    os.makedirs(os.path.join(output_dir, "metric", "all"), exist_ok=True)
    if all_metrics:
        metric_merged = pd.concat(all_metrics, ignore_index=True)
        metric_merged.drop(columns=["_case_id"], errors="ignore").to_csv(
            os.path.join(output_dir, "metric", "all", "metrics.csv"), index=False
        )
        print(f"  [metric] {len(metric_merged)} total rows")

    # Build empty pickle maps (RE2-OB has no explicit node mapping)
    with open(os.path.join(output_dir, "metric", "node_service_map.pkl"), "wb") as f:
        pickle.dump({}, f)
    with open(os.path.join(output_dir, "metric", "service_node_map.pkl"), "wb") as f:
        pickle.dump({}, f)

    # Logs
    os.makedirs(os.path.join(output_dir, "log", "all"), exist_ok=True)
    if all_logs:
        log_merged = pd.concat(all_logs, ignore_index=True)
        log_merged.drop(columns=["_case_id"], errors="ignore").to_csv(
            os.path.join(output_dir, "log", "all", "logs.csv"), index=False
        )
        print(f"  [log] {len(log_merged)} total entries")

    # Error traces
    if error_trace_rows:
        et_merged = pd.concat(error_trace_rows).drop_duplicates(subset=["span_id"])
        et_merged = et_merged.drop(columns=["_case_id"], errors="ignore")
        et_merged = et_merged.sort_values("timestamp").reset_index(drop=True)
        et_path = os.path.join(output_dir, "error_traces.txt")
        with open(et_path, "w") as f:
            f.write(et_merged.to_string())
            f.write("\n")
        print(f"  [error_traces] {len(et_merged)} traces")

    print(f"[preprocess_re2ob] Done. {len(cases)} cases processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True, help="RE2-OB root directory")
    parser.add_argument("--output", required=True, help="Output directory in canonical format")
    parser.add_argument("--full-metrics", action="store_true",
                        help="Use full 418-col metrics.csv instead of simple_metrics.csv")
    args = parser.parse_args()
    preprocess_re2ob(args.input, args.output, use_simple_metrics=not args.full_metrics)
