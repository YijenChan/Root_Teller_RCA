"""
preprocess_nezha.py
===================
Convert Nezha/Augmented-TrainTicket data into the canonical AIOPS-2022 format
so that tool_server.py, coordinator.py, and evaluate.py work without changes.

Input layout (one Nezha date directory):
  rca_data/{date}/
    trace/{HH_MM}_trace.csv
    log/{HH_MM}_log.csv
    metric/{podname}_metric.csv  +  front_service.csv
    traceid/{HH_MM}_traceid.csv
    {date}-fault_list.json

Output layout:
  {output_dir}/
    trace/all/trace_jaeger-span.csv
    metric/all/metrics.csv  +  node_service_map.pkl  +  service_node_map.pkl
    log/all/logs.csv
    groundtruth.csv
    error_traces.txt

Usage:
  python preprocess/preprocess_nezha.py \\
      --input  datasets/Nezha/rca_data/2023-01-29 \\
      --output data/nezha/2023-01-29
"""

import argparse
import json
import os
import pickle
import re
import glob

import pandas as pd
import numpy as np

pd.set_option("display.max_colwidth", None)


# ── helpers ───────────────────────────────────────────────────────────────────

def _pod_to_service(pod_name: str) -> str:
    """
    Extract the service (deployment) base name from a K8s pod name.
    e.g. 'ts-contacts-service-866bd68c97-xcqfx' → 'ts-contacts-service'
    Strategy: strip the last two hyphen-separated segments (replicaset hash + pod hash).
    """
    parts = pod_name.rsplit("-", 2)
    if len(parts) >= 3 and len(parts[-1]) >= 4 and len(parts[-2]) >= 4:
        return parts[0]
    # Fallback: strip just the last segment
    parts2 = pod_name.rsplit("-", 1)
    if len(parts2) == 2 and len(parts2[-1]) >= 4:
        return parts2[0]
    return pod_name


def _parse_log_json(log_str: str) -> str:
    """Extract severity + message from Nezha JSON log field."""
    try:
        obj = json.loads(log_str)
        raw = obj.get("log", "")
        # Parse severity from log line like "16:42:22.382 INFO t.s.Impl#532 ..."
        m = re.match(r"\S+\s+(INFO|WARN|ERROR|DEBUG|TRACE)\s+", raw)
        severity = m.group(1).lower() if m else "unknown"
        message = raw[m.end():].strip() if m else raw.strip()
        return f"severity: {severity}, message: {message}"
    except (json.JSONDecodeError, TypeError):
        return str(log_str)


# ── trace conversion ──────────────────────────────────────────────────────────

def convert_traces(input_dir: str, output_dir: str) -> pd.DataFrame:
    trace_dir = os.path.join(input_dir, "trace")
    out_path = os.path.join(output_dir, "trace", "all", "trace_jaeger-span.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    frames = []
    for f in sorted(glob.glob(os.path.join(trace_dir, "*_trace.csv"))):
        try:
            df = pd.read_csv(f)
            if len(df) > 0:
                frames.append(df)
        except Exception as e:
            print(f"  [warn] trace {f}: {e}")

    if not frames:
        raise RuntimeError(f"No trace files found in {trace_dir}")

    raw = pd.concat(frames, ignore_index=True)
    print(f"  [trace] {len(raw)} spans from {len(frames)} files")

    # Map to canonical columns
    out = pd.DataFrame({
        "timestamp":      (raw["StartTimeUnixNano"] // 1_000_000).astype(int),  # nano → ms
        "cmdb_id":        raw["PodName"],
        "span_id":        raw["SpanID"],
        "trace_id":       raw["TraceID"],
        "duration":       raw["Duration"].astype(int),  # keep as-is (nanoseconds)
        "type":           "rpc",
        "status_code":    "0",
        "operation_name": raw["OperationName"],
        "parent_span":    raw["ParentID"].replace("root", np.nan),
    })
    out.to_csv(out_path, index=False)
    print(f"  [trace] written → {out_path}")
    return out


# ── metric conversion ─────────────────────────────────────────────────────────

def convert_metrics(input_dir: str, output_dir: str):
    metric_dir = os.path.join(input_dir, "metric")
    out_csv = os.path.join(output_dir, "metric", "all", "metrics.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    node_service_map = {}
    service_node_map = {}
    all_rows = []

    for f in sorted(glob.glob(os.path.join(metric_dir, "*_metric.csv"))):
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"  [warn] metric {f}: {e}")
            continue
        if "PodName" not in df.columns or "TimeStamp" not in df.columns:
            continue

        pod_name = df["PodName"].iloc[0] if len(df) > 0 else os.path.basename(f).replace("_metric.csv", "")
        service_name = pod_name  # use full pod name as cmdb_id
        node_id = ""

        # Extract node info if available
        if "NodeCpuUsageRate(%)" in df.columns:
            node_id = "node-1"  # Nezha doesn't give explicit node IDs

        skip_cols = {"Time", "TimeStamp", "PodName"}
        kpi_cols = [c for c in df.columns if c not in skip_cols]

        for _, row in df.iterrows():
            ts = int(row["TimeStamp"])
            for kpi in kpi_cols:
                val = row[kpi]
                if pd.notna(val):
                    all_rows.append({
                        "timestamp": ts,
                        "node_id": node_id,
                        "service_name": service_name,
                        "kpi_name": kpi,
                        "value": val,
                    })

        if node_id:
            node_service_map.setdefault(node_id, set()).add(service_name)
            service_node_map[service_name] = node_id

    # Also handle front_service.csv (service-level aggregates)
    front_path = os.path.join(metric_dir, "front_service.csv")
    if os.path.exists(front_path):
        try:
            df = pd.read_csv(front_path)
            skip = {"Time", "TimeStamp", "ServiceName"}
            kpi_cols = [c for c in df.columns if c not in skip]
            for _, row in df.iterrows():
                ts = int(row["TimeStamp"])
                svc = row.get("ServiceName", "Frontend")
                for kpi in kpi_cols:
                    val = row[kpi]
                    if pd.notna(val):
                        all_rows.append({
                            "timestamp": ts, "node_id": "", "service_name": svc,
                            "kpi_name": kpi, "value": val,
                        })
        except Exception as e:
            print(f"  [warn] front_service.csv: {e}")

    metrics_df = pd.DataFrame(all_rows)
    metrics_df.to_csv(out_csv, index=False)
    print(f"  [metric] {len(metrics_df)} rows → {out_csv}")

    # Pickle maps
    with open(os.path.join(output_dir, "metric", "node_service_map.pkl"), "wb") as f:
        pickle.dump(node_service_map, f)
    with open(os.path.join(output_dir, "metric", "service_node_map.pkl"), "wb") as f:
        pickle.dump(service_node_map, f)


# ── log conversion ────────────────────────────────────────────────────────────

def convert_logs(input_dir: str, output_dir: str):
    log_dir = os.path.join(input_dir, "log")
    out_csv = os.path.join(output_dir, "log", "all", "logs.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    frames = []
    for f in sorted(glob.glob(os.path.join(log_dir, "*_log.csv"))):
        try:
            df = pd.read_csv(f)
            frames.append(df)
        except Exception as e:
            print(f"  [warn] log {f}: {e}")

    if not frames:
        print("  [log] no log files found, creating empty")
        pd.DataFrame(columns=["log_id", "timestamp", "cmdb_id", "log_name", "value"]).to_csv(out_csv, index=False)
        return

    raw = pd.concat(frames, ignore_index=True)
    print(f"  [log] {len(raw)} entries from {len(frames)} files")

    out = pd.DataFrame({
        "log_id":    [f"nezha_{i}" for i in range(len(raw))],
        "timestamp": (raw["TimeUnixNano"] // 1_000_000_000).astype(int),  # nano → sec
        "cmdb_id":   raw["PodName"],
        "log_name":  "log_" + raw["Container"].astype(str),
        "value":     raw["Log"].apply(_parse_log_json),
    })
    out.to_csv(out_csv, index=False)
    print(f"  [log] written → {out_csv}")


# ── groundtruth ───────────────────────────────────────────────────────────────

# Nezha's inject_timestamp timezone varies by date:
# - 2023-01-29: inject_timestamp is LOCAL (UTC+8) epoch → needs +8h
# - 2023-01-30: inject_timestamp is already UTC → no offset needed
# We auto-detect by comparing inject_time (local) with utcfromtimestamp(inject_timestamp).

def _detect_tz_offset(fault_data: dict) -> int:
    """Auto-detect timezone offset between inject_timestamp and actual UTC."""
    from datetime import datetime
    for hour_key, faults in fault_data.items():
        if not isinstance(faults, list):
            faults = [faults]
        for fault in faults:
            inject_time_str = fault.get("inject_time", "")
            raw_ts = int(fault["inject_timestamp"])
            utc_dt = datetime.utcfromtimestamp(raw_ts)
            # Parse local time from inject_time
            try:
                parts = inject_time_str.split()
                time_parts = parts[1].split(":")
                local_hour = int(time_parts[0])
                utc_hour = utc_dt.hour
                offset_hours = round((local_hour - utc_hour) % 24)
                if offset_hours > 12:
                    offset_hours -= 24
                return offset_hours * 3600
            except Exception:
                continue
    return 0  # default: no offset


_TZ_OFFSET_SEC = None  # will be set during preprocessing


def convert_groundtruth(input_dir: str, output_dir: str):
    global _TZ_OFFSET_SEC
    date = os.path.basename(input_dir)
    fault_path = os.path.join(input_dir, f"{date}-fault_list.json")
    out_path = os.path.join(output_dir, "groundtruth.csv")

    with open(fault_path) as f:
        fault_data = json.load(f)

    # Auto-detect timezone offset
    _TZ_OFFSET_SEC = _detect_tz_offset(fault_data)
    print(f"  [timezone] detected offset = {_TZ_OFFSET_SEC}s ({_TZ_OFFSET_SEC//3600}h)")

    rows = []
    for hour_key, faults in fault_data.items():
        if not isinstance(faults, list):
            faults = [faults]
        for fault in faults:
            pod = fault["inject_pod"]
            utc_ts = int(fault["inject_timestamp"]) + _TZ_OFFSET_SEC
            rows.append({
                "timestamp":    utc_ts,
                "level":        "pod",
                "cmdb_id":      pod,
                "failure_type": fault["inject_type"],
            })

    gt = pd.DataFrame(rows)
    gt.to_csv(out_path, index=False)
    print(f"  [groundtruth] {len(gt)} faults → {out_path}")
    return gt


# ── error traces ──────────────────────────────────────────────────────────────

def generate_error_traces(input_dir: str, output_dir: str, trace_df: pd.DataFrame):
    """
    Two-phase strategy:
    1. Use traceid/ files to find traces active during each fault window.
    2. Fallback to ±120s timestamp matching for any uncovered faults.
    """
    date = os.path.basename(input_dir)
    fault_path = os.path.join(input_dir, f"{date}-fault_list.json")
    traceid_dir = os.path.join(input_dir, "traceid")
    out_path = os.path.join(output_dir, "error_traces.txt")

    with open(fault_path) as f:
        fault_data = json.load(f)

    # Build traceid index: {HH_MM → set of trace_ids}
    traceid_index = {}
    if os.path.isdir(traceid_dir):
        for fname in os.listdir(traceid_dir):
            if fname.endswith("_traceid.csv"):
                hhmm = fname.replace("_traceid.csv", "")
                ids = set()
                with open(os.path.join(traceid_dir, fname)) as f:
                    for line in f:
                        tid = line.strip()
                        if tid:
                            ids.add(tid)
                traceid_index[hhmm] = ids

    root_spans = trace_df[trace_df["parent_span"].isna()].copy()
    selected = []

    for hour_key, faults in fault_data.items():
        if not isinstance(faults, list):
            faults = [faults]
        for fault in faults:
            inject_utc_sec = int(fault["inject_timestamp"]) + _TZ_OFFSET_SEC
            inject_utc_ms = inject_utc_sec * 1000

            found = False

            # Strategy 1: Use traceid files.
            # The inject_time is local time like "2023-01-29 08:43:04".
            # Extract HH_MM from inject_time to find the matching traceid file.
            inject_local_time = fault.get("inject_time", "")
            if inject_local_time:
                parts = inject_local_time.split()
                if len(parts) >= 2:
                    time_parts = parts[1].split(":")
                    if len(time_parts) >= 2:
                        hhmm = f"{time_parts[0]}_{time_parts[1]}"
                        if hhmm in traceid_index:
                            trace_ids = traceid_index[hhmm]
                            matching = root_spans[root_spans["trace_id"].isin(trace_ids)]
                            if len(matching) > 0:
                                top = matching.nlargest(min(5, len(matching)), "duration")
                                selected.append(top)
                                found = True
                        # Also check ±1 minute
                        for delta in [-1, 1]:
                            m = int(time_parts[1]) + delta
                            h = int(time_parts[0])
                            if m < 0: m = 59; h -= 1
                            if m > 59: m = 0; h += 1
                            alt_hhmm = f"{h:02d}_{m:02d}"
                            if alt_hhmm in traceid_index and not found:
                                trace_ids = traceid_index[alt_hhmm]
                                matching = root_spans[root_spans["trace_id"].isin(trace_ids)]
                                if len(matching) > 0:
                                    top = matching.nlargest(min(5, len(matching)), "duration")
                                    selected.append(top)
                                    found = True

            # Strategy 2: Timestamp window fallback (using corrected UTC timestamp).
            if not found:
                window = root_spans[
                    root_spans["timestamp"].between(inject_utc_ms - 120_000, inject_utc_ms + 120_000)
                ]
                if len(window) > 0:
                    top = window.nlargest(min(5, len(window)), "duration")
                    selected.append(top)

    if not selected:
        threshold = root_spans["duration"].quantile(0.95)
        selected = [root_spans[root_spans["duration"] > threshold]]

    combined = pd.concat(selected).drop_duplicates(subset=["span_id"])
    combined = combined.sort_values("timestamp").reset_index(drop=True)

    with open(out_path, "w") as fout:
        fout.write(combined.to_string())
        fout.write("\n")

    print(f"  [error_traces] {len(combined)} traces → {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def preprocess_nezha(input_dir: str, output_dir: str):
    print(f"[preprocess_nezha] {input_dir} → {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    trace_df = convert_traces(input_dir, output_dir)
    convert_metrics(input_dir, output_dir)
    convert_logs(input_dir, output_dir)
    convert_groundtruth(input_dir, output_dir)
    generate_error_traces(input_dir, output_dir, trace_df)

    print(f"[preprocess_nezha] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True, help="Nezha rca_data/{date} directory")
    parser.add_argument("--output", required=True, help="Output directory in canonical format")
    args = parser.parse_args()
    preprocess_nezha(args.input, args.output)
