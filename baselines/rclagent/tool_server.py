"""
tool_server.py – Flask data-API server for RCLAgent.

Serves preprocessed trace, metric, and log data to RCLAgent's Dedicated
Agents. ``DATA_ROOT``, host, and port are read from ``config.py`` and can
be overridden via environment variables.

Endpoints
---------
GET /search_span?span_id=<id>
    Return the span row(s) matching span_id.

GET /search_traces?parent_span_id=<id>
    Return child spans whose parent_span == parent_span_id.

GET /search_logs?service_name=<name>&timestamp=<unix>
    Return non-INFO/DEBUG log lines for service_name within ±60 s of the
    given timestamp.

GET /search_fluctuating_metrics?service_name=<name>&timestamp=<unix>
    Return KPIs showing a 3-sigma spike in the ±10 min window around the
    given timestamp, measured against a ±20 min baseline.
"""

import os
import pickle

import pandas as pd
from flask import Flask, request, jsonify

import config

pd.set_option("display.max_rows",     None)
pd.set_option("display.max_columns",  None)
pd.set_option("display.width",        None)
pd.set_option("display.max_colwidth", None)

app       = Flask(__name__)
DATA_ROOT = config.DATA_ROOT


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_log_df() -> pd.DataFrame:
    """Load log CSVs, skipping huge envoy logs (3GB+) that rarely help diagnosis."""
    log_base = os.path.join(DATA_ROOT, "log")
    frames   = []
    for root, _, files in os.walk(log_base):
        for fname in files:
            if not fname.endswith(".csv"):
                continue
            # Skip envoy proxy logs — they're enormous and rarely contain
            # root-cause signals; service application logs are sufficient.
            if "envoy" in fname.lower():
                print(f"  [log] skipping envoy log: {fname}")
                continue
            fpath = os.path.join(root, fname)
            try:
                print(f"  [log] loading {fname} ...")
                df = pd.read_csv(fpath, low_memory=False)
                if "value" in df.columns:
                    noise = df["value"].astype(str).str.lower().str.contains(
                        r"severity:\s*(?:info|debug)", regex=True, na=False
                    )
                    df = df[~noise]
                frames.append(df)
            except Exception as exc:
                print(f"  [warn] skipping {fpath}: {exc}")
    if not frames:
        return pd.DataFrame(columns=["log_id", "timestamp", "cmdb_id", "log_name", "value"])
    return pd.concat(frames, ignore_index=True)


def _load_maps():
    ns_path = os.path.join(DATA_ROOT, "metric", "node_service_map.pkl")
    sn_path = os.path.join(DATA_ROOT, "metric", "service_node_map.pkl")
    ns, sn  = {}, {}
    try:
        with open(ns_path, "rb") as f:
            ns = pickle.load(f)
        with open(sn_path, "rb") as f:
            sn = pickle.load(f)
    except FileNotFoundError:
        print(f"[warn] mapping pickles not found — run preprocessing_metrics.py first")
    return ns, sn


# ── Load once at startup ──────────────────────────────────────────────────────

node_service_map, service_node_map = _load_maps()

print("[tool_server] Loading trace data ...")
trace_df  = pd.read_csv(os.path.join(DATA_ROOT, "trace",  "all", "trace_jaeger-span.csv"),
                         dtype={"status_code": str}, low_memory=False)
print("[tool_server] Loading metric data ...")
metric_df = pd.read_csv(os.path.join(DATA_ROOT, "metric", "all", "metrics.csv"),
                         low_memory=False)
log_df    = _load_log_df()

print(f"[tool_server] DATA_ROOT={DATA_ROOT}  spans={len(trace_df)}  "
      f"metrics={len(metric_df)}  logs={len(log_df)}")


# ── Endpoints ─────────────────────────────────────────────────────────────────

def _rows_to_json(rows: pd.DataFrame):
    """Convert DataFrame rows to JSON-safe list (NaN -> None)."""
    return rows.where(pd.notna(rows), None).to_dict(orient="records")


@app.route("/search_span", methods=["GET"])
def search_span():
    span_id = request.args.get("span_id")
    if not span_id:
        return jsonify({"error": "span_id parameter missing"}), 400
    rows = trace_df[trace_df["span_id"] == span_id]
    if rows.empty:
        return jsonify({"message": f"No span found with span_id={span_id}"}), 200
    return jsonify(_rows_to_json(rows))


@app.route("/search_traces", methods=["GET"])
def search_traces():
    parent_span_id = request.args.get("parent_span_id")
    if not parent_span_id:
        return jsonify({"error": "parent_span_id parameter missing"}), 400
    rows = trace_df[trace_df["parent_span"] == parent_span_id]
    if rows.empty:
        return jsonify({"message": f"No child spans for parent_span={parent_span_id}"}), 200
    return jsonify(_rows_to_json(rows))


@app.route("/search_logs", methods=["GET"])
def search_logs():
    service_name  = request.args.get("service_name")
    timestamp_str = request.args.get("timestamp")
    if not service_name:
        return jsonify({"error": "service_name parameter missing"}), 400
    if not timestamp_str:
        return jsonify({"error": "timestamp parameter missing"}), 400
    try:
        ts = int(str(timestamp_str)[:10])
    except ValueError:
        return jsonify({"error": "Invalid timestamp format"}), 400

    filtered = log_df[
        log_df["cmdb_id"].astype(str).str.contains(service_name, na=False) &
        log_df["timestamp"].between(ts - 60, ts + 60)
    ]
    return filtered.to_csv(index=False)


@app.route("/search_fluctuating_metrics", methods=["GET"])
def search_fluctuating_metrics():
    service_name  = request.args.get("service_name")
    timestamp_str = request.args.get("timestamp")
    if not service_name:
        return jsonify({"error": "service_name parameter missing"}), 400
    if not timestamp_str:
        return jsonify({"error": "timestamp parameter missing"}), 400
    try:
        ts = int(str(timestamp_str)[:10])
    except ValueError:
        return jsonify({"error": "Invalid timestamp format"}), 400

    node_id = service_node_map.get(service_name)
    if node_id:
        cond = (
            (
                metric_df["service_name"].str.contains(service_name, case=False, na=False) |
                (
                    (metric_df["node_id"] == node_id) &
                    (metric_df["service_name"].fillna("") == "")
                )
            ) &
            metric_df["timestamp"].between(ts - 1200, ts + 1200)
        )
    else:
        cond = (
            metric_df["service_name"].str.contains(service_name, case=False, na=False) &
            metric_df["timestamp"].between(ts - 1200, ts + 1200)
        )

    baseline_df = metric_df[cond]
    if baseline_df.empty:
        return jsonify({"message": "No matching records found."}), 200

    kpi_dict = {}
    for (kpi, nid, svc), group in baseline_df.groupby(["kpi_name", "node_id", "service_name"]):
        mean_val = group["value"].mean()
        std_val  = group["value"].std()
        window   = group[group["timestamp"].between(ts - 600, ts + 600)]
        if pd.isna(mean_val) or pd.isna(std_val) or std_val == 0:
            continue
        threshold = 3 * std_val
        is_spike  = (
            (window["value"] < mean_val - threshold) |
            (window["value"] > mean_val + threshold)
        ).any()
        if not is_spike:
            continue
        key = kpi
        if not pd.isna(svc) and svc:
            key = f"{svc}.{key}"
        if not pd.isna(nid) and nid:
            key = f"{nid}.{key}"
        kpi_dict[key] = {
            "regular_mean":    round(mean_val, 2),
            "regular_std_dev": round(std_val, 2),
            "current_mean":    round(window["value"].mean(), 2),
            "current_std_dev": round(window["value"].std(), 2),
        }

    if not kpi_dict:
        return jsonify({"message": "No fluctuating metrics found."}), 200

    rows   = [["key", "regular_mean", "regular_std_dev", "current_mean", "current_std_dev"]]
    rows  += [[k] + list(v.values()) for k, v in kpi_dict.items()]
    df_out = pd.DataFrame(rows[1:], columns=rows[0])
    return df_out.to_csv(index=False)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(
        host=config.TOOL_SERVER_HOST,
        port=config.TOOL_SERVER_PORT,
        debug=False,
    )
