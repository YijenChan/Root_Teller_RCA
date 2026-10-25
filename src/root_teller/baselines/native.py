from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import re
import sys
import threading
import types

import pandas as pd

from .data import canonical, diagnostic_document, _json_capture, _healthy_metrics
from .llm import InferenceError


MODEL_DEFAULTS = {
    "thinkfl": "Zhang-Lingzhe/ThinkFL-Llama3.2-3B",
    "rclagent": "qwen3.6-plus",
    "rcacopilot": "gpt-4",
    "rcagent": "lmsys/vicuna-13b-v1.5-16k",
}
_IMPORT_LOCK = threading.RLock()


def load_document(spec, workspace, cache):
    document = copy.deepcopy(diagnostic_document(spec, workspace, cache))
    records = []
    metric_frames = []
    if spec.dataset.startswith("re2_"):
        metric_times = pd.read_csv(spec.directory / "simple_metrics.csv", usecols=["time"])["time"]
        start, end = float(metric_times.min()), float(metric_times.max()) + 1
        frame = pd.read_csv(spec.directory / "simple_metrics.csv", low_memory=False)
        times = pd.to_numeric(frame["time"], errors="coerce")
        for column in frame:
            if column != "time" and "_" in column:
                service, metric = column.split("_", 1)
                values = pd.to_numeric(frame[column], errors="coerce")
                reference = values[(times >= start) & (times < start + 60)].dropna()
                metric_frames.append((canonical(service, spec.dataset), metric, times, values,
                                      float(reference.mean()), float(reference.std(ddof=0))))
        frame = pd.read_csv(spec.directory / "traces.csv", low_memory=False)
        for row in frame.itertuples(index=False):
            code = float(row.statusCode) if pd.notna(row.statusCode) else 0
            records.append({"trace": str(row.traceID), "id": str(row.spanID),
                            "parent": str(row.parentSpanID) if pd.notna(row.parentSpanID) else "",
                            "service": canonical(row.serviceName, spec.dataset),
                            "timestamp": float(row.startTime) / 1e6,
                            "duration": float(row.duration), "error": code >= 400 or code == 2})
    else:
        start, end = float(spec.start), float(spec.end)
        references = _healthy_metrics(str(workspace))
        for path in sorted((spec.directory / "metrics").glob("*.csv")):
            service = canonical(path.stem, spec.dataset)
            frame = pd.read_csv(path)
            times = pd.to_numeric(frame["timestamp"], errors="coerce")
            for column in frame:
                if (service, column) in references:
                    mean, std = references[(service, column)]
                    metric_frames.append((service, column, times, pd.to_numeric(frame[column], errors="coerce"), mean, std))
        for group in _json_capture(str(spec.directory / "spans.json")):
            processes = group.get("processes", {})
            for row in group.get("spans", []):
                parents = [r["spanID"] for r in row.get("references", []) if r.get("refType") == "CHILD_OF"]
                tags = {r["key"]: r.get("value") for r in row.get("tags", [])}
                records.append({"trace": str(row["traceID"]), "id": str(row["spanID"]),
                                "parent": str(parents[0]) if parents else "",
                                "service": canonical(processes.get(row.get("processID"), {}).get("serviceName", "unknown"), spec.dataset),
                                "timestamp": float(row["startTime"]) / 1e6 - 8 * 3600,
                                "duration": float(row["duration"]), "error": bool(tags.get("error", False))})
    document["observation_start"] = start
    document["metric_observations"] = {}
    for service, metric, times, values, mean, std in metric_frames:
        if service not in document["services"] or pd.isna(mean) or pd.isna(std):
            continue
        keep = (times >= start) & (times < end) & values.notna() & times.notna()
        samples = [[float(t), float(v)] for t, v in zip(times[keep], values[keep])
                   if float("-inf") < v < float("inf")]
        document["metric_observations"].setdefault(service, []).append({
            "metric": metric, "regular_mean": mean, "regular_std": std, "samples": samples})
    document["spans"] = [{"span_id": r["trace"] + ":" + r["id"],
                           "parent_span": r["trace"] + ":" + r["parent"] if r["parent"] else "",
                           "trace_id": r["trace"], "service_name": r["service"], "cmdb_id": r["service"],
                           "timestamp": r["timestamp"], "duration": r["duration"], "is_error": r["error"]}
                          for r in records if start <= r["timestamp"] < end and r["service"] in document["services"]]
    return document


def tool_calls(message):
    calls = message.get("tool_calls") or []
    if calls:
        return calls
    content = message.get("content") or ""
    result = []
    for match in re.finditer(r"<tool_call>\s*", content):
        try:
            value, _ = json.JSONDecoder().raw_decode(content[match.end():])
        except (ValueError, TypeError):
            continue
        if isinstance(value, dict) and "name" in value:
            arguments = value.get("arguments", {})
            result.append({"id": f"call-{len(result)}", "type": "function", "function": {
                "name": value["name"], "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments)}})
    if not result:
        name = None
        for line in content.splitlines():
            line = line.strip()
            if line.startswith(("Action: ", "\u273fFUNCTION\u273f: ")):
                name = line.split(": ", 1)[1].strip()
            elif name and line.startswith(("Action Input: ", "\u273fARGS\u273f: ")):
                raw = line.split(": ", 1)[1].strip().strip("`")
                arguments, _ = json.JSONDecoder().raw_decode(raw)
                result.append({"id": f"call-{len(result)}", "type": "function", "function": {
                    "name": name, "arguments": json.dumps(arguments)}})
    return result


def service_ranking(value, services):
    items = value.get("root_causes", value.get("ranking", []))
    if not isinstance(items, list):
        raise ValueError("Expected an ordered root-cause list")
    ranking = []
    for item in items:
        if isinstance(item, dict):
            item = next((item[k] for k in ("service", "pod", "node", "name") if k in item), None)
        if not isinstance(item, str):
            continue
        name = item.strip().lower()
        candidates = [s for s in services if name == s or name.startswith(s + "-")]
        if candidates:
            name = max(candidates, key=len)
            if name not in ranking:
                ranking.append(name)
    if not ranking:
        raise ValueError("No reported candidate maps to an observed service")
    return ranking


class TelemetryTools:
    def __init__(self, document):
        self.document = document
        self.spans = {s["span_id"]: s for s in document["spans"]}
        self.children = {}
        for span in self.spans.values():
            self.children.setdefault(span["parent_span"], []).append(span)

    def root(self):
        roots = [s for s in self.spans.values() if s["parent_span"] not in self.spans]
        if not roots:
            raise ValueError("No observed root span is available")
        errors = {}
        for span in self.spans.values():
            errors[span["trace_id"]] = errors.get(span["trace_id"], 0) + int(span["is_error"])
        return max(roots, key=lambda s: (errors[s["trace_id"]], s["duration"], s["timestamp"]))

    def call(self, name, arguments):
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if name == "search_traces":
            return self.children.get(arguments.get("parent_span_id"), [])
        if name == "search_span":
            span = self.spans.get(arguments.get("span_id"))
            return [span] if span else []
        service = arguments.get("service_name")
        if service not in self.document["services"]:
            return {"error": "Unknown observed service"}
        raw_timestamp = arguments.get("timestamp", self.root()["timestamp"])
        try:
            timestamp = float(raw_timestamp)
        except (TypeError, ValueError):
            stamp = pd.Timestamp(raw_timestamp)
            timestamp = (stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp).timestamp()
        if name == "search_fluctuating_metrics":
            result = []
            for metric in self.document.get("metric_observations", {}).get(service, []):
                values = [v for t, v in metric["samples"] if abs(t - timestamp) <= 60]
                if values:
                    current = sum(values) / len(values)
                    denominator = max(metric["regular_std"], abs(metric["regular_mean"]) * .01, 1e-6)
                    z = abs(current - metric["regular_mean"]) / denominator
                    if z >= 3:
                        result.append({"metric": metric["metric"], "current_mean": current,
                                       "regular_mean": metric["regular_mean"], "shift_z": z})
            return sorted(result, key=lambda value: -value["shift_z"])
        if name == "search_logs":
            start = self.document["observation_start"]
            return [r for r in self.document["log_lines"].get(service, [])
                    if abs(start + r["relative_seconds"] - timestamp) <= 60]
        return {"error": "Unknown telemetry tool"}


class ThinkFL:
    def __init__(self, llm, max_rounds=40):
        self.llm, self.max_rounds = llm, max_rounds

    def predict(self, document):
        tools = TelemetryTools(document)
        definitions = [
            {"name": "search_traces", "input_schema": {"type": "object", "properties": {
                "parent_span_id": {"type": "string"}}, "required": ["parent_span_id"]}},
            {"name": "search_fluctuating_metrics", "input_schema": {"type": "object", "properties": {
                "service_name": {"type": "string"}, "timestamp": {"type": "string"}}, "required": ["service_name", "timestamp"]}},
            {"name": "print_results", "input_schema": {"type": "object", "properties": {
                "root_causes": {"type": "array", "items": {"type": "object", "properties": {
                    "service": {"type": "string"}, "reason": {"type": "string"}}}}}, "required": ["root_causes"]}},
        ]
        messages = [{"role": "system", "content":
            "You are a software operations engineer. Diagnose the root cause using recursive trace exploration and fluctuating metrics. "
            "Use Action: tool_name followed by Action Input: JSON arguments when invoking a tool. "
            "Finish with print_results. Available tools: " + json.dumps(definitions)},
            {"role": "user", "content": json.dumps({"root_span": tools.root(), "services": document["services"]})}]
        trace = []
        phase = "investigate"
        for _ in range(self.max_rounds):
            message = self.llm.chat("thinkfl", messages)
            calls = tool_calls(message)
            messages.append({"role": "assistant", "content": message.get("content") or json.dumps(calls)})
            if not calls:
                raise InferenceError("ThinkFL did not invoke a diagnostic tool or print_results")
            for call in calls:
                function = call["function"]
                arguments = function["arguments"]
                arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
                if function["name"] == "print_results":
                    ranking = service_ranking(arguments, document["services"])
                    trace.append({"phase": phase, "tool": "print_results", "arguments": arguments})
                    if phase == "final":
                        return ranking, {"trajectory": trace, "model": self.llm.model}
                    messages.append({"role": "user", "content":
                        "Inspect deeper descendants with search_traces and verify metric evidence before confirming the tentative root cause."
                        if phase == "investigate" else
                        "Reconsider the complete investigation and return the final ordered root causes with print_results."})
                    phase = "rethink" if phase == "investigate" else "final"
                    break
                result = tools.call(function["name"], arguments)
                trace.append({"phase": phase, "tool": function["name"], "arguments": arguments, "result": result})
                messages.append({"role": "user", "content": json.dumps({"tool": function["name"], "result": result})})
        raise InferenceError("ThinkFL investigation budget exhausted")


class RCLAgent:
    def __init__(self, llm, source=None, max_parallel=4):
        self.llm = llm
        self.source = Path(source) if source else Path(__file__).resolve().parents[3] / "baselines" / "rclagent"
        self.max_parallel = max_parallel

    def predict(self, document):
        tools = TelemetryTools(document)
        with _IMPORT_LOCK:
            names = ("config", "llm", "tools_gpt")
            previous = {name: sys.modules.get(name) for name in names}
            config = types.ModuleType("config")
            config.TOOL_SERVER_URL = ""
            config.MAX_AGENT_PARALLEL = self.max_parallel
            config.DATASET_TYPE = "re2ob"
            config.DATA_ROOT = ""
            config.RESULT_SUB_DIR = "result"
            config.get_known_services = lambda: document["services"]
            client = types.ModuleType("llm")
            failures = []

            def chat_api(messages, tools=None):
                try:
                    message = self.llm.chat("rclagent", messages, tools)
                except Exception as error:
                    failures.append(type(error).__name__)
                    raise
                return message.get("content") or "", tool_calls(message)

            client.chat_api = chat_api
            try:
                sys.modules["config"], sys.modules["llm"] = config, client
                for name in ("tools_gpt", "coordinator"):
                    spec = importlib.util.spec_from_file_location(name, self.source / f"{name}.py")
                    if spec is None or spec.loader is None:
                        raise ImportError("RCLAgent source is unavailable")
                    module = importlib.util.module_from_spec(spec)
                    if name == "tools_gpt":
                        sys.modules[name] = module
                    spec.loader.exec_module(module)
                module.SYSTEM_PROMPT = "You are a root-cause localization agent. Observed services: " + json.dumps(document["services"]) + module._COMMON_GUIDELINES
                module._fetch_span = lambda span_id: tools.spans[span_id]
                module._query_children = lambda span_id: tools.children.get(span_id, [])
                module._get = lambda name, args: json.dumps(tools.call(name, args))
                graph = module.build_trace_graph(tools.root()["span_id"])
                result = module.inspect_trace(graph, max_parallel=self.max_parallel)
                if failures:
                    raise InferenceError("RCLAgent span investigation encountered a model-interface failure")
                return service_ranking(result, document["services"]), {"result": result, "model": self.llm.model}
            finally:
                for name, old in previous.items():
                    if old is None:
                        sys.modules.pop(name, None)
                    else:
                        sys.modules[name] = old
