"""
coordinator.py – RCLAgent multi-agent coordinator.

Each span in an anomalous trace is assigned a *Dedicated Agent*. Agents are
organised recursively along the trace-graph topology and run in parallel via
an AgentPool. Every agent performs two phases:

    1. self_state_verification – bounded local context, invokes log and
       metric tools to gather evidence for its own span.
    2. consolidation           – synthesises its own evidence with the
       structured reports propagated up from its child agents.

Per-span evidence is recorded in a Global Evidence Graph. The root agent
produces a Root-Level Diagnosis Report, which is combined with the Global
Evidence Graph by the Diagnosis Synthesizer to produce the final ranked list
of root cause candidates.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

import config
from llm import chat_api
from tools_gpt import (
    search_logs_function,
    search_fluctuating_metrics_function,
    print_result_function,
)

BASE_URL = config.TOOL_SERVER_URL


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _pack_params(d) -> str:
    if isinstance(d, str):
        d = json.loads(d)
    return "&".join(f"{k.strip()}={str(v).strip()}" for k, v in d.items())


def _get(path: str, params: dict) -> str:
    url = f"{BASE_URL}/{path}?{_pack_params(params)}"
    try:
        r = requests.get(url, timeout=15)
        return r.content.decode("utf-8")
    except Exception as exc:
        return f"[tool error] {exc}"


def _query_children(parent_span_id: str) -> list:
    url = f"{BASE_URL}/search_traces"
    try:
        r    = requests.get(url, params={"parent_span_id": parent_span_id}, timeout=10)
        text = r.text.replace("NaN", "null").replace("Infinity", '"Infinity"')
        data = json.loads(text)
        return data if isinstance(data, list) else []
    except Exception as exc:
        print(f"[warn] query_children({parent_span_id}): {exc}")
        return []


def _fetch_span(span_id: str) -> dict:
    url = f"{BASE_URL}/search_span"
    try:
        r    = requests.get(url, params={"span_id": span_id}, timeout=10)
        text = r.text.replace("NaN", "null").replace("Infinity", '"Infinity"')
        data = json.loads(text)
        if isinstance(data, list) and data:
            return data[0]
    except Exception as exc:
        print(f"[warn] fetch_span({span_id}): {exc}")
    return {"span_id": span_id}


# ── Global Evidence Graph ─────────────────────────────────────────────────────

class GlobalEvidenceGraph:
    """
    Collects self_evidence from every Dedicated Agent.
    Provides a compact summary for the root-level synthesis step.
    """

    def __init__(self):
        self._nodes: list = []

    def record(self, span_id: str, service_name: str, evidence: str):
        self._nodes.append(
            {"span_id": span_id, "service_name": service_name, "evidence": evidence}
        )

    def summary(self) -> str:
        return json.dumps(self._nodes, ensure_ascii=False, indent=2)

    def abnormal_nodes(self) -> list:
        result = []
        for node in self._nodes:
            ev = node.get("evidence", "")
            if isinstance(ev, str) and ('"is_abnormal": true' in ev.lower() or "abnormal" in ev.lower()):
                result.append(node)
        return result


# ── Agent pool ────────────────────────────────────────────────────────────────

class AgentPool:
    def __init__(self, max_parallel: int = config.MAX_AGENT_PARALLEL):
        self._executor = ThreadPoolExecutor(max_workers=max_parallel)

    def submit(self, fn, *args, **kwargs):
        return self._executor.submit(fn, *args, **kwargs)

    def shutdown(self):
        self._executor.shutdown(wait=True)


# ── System prompt (dataset-aware) ──────────────────────────────────────────────

_SYSTEM_PROMPTS = {
    "aiops2022": """
You are a Root Cause Localization (RCL) agent in a microservice system.
A user-reported failure has occurred. Your task is to identify the DEEPEST
ROOT CAUSE SERVICE that initiated the failure chain.

The system is a HipsterShop microservice application. Known services:
  adservice, cartservice, checkoutservice, currencyservice, emailservice,
  frontend, paymentservice, productcatalogservice, recommendationservice,
  shippingservice
Each service has pods named like <service>-0, <service>-1, <service>-2,
and a secondary variant <service>2-0. Nodes are named node-1 through node-6.

IMPORTANT: Use the EXACT names above (e.g. "paymentservice" not "payment-service").
""",
    "nezha": """
You are a Root Cause Localization (RCL) agent in a microservice system.
A user-reported failure has occurred. Your task is to identify the ROOT CAUSE
SERVICE that initiated the failure chain.

The system is a TrainTicket microservice application. Known services:
  ts-gateway-service, ts-auth-service, ts-basic-service, ts-config-service,
  ts-contacts-service, ts-delivery-service, ts-execute-service, ts-food-service,
  ts-inside-payment-service, ts-order-service, ts-order-other-service,
  ts-payment-service, ts-preserve-service, ts-preserve-other-service,
  ts-price-service, ts-route-service, ts-seat-service, ts-security-service,
  ts-station-service, ts-station-food-service, ts-train-service,
  ts-train-food-service, ts-travel-service, ts-travel2-service,
  ts-user-service, ts-verification-code-service, ts-assurance-service
Pods have K8s names like ts-contacts-service-866bd68c97-xcqfx.
When identifying root causes, use the SERVICE name (e.g. "ts-contacts-service"),
not the full pod name.

Use the available trace, metric, and log evidence to infer the most likely root-cause service. Do not rely on dataset-specific service priors; rank candidates only according to the evidence observed in the current failure instance.
""",
    "re2ob": """
You are a Root Cause Localization (RCL) agent in a microservice system.
A user-reported failure has occurred. Your task is to identify the DEEPEST
ROOT CAUSE SERVICE that initiated the failure chain.

The system is an Online Boutique (HipsterShop) microservice application. Known services:
  adservice, cartservice, checkoutservice, currencyservice, emailservice,
  frontend, frontendservice, paymentservice, productcatalogservice,
  recommendationservice, shippingservice, redis
Use the exact service names as they appear in trace data.
""",
}

_COMMON_GUIDELINES = """
Guidelines:
- Always assume the trace contains a real fault.
- ERROR/EXCEPTION logs are the #1 indicator of a root cause. A service with
  error logs should always be ranked above one with only latency anomalies.
- Fluctuating metrics (CPU, memory spikes) are the #2 indicator.
- High latency alone is a WEAK signal — it usually means victim, not cause.
- Do NOT blame the frontend/gateway unless evidence clearly points there.
- Return ONLY valid JSON when asked for structured output; no markdown fences.
"""

SYSTEM_PROMPT = _SYSTEM_PROMPTS.get(config.DATASET_TYPE, _SYSTEM_PROMPTS["aiops2022"]) + _COMMON_GUIDELINES


# ── Dedicated Agent ───────────────────────────────────────────────────────────

class DedicatedAgent:
    """
    One agent per span.  Analyses its own span in bounded context, then
    consolidates downstream evidences from child agents.
    """

    def __init__(self, span_id: str, raw: dict, pool: AgentPool, geg: GlobalEvidenceGraph):
        self.span_id   = span_id
        self.raw       = raw
        self.pool      = pool
        self.geg       = geg
        self.children  = []
        self.self_evidence: str = ""

    def add_child(self, child):
        self.children.append(child)

    # ── step 1: self-state verification ──────────────────────────────────────

    def self_state_verification(self) -> str:
        prompt = f"""Analyse the following span for self-contained anomalies.

Span data:
{json.dumps(self.raw, indent=2, ensure_ascii=False, default=str)}

Steps:
1. Call search_logs with the service name and timestamp from this span.
   Pay special attention to ERROR/EXCEPTION severity logs — these are the
   strongest indicators of a root cause.
2. Call search_fluctuating_metrics with the same service name and timestamp.
   Look for metrics where current_mean significantly exceeds regular_mean.
3. Determine whether THIS span shows anomaly symptoms.

IMPORTANT evidence ranking:
- ERROR/EXCEPTION logs → strong anomaly signal (set has_error_logs=true)
- Metric deviations (current_mean >> regular_mean) → moderate signal
- High latency alone → WEAK signal (may just be a victim of downstream issues)

Use the tools iteratively.  Once you have enough evidence produce a final
JSON conclusion (no markdown):
{{
  "span_id": "...",
  "service_name": "...",
  "is_abnormal": true/false,
  "has_error_logs": true/false,
  "has_metric_anomaly": true/false,
  "key_symptoms": "brief summary or null",
  "hypothesis": "why it might be faulty or not"
}}"""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ]
        tools      = [search_logs_function, search_fluctuating_metrics_function]
        tool_names = {t["function"]["name"] for t in tools}

        seen_tool_requests = set()
        while True:
            content, tool_calls = chat_api(messages, tools=tools)

            if not tool_calls:
                if content.strip():
                    return content.strip()
                break

            # Stop only if the model repeats the exact same tool request, which
            # indicates a protocol loop rather than useful additional evidence.
            request_signature = tuple(
                (tc.get("function", {}).get("name", ""), tc.get("function", {}).get("arguments", ""))
                for tc in tool_calls
            )
            if request_signature in seen_tool_requests:
                break
            seen_tool_requests.add(request_signature)

            # ── correct OpenAI multi-turn tool-call protocol ──
            messages.append({
                "role":       "assistant",
                "content":    content or "",
                "tool_calls": tool_calls,
            })
            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                try:
                    fn_args = json.loads(tc["function"]["arguments"])
                except (json.JSONDecodeError, TypeError):
                    fn_args = {}
                tool_result = _get(fn_name, fn_args) if fn_name in tool_names else f"Unknown tool: {fn_name}"
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content":      tool_result,
                })

        # Fallback: ask for structured conclusion after tool rounds.
        fallback = f"""Based on all retrieved evidence above, produce a final
JSON conclusion for this span (no markdown):
{{
  "span_id": "{self.span_id}",
  "service_name": "{self.raw.get('cmdb_id', self.raw.get('service_name', 'unknown'))}",
  "is_abnormal": true/false,
  "key_symptoms": "brief summary or null",
  "hypothesis": "..."
}}"""
        messages.append({"role": "user", "content": fallback})
        final_content, _ = chat_api(messages, tools=None)
        return final_content.strip()

    # ── step 2: consolidation (called after all children have finished) ──────

    def consolidate(self, downstream: list) -> str:
        consolidation = f"""You are analysing span: {json.dumps(self.raw, ensure_ascii=False, default=str)}

Your own self-evidence:
{self.self_evidence}

Downstream evidences from {len(downstream)} child span(s):
{json.dumps(downstream, indent=2, ensure_ascii=False)}

Task: Synthesise these into a LOCAL root-cause hypothesis for your parent.
CRITICAL REASONING RULES:
- A child/descendant with ERROR LOGS is almost certainly the root cause.
  Propagate that service upward with high confidence.
- If multiple children are abnormal, prefer the one with ERROR LOGS over
  one with only latency or metric anomalies.
- If a child reports a root cause from even deeper (with error logs),
  propagate THAT upward.
- Only blame the current span if NO downstream child has error logs.
- High latency in the current span is usually CAUSED BY a slow child —
  this is NOT sufficient evidence to blame the current span.
- Output JSON only (no markdown):
{{
  "span_id": "...",
  "service_name": "...",
  "local_root_cause": "service name",
  "has_error_logs": true/false,
  "reason": "...",
  "confidence": 0.0
}}"""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": consolidation},
        ]
        content, _ = chat_api(messages, tools=None)
        return content


# ── Trace graph construction ──────────────────────────────────────────────────

def build_trace_graph(root_span_id: str) -> dict:
    root_raw = _fetch_span(root_span_id)
    visited = set()

    def dfs(span_id: str, raw: dict) -> dict:
        node = {"span_id": span_id, "raw": raw, "children": []}
        if span_id in visited:
            return node
        visited.add(span_id)
        seen_ids = set()
        for child_row in _query_children(span_id):
            child_id = child_row.get("span_id")
            if not child_id or child_id in seen_ids:
                continue
            seen_ids.add(child_id)
            node["children"].append(dfs(child_id, child_row))
        return node

    return dfs(root_span_id, root_raw)


def build_agent_tree(trace_node: dict, pool: AgentPool, geg: GlobalEvidenceGraph) -> DedicatedAgent:
    span_id = trace_node.get("span_id", "unknown")
    raw     = trace_node.get("raw", {})
    agent   = DedicatedAgent(span_id, raw, pool, geg)
    for child in trace_node.get("children", []):
        agent.add_child(build_agent_tree(child, pool, geg))
    return agent


# ── Service name normalisation ─────────────────────────────────────────────────

# Service base names — loaded from config for the active dataset.
_KNOWN_SERVICES = config.get_known_services()

def _normalise_name(name: str) -> str:
    """
    Fix common LLM hallucinations like 'payment-service' → 'paymentservice',
    'checkout-service' → 'checkoutservice', etc.
    """
    low = name.lower().strip()
    # Already valid?
    base = low.rsplit("-", 1)[0] if low[-1:].isdigit() else low
    if base in _KNOWN_SERVICES:
        return low
    # Try removing hyphens: "payment-service-0" → "paymentservice-0"
    dehyphen = low.replace("-service", "service").replace("-cart", "-cart")
    base2 = dehyphen.rsplit("-", 1)[0] if dehyphen[-1:].isdigit() else dehyphen
    if base2 in _KNOWN_SERVICES:
        return dehyphen
    # Try without any hyphens in the service part
    parts = low.split("-")
    if len(parts) >= 2 and parts[-1].isdigit():
        svc = "".join(parts[:-1])
        if svc in _KNOWN_SERVICES:
            return f"{svc}-{parts[-1]}"
    elif len(parts) >= 2:
        svc = "".join(parts)
        if svc in _KNOWN_SERVICES:
            return svc
    return low


def _normalise_results(result: dict) -> dict:
    """Normalise all root_causes names and deduplicate."""
    rcs = result.get("root_causes", [])
    seen = set()
    normalised = []
    for rc in rcs:
        n = _normalise_name(rc)
        if n not in seen:
            seen.add(n)
            normalised.append(n)
    result["root_causes"] = normalised
    return result


# ── Top-level diagnosis ───────────────────────────────────────────────────────

def _collect_agents_by_depth(agent: "DedicatedAgent", depth: int = 0, result: dict = None) -> dict:
    """Collect all agents grouped by their depth in the tree (for BFS execution)."""
    if result is None:
        result = {}
    result.setdefault(depth, []).append(agent)
    for child in agent.children:
        _collect_agents_by_depth(child, depth + 1, result)
    return result


def inspect_trace(trace_graph: dict, max_parallel: int = config.MAX_AGENT_PARALLEL) -> dict:
    geg  = GlobalEvidenceGraph()
    pool = AgentPool(max_parallel)

    root_agent = build_agent_tree(trace_graph, pool, geg)

    # ── Bottom-up BFS execution (avoids recursive ThreadPoolExecutor deadlock) ──
    # Phase 1: Run self_state_verification for ALL agents in parallel (any order).
    depth_map   = _collect_agents_by_depth(root_agent)
    all_agents  = [a for depth in sorted(depth_map) for a in depth_map[depth]]

    futures = {pool.submit(a.self_state_verification): a for a in all_agents}
    for f in as_completed(futures):
        agent = futures[f]
        try:
            agent.self_evidence = f.result()
        except Exception as exc:
            agent.self_evidence = f'{{"error": "{exc}"}}'
        service_name = agent.raw.get("cmdb_id", agent.raw.get("service_name", agent.span_id))
        agent.geg.record(agent.span_id, service_name, agent.self_evidence)

    # Phase 2: Consolidate bottom-up, one depth level at a time.
    # Leaves (max depth) have no children — their consolidation is just self_evidence.
    max_depth = max(depth_map.keys())
    # Store consolidation results keyed by agent id.
    consolidation_results = {}

    for depth in range(max_depth, -1, -1):
        agents_at_depth = depth_map[depth]
        futures_c = {}
        for agent in agents_at_depth:
            downstream = [consolidation_results[id(c)] for c in agent.children
                          if id(c) in consolidation_results and consolidation_results[id(c)]]
            futures_c[pool.submit(agent.consolidate, downstream)] = agent
        for f in as_completed(futures_c):
            agent = futures_c[f]
            try:
                consolidation_results[id(agent)] = f.result()
            except Exception as exc:
                consolidation_results[id(agent)] = f'{{"error": "{exc}"}}'

    root_level_report = consolidation_results.get(id(root_agent), "")
    pool.shutdown()

    # Build a compact evidence summary focusing on error logs
    error_log_services = []
    metric_anomaly_services = []
    for node in geg._nodes:
        ev = node.get("evidence", "")
        if isinstance(ev, str):
            if "has_error_logs" in ev and '"has_error_logs": true' in ev.lower():
                error_log_services.append(node["service_name"])
            elif '"is_abnormal": true' in ev.lower():
                metric_anomaly_services.append(node["service_name"])

    evidence_hint = ""
    if error_log_services:
        evidence_hint = f"\n⚠️ Services with ERROR LOGS (strongest evidence): {', '.join(set(error_log_services))}"
    if metric_anomaly_services:
        evidence_hint += f"\n⚡ Services with metric anomalies only: {', '.join(set(metric_anomaly_services))}"

    final_prompt = f"""You are the ROOT AGENT producing the FINAL diagnosis.

Root-Level Diagnosis Report (from recursive agent tree):
{root_level_report}

Global Evidence Graph ({len(geg._nodes)} span evidences collected):
{geg.summary()}
{evidence_hint}

Goal: identify the root-cause service.

Ranking instructions:
1. Rank candidates according to the evidence observed in the current trace, metrics, and logs.
2. Prefer components whose abnormal evidence can causally explain upstream symptoms.
3. Consider all three granularity levels: service, pod, node.
4. You MUST call the `print_results` function with at least 10 candidates.
   Do not output anything else."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": final_prompt},
    ]
    content, tool_calls = chat_api(messages, tools=[print_result_function])

    result = None
    if tool_calls:
        args = tool_calls[0]["function"]["arguments"]
        try:
            result = json.loads(args) if isinstance(args, str) else args
            result = _normalise_results(result)
        except Exception:
            pass

    # Fallback: extract JSON from plain text.
    if result is None:
        try:
            start = content.find("{")
            end   = content.rfind("}") + 1
            if start >= 0 and end > start:
                result = _normalise_results(json.loads(content[start:end]))
        except Exception:
            pass

    if result is None:
        result = {
            "root_causes": ["unknown"],
            "note":        "model did not call print_results",
            "raw_content": content,
        }

    return result


# ── Evidence-based re-ranking (Nezha post-processing) ────────────────────────

import re as _re

def _svc_base(name: str) -> str:
    """Extract service base name from pod name or service name.
    Requires hash segments to contain at least one digit to avoid matching
    service-name components like 'travel' or 'service'.
    """
    low = name.lower()
    m = _re.match(r'^(.+?)-([a-z0-9]{5,})-([a-z0-9]{4,})$', low)
    if m and any(c.isdigit() for c in m.group(2)):
        return m.group(1)
    if low[-1:].isdigit():
        return low.rsplit("-", 1)[0]
    return low


def _compute_network_overhead(trace_graph: dict) -> dict:
    """
    Compute per-service network overhead from trace graph.
    For each cross-service call (parent_service → child_service),
    overhead = parent_call_span_duration - child_span_duration.
    A large overhead (>500ms) indicates network_delay on the child service.
    Returns dict: service_base_name -> max overhead in microseconds.
    """
    overhead: dict = {}

    def _walk(node):
        raw = node.get("raw", {})
        parent_svc = _svc_base(str(raw.get("cmdb_id", raw.get("service_name", ""))))
        try:
            parent_dur = int(raw.get("duration", 0) or 0)
        except (ValueError, TypeError):
            parent_dur = 0

        for child in node.get("children", []):
            child_raw = child.get("raw", {})
            child_svc = _svc_base(str(child_raw.get("cmdb_id", child_raw.get("service_name", ""))))
            if not child_svc or child_svc == parent_svc:
                _walk(child)
                continue
            try:
                child_dur = int(child_raw.get("duration", 0) or 0)
            except (ValueError, TypeError):
                child_dur = 0
            net = parent_dur - child_dur
            if net > 0:
                if child_svc not in overhead or overhead[child_svc] < net:
                    overhead[child_svc] = net
            _walk(child)

    _walk(trace_graph)
    return overhead


def _collect_trace_services(trace_graph: dict) -> set:
    """Collect all unique service base names from the trace graph."""
    services = set()
    def _walk(node):
        raw = node.get("raw", {})
        svc = raw.get("cmdb_id", raw.get("service_name", ""))
        if svc:
            services.add(_svc_base(str(svc)))
        for child in node.get("children", []):
            _walk(child)
    _walk(trace_graph)
    return services


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_all(data_root: str = config.DATA_ROOT):
    error_file = os.path.join(data_root, "error_traces.txt")
    result_dir = os.path.join(data_root, config.RESULT_SUB_DIR)
    os.makedirs(result_dir, exist_ok=True)

    with open(error_file) as f:
        lines = f.readlines()

    total = len(lines) - 1
    for i, line in enumerate(lines[1:], start=1):
        parts = line.split()
        if len(parts) < 4:
            continue
        root_span_id = parts[3].strip()
        if not root_span_id:
            continue

        out_path = os.path.join(result_dir, f"conversation_trace_{i}.txt")
        if os.path.exists(out_path):
            print(f"[{i}/{total}] already done, skipping.")
            continue

        print(f"[{i}/{total}] span_id={root_span_id}")
        try:
            trace_graph       = build_trace_graph(root_span_id)
            trace_graph["error_trace_line"] = line.strip()
            result            = inspect_trace(trace_graph)
            with open(out_path, "w") as fw:
                json.dump(result, fw, ensure_ascii=False, indent=2)
            print(f"  root_causes[:3] = {result.get('root_causes', [])[:3]}")
        except Exception as exc:
            print(f"  [error] {exc}")


if __name__ == "__main__":
    run_all()
