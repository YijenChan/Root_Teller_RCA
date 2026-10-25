from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from root_teller.baselines.llm import LLM, InferenceError
from root_teller.baselines.native import MODEL_DEFAULTS, RCLAgent, ThinkFL, TelemetryTools, tool_calls, service_ranking
from root_teller.module3.refinement.text_only import TextOnlyPolicy
from root_teller.module3.structured.adapters import from_synthetic_ledger
from root_teller.module3.evaluate_feedback import run_trajectory
from root_teller.multidataset import rq1_protocol_v2, rq1_three_seed, rq1_sn


def document():
    spans = [{"span_id": "trace:root", "parent_span": "", "trace_id": "trace",
              "service_name": "frontend", "cmdb_id": "frontend", "timestamp": 1000., "duration": 200, "is_error": True},
             {"span_id": "trace:child", "parent_span": "trace:root", "trace_id": "trace",
              "service_name": "payment", "cmdb_id": "payment", "timestamp": 1001., "duration": 100, "is_error": True}]
    return {"services": ["frontend", "payment"], "spans": spans, "observation_start": 1000.,
            "metric_observations": {"payment": [{"metric": "cpu", "regular_mean": 1., "regular_std": 1.,
                "samples": [[1000., 8.], [1001., 10.], [1500., 10000.]]}]},
            "log_lines": {"payment": [{"relative_seconds": 1., "text": "timeout"},
                                       {"relative_seconds": 500., "text": "unrelated"}]}}


def call(name, **arguments):
    return {"content": "Action: " + name + "\nAction Input: " + json.dumps(arguments)}


def test_paper_baseline_model_defaults():
    assert MODEL_DEFAULTS == {"thinkfl": "Zhang-Lingzhe/ThinkFL-Llama3.2-3B", "rclagent": "qwen3.6-plus",
                              "rcacopilot": "gpt-4", "rcagent": "lmsys/vicuna-13b-v1.5-16k"}


@pytest.mark.parametrize("content", [
    '<tool_call>{"name":"search_traces","arguments":{"parent_span_id":"trace:root"}}</tool_call>',
    'Action: search_traces\nAction Input: {"parent_span_id":"trace:root"}',
    '\u273fFUNCTION\u273f: search_traces\n\u273fARGS\u273f: {"parent_span_id":"trace:root"}',
])
def test_native_tool_protocols(content):
    parsed = tool_calls({"content": content})
    assert parsed[0]["function"]["name"] == "search_traces"
    assert json.loads(parsed[0]["function"]["arguments"]) == {"parent_span_id": "trace:root"}


def test_tools_respect_span_parent_and_timestamp():
    tools = TelemetryTools(document())
    assert tools.root()["span_id"] == "trace:root"
    assert tools.call("search_traces", {"parent_span_id": "trace:root"})[0]["span_id"] == "trace:child"
    metrics = tools.call("search_fluctuating_metrics", {"service_name": "payment", "timestamp": 1000})
    assert metrics[0]["current_mean"] == 9.
    logs = tools.call("search_logs", {"service_name": "payment", "timestamp": "1970-01-01T00:16:40Z"})
    assert [r["text"] for r in logs] == ["timeout"]


def test_thinkfl_explores_then_rethinks_before_final_output():
    answers = [call("search_traces", parent_span_id="trace:root"),
               call("print_results", root_causes=[{"service": "frontend"}]),
               call("search_fluctuating_metrics", service_name="payment", timestamp="1001"),
               call("print_results", root_causes=[{"service": "payment"}]),
               call("print_results", root_causes=[{"service": "payment"}, {"service": "frontend"}])]
    class Actor:
        model = MODEL_DEFAULTS["thinkfl"]
        def chat(self, role, messages):
            assert role == "thinkfl"
            return answers.pop(0)
    ranking, diagnostics = ThinkFL(Actor()).predict(document())
    assert ranking == ["payment", "frontend"] and not answers
    assert {r["phase"] for r in diagnostics["trajectory"]} == {"investigate", "rethink", "final"}


def test_rclagent_runs_each_span_and_restores_imports():
    calls = []
    class Actor:
        model = MODEL_DEFAULTS["rclagent"]
        def chat(self, role, messages, tools=None):
            calls.append(copy.deepcopy(messages))
            if tools and tools[0]["function"]["name"] == "print_results":
                return {"content": "", "tool_calls": [{"id": "final", "type": "function", "function": {
                    "name": "print_results", "arguments": json.dumps({"root_causes": ["payment-0", "frontend"]})}}]}
            if tools and messages[-1]["role"] != "tool":
                return {"content": "", "tool_calls": [{"id": "logs", "type": "function", "function": {
                    "name": "search_logs", "arguments": json.dumps({"service_name": "payment", "timestamp": "1001"})}}]}
            return {"content": json.dumps({"is_abnormal": True, "has_error_logs": True})}
    before = {name: sys.modules.get(name) for name in ("config", "llm", "tools_gpt")}
    ranking, _ = RCLAgent(Actor(), max_parallel=2).predict(document())
    assert ranking == ["payment", "frontend"]
    assert len(calls) == 7
    assert sum("Analyse the following span" in m[-1].get("content", "") for m in calls) == 2
    assert before == {name: sys.modules.get(name) for name in before}


def test_model_interface_failure_cannot_become_native_ranking():
    class Broken:
        model = "fixture"
        def chat(self, *args, **kwargs):
            raise InferenceError("fixture failure")
    with pytest.raises(InferenceError):
        ThinkFL(Broken()).predict(document())
    with pytest.raises(InferenceError):
        RCLAgent(Broken()).predict(document())
    with pytest.raises(ValueError):
        service_ranking({"root_causes": ["unknown"]}, ["payment"])


def test_native_chat_replay_has_no_network_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr("root_teller.baselines.llm.requests.post", lambda *a, **k: pytest.fail("network call"))
    actor = LLM(tmp_path, tmp_path / "cache", MODEL_DEFAULTS["thinkfl"], "replay")
    with pytest.raises(InferenceError, match="Cache miss"):
        actor.chat("thinkfl", [{"role": "user", "content": "test"}])
    assert actor.calls == 0


def test_native_chat_records_actual_model_and_replays(tmp_path, monkeypatch):
    bodies = []
    class Response:
        def raise_for_status(self):
            pass
        def json(self):
            return {"model": "ThinkFL", "usage": {"total_tokens": 12},
                    "choices": [{"message": call("print_results", root_causes=[{"service": "payment"}])}]}
    def post(url, **kwargs):
        bodies.append(kwargs["json"])
        return Response()
    monkeypatch.setenv("ROOTTELLER_API_KEY", "fixture")
    monkeypatch.setattr("root_teller.baselines.llm.requests.post", post)
    actor = LLM(tmp_path, tmp_path / "cache", MODEL_DEFAULTS["thinkfl"], "live")
    messages = [{"role": "user", "content": "test"}]
    result = actor.chat("thinkfl", messages)
    replay = LLM(tmp_path, tmp_path / "cache", MODEL_DEFAULTS["thinkfl"], "replay")
    assert replay.chat("thinkfl", messages) == result
    assert bodies[0]["model"] == MODEL_DEFAULTS["thinkfl"]
    assert actor.tokens == 12 and replay.calls == 0 and replay.cache_hits == 1


def initial():
    return from_synthetic_ledger(json.loads((Path(__file__).parent / "fixtures/feedback_ledger.json").read_text(encoding="utf-8")))


def test_text_only_uses_conversation_without_structured_state():
    state = initial()
    before = copy.deepcopy(state.artifact())
    class Actor:
        def __init__(self):
            self.payloads = []
        def complete(self, *, payload, validator, **kwargs):
            self.payloads.append(payload)
            return validator({"ranked_entities": ["S-C", "S-P"]}), {}
    actor = Actor()
    policy = TextOnlyPolicy(state, actor)
    assert not hasattr(policy, "state")
    policy.submit("one", policy.leader(), "challenge")
    assert policy.leader() == "S-C"
    assert actor.payloads[0].keys() == {"conversation"}
    assert "operator rejected" in actor.payloads[0]["conversation"][-1]["content"]
    assert state.artifact() == before
    row = run_trajectory(state, "S-C", strategy="text_only", client=actor)
    assert row["success"] and row["interactions"] == 2
    assert row["execution_mode"] == "text_only_model_feedback"


def test_text_only_without_model_is_failure_not_simulated_gain():
    state = initial()
    row = run_trajectory(state, "S-C", strategy="text_only")
    assert not row["success"] and row["failure"] == "PROCESSING_FAILED"


def test_three_seed_forwards_one_second_preset(monkeypatch):
    commands = []
    monkeypatch.setattr(rq1_three_seed.subprocess, "run", lambda command, **kwargs: commands.append(command))
    monkeypatch.setattr(rq1_three_seed, "AGGREGATION_SECONDS", 1)
    rq1_three_seed.run_protocol(dataset="re2_ob", seed=41, stage="export", workers=1, resume=False)
    assert commands[0][-2:] == ["--aggregation-seconds", "1"]
    assert "aggregation_1s" in str(rq1_three_seed.result_path("re2_ob", 41))


def test_protocol_predicts_sixty_steps_per_window(monkeypatch):
    case = {"bin_seconds": 1, "start_time": 1000., "incident_id": "fixture", "root_cause_service": "payment",
            "services": ["frontend", "payment"]}
    for modality in ("metric", "log", "trace"):
        case[modality + "_x"] = np.ones((2, 120, 1), dtype=np.float32)
        case[modality + "_mask"] = np.ones((2, 120), dtype=bool)
    steps = []
    monkeypatch.setattr(rq1_protocol_v2, "to_torch_case", lambda sliced, *a: {**sliced, "target_index": 1})
    class Model:
        def eval(self):
            pass
        def __call__(self, tensor):
            steps.append(tensor["metric_x"].shape[1])
            return {"localization_probabilities": torch.tensor([.1, .9])}
    result = rq1_protocol_v2.predict(Model(), [case], None, torch.device("cpu"))
    assert steps == [60, 60] and result[0]["rank"] == 1


def test_mismatched_checkpoint_resolution_is_rejected(monkeypatch):
    monkeypatch.setattr(torch, "load", lambda *a, **k: {"feature_config": {"bin_seconds": 60}})
    with pytest.raises(ValueError, match="aggregation intervals differ"):
        rq1_protocol_v2.load_checkpoint(Path("fixture.pt"), {"bin_seconds": 1}, torch.device("cpu"))


def test_sn_resolution_has_separate_cache(monkeypatch):
    monkeypatch.setattr(rq1_sn, "CACHE", rq1_sn.CACHE)
    monkeypatch.setattr(rq1_sn, "BIN_SECONDS", rq1_sn.BIN_SECONDS)
    rq1_sn.configure_intervals(1)
    first = rq1_sn.CACHE
    assert rq1_sn.BIN_SECONDS == 1
    rq1_sn.configure_intervals(60)
    assert first != rq1_sn.CACHE and rq1_sn.BIN_SECONDS == 60
