from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from root_teller.baselines import classical
from root_teller.module2.config import Module2Config
from root_teller.module2.run import _control, _diagnosis_state


ROOT = Path(__file__).resolve().parents[1]


def state_at_boundary(*, normal=False, exhausted=True):
    config = Module2Config()
    record = {"anomaly_score": .05 if normal else .9, "role_probabilities": {"normal": .9 if normal else .05}}
    pack = {"window": {"activation_order": 0, "window_id": "W00"}, "ranked_candidates": [
        {"entity_id": "A", "candidate_chains": [{"score": 1., "entities": ["A", "B"],
            "evidence_id": "chain", "temporal_status": "consistent"}]}]}
    rmg = SimpleNamespace(windows={"W00": pack}, entity_record=lambda *a: record, evidence_ids=lambda: {"E1"})
    steward = {"selected_entity": "A", "ranked_entities": ["A", "B"], "fused_scores": {"A": .9, "B": .1},
               "claims": [{"entity": "A", "evidence_ids": ["E1"]}], "limitations": []}
    validation = {"verdict": "SUPPORTED", "supporting_evidence_ids": ["E1"],
                  "contradicting_evidence_ids": [], "limitations": []}
    return _diagnosis_state(rmg, steward, validation, config, total_windows=1 if exhausted else 2)


def test_history_exhaustion_does_not_resolve_anomalous_left_boundary():
    state = state_at_boundary()
    assert state["history_exhausted"] and not state["boundary_resolved"]
    assert state["boundary_resolution"] == "unresolved"
    assert _control(state, 0, Module2Config())[0] == "ABSTAIN"


def test_unresolved_boundary_requests_earlier_evidence_when_available():
    state = state_at_boundary(exhausted=False)
    assert state["needs_earlier_evidence"]
    assert _control(state, 0, Module2Config())[0] == "EXPAND"
    assert _control(state, Module2Config().max_expansions, Module2Config())[0] == "ABSTAIN"


def test_supported_normal_baseline_can_resolve_boundary():
    state = state_at_boundary(normal=True)
    assert state["boundary_resolved"] and state["boundary_resolution"] == "observed-normal-baseline"
    assert _control(state, 0, Module2Config())[0] == "CONCLUDE"


def load_sn_adapter():
    path = ROOT / "evaluation/rq1/baseline_adapters/run_eadro_sn.py"
    spec = importlib.util.spec_from_file_location("public_sn_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_eadro_sn_loads_shipped_adapter_without_tick_for_model_construction():
    module = load_sn_adapter()
    module.SERVICES = ("frontend", "payment")
    module.SERVICE_INDEX = {name: i for i, name in enumerate(module.SERVICES)}
    adapter = module.load_eadro_module()
    model = adapter.EadroCompat(256, [(0, 1)]).eval()
    with torch.no_grad():
        detection, localization = model(torch.ones(2, 2, 60, 7), torch.ones(2, 2, 60, 1), torch.ones(2, 2, 256))
    assert detection.shape == localization.shape == (2, 2)
    assert Path(adapter.__file__) == ROOT / "evaluation/rq1/baseline_adapters/eadro.py"


def capture(start, anomaly=False):
    rng = np.random.default_rng(17 + int(anomaly))
    length = 180
    metric = pd.DataFrame({"time": start + np.arange(length), "frontend_cpu": rng.normal(2, 1, length),
                           "payment_cpu": rng.normal(20 if anomaly else 2, 1, length)})
    logs, spans = [], []
    for index in range(12):
        stamp = start + 10 * index
        spans.extend([{"trace": str(index), "id": "root", "parent": "", "service": "frontend", "time": stamp,
                       "duration": 100000., "operation": "checkout", "error": anomaly},
                      {"trace": str(index), "id": "child", "parent": "root", "service": "payment", "time": stamp + .01,
                       "duration": 50000., "operation": "pay", "error": anomaly}])
        logs.append({"service": "payment", "time": stamp + .02,
                     "text": "payment timeout" if anomaly else "payment completed successfully"})
    return classical.Capture(metric, logs, spans, float(start), float(start + length))


def test_causal_reference_boundary_is_constructed_without_fault_annotations():
    data, boundary = classical.causal_inputs([capture(10000)], capture(50000, True))
    assert boundary == 180 and len(data["metric"]) == 360
    for frame in data.values():
        assert frame.time.tolist() == list(range(360))
    assert any("_log-" in name for name in data["logts"].columns)


def test_multisource_receives_metric_and_log_but_no_trace(monkeypatch):
    received = []
    class Method:
        def __enter__(self):
            def run(data, **kwargs):
                received.append(copy.deepcopy(data))
                return {"ranks": ["payment_cpu", "frontend_cpu"]}
            return run
        def __exit__(self, *args):
            pass
    monkeypatch.setattr(classical, "causal_method", lambda *a: Method())
    ranking, details = classical.causal_rank("multisource_rcd", [capture(0)], capture(300, True))
    assert ranking == ["payment", "frontend"]
    assert any("_log-" in name for name in received[0]["metric"].columns)
    assert received[0]["tracets_err"].empty and received[0]["tracets_lat"].empty
    assert details["input_modalities"] == ["metric", "log"]


@pytest.mark.parametrize("method", ["multisource_rcd", "torai"])
def test_actual_upstream_causal_ranker_runs_on_synthetic_telemetry(method):
    pytest.importorskip("causallearn")
    pytest.importorskip("sklearn")
    before = {k: v for k, v in sys.modules.items() if k == "RCAEval" or k.startswith("RCAEval.")}
    ranking, details = classical.causal_rank(method, [capture(0)], capture(300, True))
    assert ranking and len(ranking) == len(set(ranking))
    assert set(ranking) <= {"frontend", "payment"}
    assert details["upstream_ranks"]
    assert before == {k: v for k, v in sys.modules.items() if k == "RCAEval" or k.startswith("RCAEval.")}


def test_multisource_actual_ranking_is_invariant_to_trace_records():
    pytest.importorskip("causallearn")
    production = capture(300, True)
    first, _ = classical.causal_rank("multisource_rcd", [capture(0)], production)
    production.spans = []
    second, _ = classical.causal_rank("multisource_rcd", [capture(0)], production)
    assert first == second


def test_actual_nezha_event_graph_and_pattern_ranking_runs_without_padding():
    pytest.importorskip("drain3")
    pytest.importorskip("more_itertools")
    pytest.importorskip("matplotlib")
    before = {name: sys.modules.get(name) for name in ("log", "alarm", "data_integrate", "pattern_ranker")}
    ranking, details = classical.nezha_rank([capture(0)], capture(300, True))
    assert ranking == ["payment"]
    assert len(ranking) == len(set(ranking)) and set(ranking) <= {"frontend", "payment"}
    assert details["reference_graphs"] == details["production_graphs"] == 12
    assert details["input_modalities"] == ["log", "trace"]
    assert ranking == list(dict.fromkeys(r["pod"].rsplit("-", 2)[0] for r in details["ranked_patterns"] if r["pod"].rsplit("-", 2)[0] in {"frontend", "payment"}))
    assert before == {name: sys.modules.get(name) for name in before}


def test_empty_upstream_ranking_is_not_padded(monkeypatch):
    class Empty:
        def __enter__(self):
            return lambda *a, **k: {"ranks": []}
        def __exit__(self, *args):
            pass
    monkeypatch.setattr(classical, "causal_method", lambda *a: Empty())
    assert classical.causal_rank("torai", [capture(0)], capture(300, True))[0] == []


@pytest.mark.parametrize("module", ["root_teller.baselines.classical_run", "root_teller.baselines.run"])
def test_runner_help_does_not_require_unrelated_models_or_data(module):
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    result = subprocess.run([sys.executable, "-m", module, "--help"], cwd=ROOT, env=environment,
                            text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert "--fold" in result.stdout and "--mode" in result.stdout


def test_re2_capture_reader_preserves_parentage_and_observation_scope(tmp_path):
    metric = pd.DataFrame({"time": [1000., 1001., 1002.], "paymentservice_cpu": [1., 2., 3.], "gke-node_cpu": [9., 9., 9.]})
    metric.to_csv(tmp_path / "simple_metrics.csv", index=False)
    pd.DataFrame([{"traceID": "T", "spanID": "root", "parentSpanID": "", "serviceName": "frontend", "startTime": 1000e6,
                   "duration": 2000., "operationName": "checkout", "statusCode": 0},
                  {"traceID": "T", "spanID": "child", "parentSpanID": "root", "serviceName": "paymentservice", "startTime": 1001e6,
                   "duration": 1000., "operationName": "pay", "statusCode": 500}]).to_csv(tmp_path / "traces.csv", index=False)
    pd.DataFrame([{"container_name": "paymentservice", "timestamp": 1001e9, "message": "timeout"}]).to_csv(tmp_path / "logs.csv", index=False)
    value = classical.read_capture(tmp_path, "re2_ob", 1001., 1003.)
    assert value.start == 1001. and value.end == 1003.
    assert [s["id"] for s in value.spans] == ["child"] and value.spans[0]["parent"] == "root"
    assert value.spans[0]["error"] and len(value.logs) == 1
    assert "gke-node_cpu" not in value.metrics and "gke-node" not in value.services


def test_classical_preparation_writes_no_fabricated_metrics(tmp_path, monkeypatch):
    from root_teller.baselines import classical_run
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    (manifests / "re2_ob_nested_folds.json").write_text("[]", encoding="utf-8")
    spec = SimpleNamespace(role="test", opaque_id="public-fixture", directory=tmp_path,
                           start=None, end=None, incident_id="private-id", target="payment")
    monkeypatch.setattr(classical_run, "load_specs", lambda *a: [spec])
    monkeypatch.setattr(classical_run, "read_capture", lambda *a: capture(0))
    monkeypatch.setattr(classical_run, "causal_rank", lambda *a: pytest.fail("inference during preparation"))
    destination = tmp_path / "prepared"
    monkeypatch.setattr(sys, "argv", ["runner", "--method", "torai", "--dataset", "re2_ob", "--workspace", str(tmp_path),
                                      "--manifests", str(manifests), "--mode", "prepare", "--output", str(destination)])
    classical_run.main()
    result = json.loads((destination / "summary.json").read_text(encoding="utf-8"))
    assert result["status"] == "prepared_inputs" and result["metrics"] is None
    assert result["cases"][0]["opaque_id"] == "public-fixture"
