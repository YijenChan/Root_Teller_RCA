from __future__ import annotations

import copy
import numpy as np
import pytest

from root_teller.module1.config import FeatureConfig
from root_teller.module2.window_export import _slice_case, _window_count
from root_teller.module2.rmg import RCAMemoryGraph
from root_teller.module2.cross_window import CrossWindowPolicy


def pack(index):
    wid = f"W{index:02d}"
    entities = [{"entity_id": e, "evidence_id": f"{wid}-{e}", "diagnostic_role": "root",
                 "localization_probability": 0.6 if e == "A" else 0.4,
                 "role_probabilities": {"root": 0.6, "propagation": 0.3, "normal": 0.1},
                 "anomaly_score": 0.9, "onset_bin": 1} for e in ["A", "B"]]
    candidates = [{**r, "rank": i + 1, "candidate_chains": [
        {"evidence_id": f"{wid}-chain-{i}", "entities": ["A", "B"], "temporal_status": "consistent"}]
        } for i, r in enumerate(entities)]
    return {"incident_id": "public-synthetic", "window": {"window_id": wid,
            "activation_order": index, "start_time": 600 - index * 60, "end_time": 660 - index * 60},
            "entity_evidence": entities, "ranked_candidates": candidates,
            "dependency_edges": [["A", "B"]], "quality_notes": {"unavailable_modalities": []}}


def test_non_adjacent_links_preserve_provenance_without_causal_confirmation():
    graph = RCAMemoryGraph("public-synthetic", cross_window_policy=CrossWindowPolicy(max_neighbors=1))
    for i in range(4):
        graph.add_window(pack(i))
    links = [r for r in graph.relations if r.get("link_scope") == "non-adjacent"]
    assert links
    for relation in links:
        assert relation["causal_support_eligible"] is False
        assert relation["final_support_eligible"] is False
        assert set(relation["source_pack_hashes"]) <= set(graph.windows)
        assert abs(int(relation["source_window"][1:]) - int(relation["destination_window"][1:])) > 1
    before = copy.deepcopy(graph.relations)
    graph.add_window(pack(3))
    assert graph.relations == before


def test_history_access_remains_adjacent_even_when_links_are_non_adjacent():
    graph = RCAMemoryGraph("public-synthetic")
    graph.add_window(pack(0))
    with pytest.raises(ValueError, match="adjacent earlier"):
        graph.add_window(pack(2))


def test_legacy_cross_window_policy_is_explicitly_available():
    graph = RCAMemoryGraph("public-synthetic", cross_window_policy=CrossWindowPolicy(enabled=False))
    for i in range(3):
        graph.add_window(pack(i))
    assert not any(r.get("link_scope") == "non-adjacent" for r in graph.relations)


@pytest.mark.parametrize("interval,length,expected_steps", [(1, 120, 60), (60, 2, 1)])
def test_window_export_preserves_within_window_resolution(interval, length, expected_steps):
    case = {"bin_seconds": interval, "start_time": 1000.0}
    for modality in ["metric", "log", "trace"]:
        case[modality + "_x"] = np.arange(length * 2).reshape(1, length, 2).astype("float32")
        case[modality + "_mask"] = np.ones((1, length), dtype=bool)
    case["metric_mask"][0, 1] = False
    assert _window_count(case) == 2
    first, second = _slice_case(case, 0), _slice_case(case, 1)
    assert first["metric_x"].shape[1] == second["metric_x"].shape[1] == expected_steps
    assert second["start_time"] == 1060.0 and second["window_seconds"] == 60
    assert first["metric_x"].shape[1] + second["metric_x"].shape[1] == length


def test_fine_resolution_preset_does_not_relabel_legacy_configuration():
    assert FeatureConfig.paper_intervals().bin_seconds == 1
    assert FeatureConfig().bin_seconds == 60
