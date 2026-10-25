"""Repository-owned documentation contracts; no network or model inference."""
from __future__ import annotations

import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

import pytest

from root_teller.module1.config import FeatureConfig
from root_teller.module2.config import Module2Config
from root_teller.module3.session import feedback_config, refinement_config


ROOT = Path(__file__).resolve().parents[1]
DOCS = [
    "README.md", "docs/audit/README.md",
    "system/README.md", "system/ARCHITECTURE.md", "system/VALIDATION.md",
    "evaluation/README.md", "evaluation/UPLOAD_CHECKLIST.md",
    "evaluation/rq1/README.md", "evaluation/rq2/README.md",
    "evaluation/feedback/README.md",
    "evaluation/rq1/baseline_adapters/THIRD_PARTY_STATUS.md",
    "baselines/README.md",
    "baselines/rcacopilot/README.md", "baselines/rcagent/README.md",
]


@pytest.mark.parametrize("relative", DOCS)
def test_repository_documentation_links_resolve(relative):
    source = ROOT / relative
    text = source.read_text(encoding="utf-8")
    for destination in re.findall(r"\[[^\]\n]+\]\(([^)\n]+)\)", text):
        url = urlsplit(destination)
        if url.scheme or url.netloc:
            continue
        target = (source.parent / unquote(url.path)).resolve() if url.path else source
        assert target.exists(), f"{relative}: missing {destination}"
        if url.fragment and target.suffix == ".md":
            headings = re.findall(r"^#{1,6}\s+(.+)$", target.read_text(encoding="utf-8"), re.M)
            anchors = {re.sub(r"[^\w\- ]", "", h.lower()).replace(" ", "-") for h in headings}
            assert unquote(url.fragment) in anchors, f"{relative}: missing heading {destination}"


def test_revision_reference_matches_runtime_presets():
    config = json.loads((ROOT / "configs/revision2_method.json").read_text(encoding="utf-8"))
    observation, cross, feedback = (config[k] for k in ("observation", "cross_window", "feedback"))
    module2, core = Module2Config(), feedback_config()
    assert observation["aggregation_seconds"] == FeatureConfig.paper_intervals().bin_seconds
    assert observation["legacy_released_checkpoint_aggregation_seconds"] == FeatureConfig().bin_seconds
    assert observation["rca_window_seconds"] == module2.window_seconds
    assert cross == {"enabled": module2.cross_window_enabled, "max_neighbors": module2.cross_window_neighbors,
        "max_chain_pairs": module2.cross_window_chain_pairs, "anomaly_threshold": module2.cross_window_anomaly_threshold}
    for key in ["max_items", "max_tasks", "max_windows_per_task", "gamma"]:
        assert feedback[key] == core[key]
    assert feedback["interaction_budget"] == core["submission_budget"]
    assert feedback["model_rank_weight"] == refinement_config()["scoring"]["model_rank_weight"]
    assert config["runtime"]["model"] == module2.model
