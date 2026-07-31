from __future__ import annotations

import csv
from pathlib import Path

from root_teller.module1.config import FeatureConfig
from root_teller.module1.model import ModelConfig
from root_teller.module2.config import Module2Config
from root_teller.module3.config import Module3Config
from root_teller.multidataset.rq1_three_seed import SEEDS


ROOT = Path(__file__).resolve().parents[1]


def test_perception_defaults_match_the_paper() -> None:
    features = FeatureConfig()
    model = ModelConfig()
    assert features.bin_seconds == 60
    assert len(features.metric_groups) == 10
    assert features.log_backend == "sbert"
    assert features.log_sbert_dim == 384
    assert features.log_extra_dim == 3
    assert features.trace_dim == 6
    assert model.hidden_dim == 48
    assert model.embedding_dim == 48
    assert model.graph_layers == 2
    assert model.dropout == 0.2
    assert model.fusion_lambda == 0.6
    assert model.use_availability_mask is True


def test_agent_defaults_match_the_paper() -> None:
    collaboration = Module2Config()
    interaction = Module3Config()
    assert collaboration.window_seconds == 60
    assert collaboration.temperature == 0.0
    assert collaboration.max_validation_tasks == 2
    assert collaboration.max_hierarchical_rounds == 2
    assert collaboration.recency_decay == 0.94
    assert collaboration.deterministic_weight == 0.80
    assert collaboration.steward_weight == 0.20
    assert interaction.verifier_budget == 3
    assert interaction.feedback_budget == 20
    assert interaction.feedback_score_decay == 0.90
    assert SEEDS == (41, 42, 43)


def test_rq1_root_teller_rows_match_the_paper() -> None:
    expected = {
        "RE2-OB": (0.717, 0.911, 0.889),
        "RE2-TT": (0.835, 0.899, 0.914),
        "Eadro-SN": (0.500, 0.789, 0.750),
    }
    with (ROOT / "evaluation/rq1/results/paper_table.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))
    actual = {
        row["dataset"]: (
            float(row["A@1"]),
            float(row["A@3"]),
            float(row["Avg@5"]),
        )
        for row in rows
        if row["method"] == "Root-Teller"
    }
    assert actual == expected
