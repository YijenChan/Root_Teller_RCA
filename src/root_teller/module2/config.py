from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from root_teller.paths import workspace_root


@dataclass(frozen=True)
class Module2Paths:
    project: Path = field(default_factory=workspace_root)

    @property
    def checkpoint(self) -> Path:
        return (
            self.project
            / "runs"
            / "module1_re2ob"
            / "final_refit_seed20260724"
            / "checkpoint.pt"
        )

    @property
    def api_config(self) -> Path:
        return self.project / "config" / "API_KEY.txt"

    @property
    def window_pack_root(self) -> Path:
        return self.project / "cache" / "module2_re2ob" / "window_evidence_packs"

    @property
    def response_cache(self) -> Path:
        return self.project / "cache" / "module2_re2ob" / "llm_responses"

    @property
    def run_root(self) -> Path:
        return self.project / "runs" / "module2_re2ob"


@dataclass(frozen=True)
class Module2Config:
    schema_version: str = "module2-config-2.0"
    prompt_version: str = "crt-re2ob-v2"
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    request_timeout_seconds: int = 60
    max_retries: int = 2
    window_seconds: int = 60
    max_expansions: int = 23
    top_k: int = 5
    max_validation_tasks: int = 2
    max_hierarchical_rounds: int = 2
    min_leading_score: float = 0.18
    ambiguity_margin: float = 0.035
    normal_anomaly_threshold: float = 0.20
    recency_decay: float = 0.94
    steward_weight: float = 0.20
    deterministic_weight: float = 0.80

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
