from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from root_teller.paths import workspace_root


@dataclass(frozen=True)
class Module3Paths:
    project: Path = field(default_factory=workspace_root)

    @property
    def module2_run(self) -> Path:
        return (
            self.project
            / "runs"
            / "module2_re2ob"
            / "checkpoint3_v2_1_default_clean_replay"
        )

    @property
    def private_labels(self) -> Path:
        return self.module2_run / "evaluation_private.json"

    @property
    def api_config(self) -> Path:
        return self.project / "config" / "API_KEY.txt"

    @property
    def response_cache(self) -> Path:
        return self.project / "cache" / "module3_re2ob" / "llm_responses"

    @property
    def run_root(self) -> Path:
        return self.project / "runs" / "module3_re2ob"


@dataclass(frozen=True)
class Module3Config:
    schema_version: str = "module3-config-1.0"
    prompt_version: str = "rca-interaction-v2"
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    request_timeout_seconds: int = 60
    max_retries: int = 2
    verifier_budget: int = 3
    feedback_budget: int = 20
    feedback_score_decay: float = 0.90
    feedback_granularity: str = "service"
    false_rejection_budgets: tuple[int, ...] = tuple(range(21))

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["false_rejection_budgets"] = list(self.false_rejection_budgets)
        return payload
