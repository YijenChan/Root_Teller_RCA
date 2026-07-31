from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from root_teller.paths import workspace_root


@dataclass(frozen=True)
class Paths:
    project: Path = field(default_factory=workspace_root)

    @property
    def clean_root(self) -> Path:
        return (
            self.project
            / "dataset"
            / "RCAEval RE"
            / "RE2"
            / "RE2-OB"
            / "RE2-OB"
        )

    @property
    def corrupted_root(self) -> Path:
        return (
            self.project
            / "dataset"
            / "dataset_corrupted"
            / "checkpoint1_RE2-OB_2026-07-24"
        )

    @property
    def manifest_dir(self) -> Path:
        return self.project / "artifact" / "telemetry_unavailability" / "manifests"

    @property
    def cache_root(self) -> Path:
        return self.project / "cache" / "module1_re2ob"


@dataclass(frozen=True)
class FeatureConfig:
    bin_seconds: int = 60
    metric_groups: tuple[str, ...] = (
        "cpu",
        "memory",
        "disk",
        "network",
        "socket",
        "request",
        "error",
        "latency",
        "bytes",
        "other",
    )
    log_hash_dim: int = 32
    # The paper configuration uses all-MiniLM-L6-v2. The lightweight hash
    # backend remains available only as an explicitly selected smoke-test mode.
    log_backend: str = "sbert"
    log_sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    log_sbert_dim: int = 384
    log_extra_dim: int = 3
    trace_dim: int = 6
    template_parser_version: str = "drain_style_message_regex_v2"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def log_content_dim(self) -> int:
        return self.log_sbert_dim if self.log_backend == "sbert" else self.log_hash_dim


CONDITIONS = (
    "CLEAN",
    "GMO_METRIC",
    "GMO_LOG",
    "GMO_TRACE",
    "IAMI_METRIC",
    "IAMI_LOG",
    "IAMI_TRACE",
)

SERVICES = (
    "adservice",
    "cartservice",
    "checkoutservice",
    "currencyservice",
    "emailservice",
    "frontend",
    "paymentservice",
    "productcatalogservice",
    "recommendationservice",
    "redis",
    "shippingservice",
)

SERVICE_INDEX = {service: index for index, service in enumerate(SERVICES)}


def canonical_service(value: object) -> str:
    service = str(value).strip().lower().replace("_", "-")
    aliases = {
        "frontendservice": "frontend",
        "frontend-external": "frontend",
        "redis-cart": "redis",
        "redis-cartservice": "redis",
    }
    service = aliases.get(service, service)
    if service.endswith("-service"):
        candidate = service.replace("-", "")
        if candidate in SERVICE_INDEX:
            service = candidate
    return service
