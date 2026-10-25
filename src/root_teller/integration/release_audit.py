from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any


IGNORED_PARTS = {
    "__pycache__",
    ".pytest_cache",
    ".git",
    ".venv",
    "build",
    "dist",
    # Independently maintained material retained in the public repository;
    # it is not part of the Root-Teller paper artifact or its checksum scope.
    "office_mini_storm_dataset",
}
IGNORED_FILES = {"release_audit.json", "checksums.json"}
TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".toml",
    ".json",
    ".csv",
    ".txt",
    ".gitignore",
    ".html",
    ".css",
    ".js",
    ".ps1",
    ".bat",
    ".sh",
    ".yml",
    ".yaml",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tracked_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and not (set(path.relative_to(root).parts) & IGNORED_PARTS)
        and not any(part.endswith(".egg-info") for part in path.relative_to(root).parts)
        and path.name not in IGNORED_FILES
        and path.suffix.lower() != ".pyc"
    )


def _paper_table_matches(root: Path) -> bool:
    expected = {
        "RE2-OB": (0.719, 0.911, 0.889),
        "RE2-TT": (0.833, 0.900, 0.914),
        "Eadro-SN": (0.500, 0.787, 0.750),
    }
    table = root / "evaluation/rq1/results/paper_table.csv"
    if not table.exists():
        return False
    try:
        with table.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    except (OSError, ValueError):
        return False
    actual = {
        row["dataset"]: (
            float(row["A@1"]),
            float(row["A@3"]),
            float(row["Avg@5"]),
        )
        for row in rows
        if row["method"] == "Root-Teller"
    }
    return actual == expected


def _historical_configuration_preserved(root: Path) -> bool:
    perception = json.loads(
        (root / "configs/module1_re2ob_frozen_2026-07-24.json").read_text(
            encoding="utf-8"
        )
    )
    collaboration = json.loads(
        (root / "configs/module2_re2ob_default_frozen_v2_2026-07-24.json").read_text(
            encoding="utf-8"
        )
    )
    feedback = json.loads(
        (root / "configs/module3_re2ob_feedback_frozen_2026-07-24.json").read_text(
            encoding="utf-8"
        )
    )
    return all(
        (
            perception["features"]["bin_seconds"] == 60,
            perception["features"]["log_embedding_dimension"] == 384,
            perception["features"]["log_count_features"] == 3,
            perception["features"]["trace_features"] == 6,
            perception["model"]["hidden_dimension"] == 48,
            perception["model"]["graph_layers"] == 2,
            perception["model"]["fusion_lambda"] == 0.6,
            perception["training"]["seeds"] == [41, 42, 43],
            collaboration["window_policy"]["window_seconds"] == 60,
            collaboration["ranking"]["recency_decay"] == 0.94,
            collaboration["ranking"]["deterministic_weight"] == 0.8,
            collaboration["ranking"]["steward_weight"] == 0.2,
            collaboration["hierarchical_rca_loop"]["max_validation_tasks_per_round"] == 2,
            collaboration["hierarchical_rca_loop"]["max_hierarchical_rounds"] == 2,
            feedback["feedback_budget"] == 20,
            feedback["feedback_score_decay"] == 0.9,
        )
    )


def _revision_configuration_matches(root: Path) -> bool:
    """Check the declared source contract, not empirical/model equivalence."""
    try:
        config = json.loads((root / "configs/revision2_method.json").read_text(encoding="utf-8"))
        observation, cross, feedback = (config[k] for k in ("observation", "cross_window", "feedback"))
        return all((
            observation["aggregation_seconds"] == 1,
            observation["rca_window_seconds"] == 60,
            observation["legacy_released_checkpoint_aggregation_seconds"] == 60,
            cross["enabled"] is True, cross["max_neighbors"] == 3,
            cross["max_chain_pairs"] == 8, cross["anomaly_threshold"] == 0.5,
            feedback["max_items"] == 3, feedback["max_tasks"] == 2,
            feedback["max_windows_per_task"] == 2, feedback["interaction_budget"] == 20,
            feedback["gamma"] == 0.9, feedback["model_rank_weight"] == 0.2,
            feedback["offline_model_rank_weight"] == 0.0,
            config["status"] == "method_source_alignment_not_empirical_validation",
        ))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def audit_release(root: Path, output_dir: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    required = [
        root / "README.md",
        root / "pyproject.toml",
        root / "configs/revision2_method.json",
        root / "evaluation/README.md",
        root / "evaluation/rq1/results/paper_table.csv",
        root / "baselines/README.md",
        root / "baselines/eadro/README.md",
        root / "src/root_teller/module1/model.py",
        root / "src/root_teller/module2/rmg.py",
        root / "src/root_teller/module3/feedback.py",
        root / "tools/telemetry_unavailability/prepare.py",
        root / "system/root_teller_system/app.py",
    ]
    missing = [str(path.relative_to(root)) for path in required if not path.exists()]
    forbidden_payloads = [
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
        and (
            path.name == "API_KEY.txt"
            or path.suffix.lower() == ".npz"
            or "runtime" in path.relative_to(root).parts
        )
    ]
    secret_pattern = re.compile("s" + "k-" + r"[A-Za-z0-9]{20,}")
    cjk_pattern = re.compile(r"[\u3400-\u9fff]")
    local_path_patterns = (
        re.compile("F:" + re.escape("\\") + "RootTeller", re.IGNORECASE),
        re.compile("C:" + re.escape("\\") + "Users" + re.escape("\\") + "PC", re.IGNORECASE),
    )
    secret_hits: list[str] = []
    cjk_hits: list[str] = []
    local_path_hits: list[str] = []
    for path in _tracked_files(root):
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name != ".gitignore":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        relative = str(path.relative_to(root))
        if secret_pattern.search(text):
            secret_hits.append(relative)
        if cjk_pattern.search(text):
            cjk_hits.append(relative)
        if any(pattern.search(text) for pattern in local_path_patterns):
            local_path_hits.append(relative)

    baselines = sorted(path.name for path in (root / "baselines").iterdir() if path.is_dir())
    evaluations = sorted(path.name for path in (root / "evaluation").iterdir() if path.is_dir())
    license_present = any(
        (root / name).is_file()
        for name in ("LICENSE", "LICENSE.md", "LICENSE.txt")
    )
    checks = {
        "required_files_present": not missing,
        "repository_license_present": license_present,
        "no_credentials_or_runtime_payloads": not forbidden_payloads,
        "no_api_key_literal": not secret_hits,
        "english_only_text": not cjk_hits,
        "no_author_machine_paths": not local_path_hits,
        "historical_configuration_preserved": _historical_configuration_preserved(root),
        "revision_reference_contract_matches": _revision_configuration_matches(root),
        "rq1_table_matches_paper": _paper_table_matches(root),
        "evaluation_release_boundary_matches": evaluations == ["feedback", "rq1", "rq2"],
        "eight_baselines_are_documented": baselines
        == ["eadro", "multisource_rcd", "nezha", "rcacopilot", "rcagent", "rclagent", "thinkfl", "torai"],
        "revised_feedback_schema_present": (root / "src/root_teller/module3/structured/feedback_schema.json").is_file(),
        "revised_feedback_entrypoint_present": (root / "src/root_teller/module3/session.py").is_file(),
    }
    manifest = {
        str(path.relative_to(root)).replace("\\", "/"): _sha256(path)
        for path in _tracked_files(root)
    }
    result = {
        "schema_version": "root-teller-public-release-audit-2.0",
        "passed": all(checks.values()),
        "checks": checks,
        "details": {
            "missing": missing,
            "forbidden_payloads": forbidden_payloads,
            "secret_hits": secret_hits,
            "cjk_hits": cjk_hits,
            "local_path_hits": local_path_hits,
            "tracked_files": len(manifest),
        },
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "release_audit.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        (output_dir / "checksums.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a public Root-Teller release.")
    parser.add_argument("root", type=Path, nargs="?", default=Path.cwd())
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = audit_release(args.root, args.output_dir)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
