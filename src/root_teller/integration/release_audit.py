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
        "RE2-OB": (0.717, 0.911, 0.889),
        "RE2-TT": (0.835, 0.899, 0.914),
        "Eadro-SN": (0.500, 0.789, 0.750),
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


def _configuration_matches(root: Path) -> bool:
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


def audit_release(root: Path, output_dir: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    required = [
        root / "README.md",
        root / "pyproject.toml",
        root / "docs/PAPER_CODE_ALIGNMENT.md",
        root / "evaluation/README.md",
        root / "evaluation/rq1/results/paper_table.csv",
        root / "evaluation/rq1/baseline_adapters/README.md",
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
        "paper_configuration_matches": _configuration_matches(root),
        "rq1_table_matches_paper": _paper_table_matches(root),
        "evaluation_release_boundary_matches": evaluations == ["rq1", "rq2"],
        "six_baselines_are_documented": baselines
        == ["eadro", "multisource_rcd", "nezha", "rclagent", "thinkfl", "torai"],
    }
    manifest = {
        str(path.relative_to(root)).replace("\\", "/"): _sha256(path)
        for path in _tracked_files(root)
    }
    result = {
        "schema_version": "root-teller-public-release-audit-1.0",
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
