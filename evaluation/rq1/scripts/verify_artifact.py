"""Fast integrity checks for the public RQ1 evaluation package."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RQ1 = ROOT / "evaluation" / "rq1"


def main() -> None:
    with (RQ1 / "results" / "paper_table.csv").open(newline="", encoding="utf-8") as stream:
        table = list(csv.DictReader(stream))
    assert len(table) == 21
    assert {row["method"] for row in table} == {
        "Eadro", "Nezha", "Multi-source RCD", "TORAI", "ThinkFL", "RCLAgent", "Root-Teller"
    }
    assert all(0.0 <= float(row[key]) <= 1.0 for row in table for key in ("A@1", "A@3", "Avg@5"))

    audit = json.loads((RQ1 / "manifests" / "protocol_audit.json").read_text(encoding="utf-8"))
    assert audit["status"] == "PASS"
    expected = {
        "re2_ob": (90, {"test": 30, "train": 45, "validation": 15}, 3),
        "re2_tt": (90, {"test": 30, "train": 45, "validation": 15}, 3),
        "eadro_sn": (36, {"test": 9, "train": 18, "validation": 9}, 4),
    }
    for dataset, (cases, roles, folds) in expected.items():
        item = audit["datasets"][dataset]
        assert item["unique_cases"] == cases
        for fold in range(folds):
            for role, count in roles.items():
                assert item["fold_role_counts"][f"{fold}:{role}"] == count
        conflicts = item.get("service_fault_family_role_conflicts", item.get("capture_role_conflicts"))
        assert conflicts == []

    for path in (RQ1 / "baseline_adapters").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        compile(text, str(path), "exec")
        assert "F:\\RootTeller" not in text
        assert "API_KEY.txt" not in text

    link_pattern = re.compile(r"\[[^]]+\]\(([^)]+)\)")
    for markdown in (
        ROOT / "README.md",
        ROOT / "evaluation" / "README.md",
        RQ1 / "README.md",
        RQ1 / "baseline_adapters" / "README.md",
    ):
        for target in link_pattern.findall(markdown.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "#")):
                continue
            resolved = (markdown.parent / target.split("#", 1)[0]).resolve()
            assert resolved.exists(), f"broken relative link in {markdown}: {target}"
    print("RQ1 artifact verification: PASS")


if __name__ == "__main__":
    main()
