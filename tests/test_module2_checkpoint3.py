from __future__ import annotations

import json
import os
from pathlib import Path
import unittest

from root_teller.module2.contracts import (
    validate_investigator_response,
    validate_steward_response,
)
from root_teller.module2.rmg import RCAMemoryGraph
from root_teller.module2.config import Module2Config
from root_teller.module2.run import run_case_default


WORKSPACE = Path(os.environ.get("ROOTTELLER_WORKSPACE", Path.cwd())).resolve()
PACK_ROOT = WORKSPACE / "cache/module2_re2ob/window_evidence_packs/validation"


def _packs() -> list[dict[str, object]]:
    case_root = next(path for path in PACK_ROOT.iterdir() if path.is_dir())
    return [
        json.loads((case_root / name).read_text(encoding="utf-8"))
        for name in ("W00.json", "W01.json")
    ]


@unittest.skipUnless(PACK_ROOT.exists(), "requires a prepared Root-Teller workspace")
class Module2Checkpoint3Tests(unittest.TestCase):
    def test_public_window_pack_has_no_private_label_or_raw_incident_id(self) -> None:
        pack = _packs()[0]
        serialized = json.dumps(pack).lower()
        self.assertTrue(pack["incident_id"].startswith("re2ob-"))
        self.assertNotIn("root_cause_service", serialized)
        self.assertNotIn("fault_type", serialized)
        self.assertNotIn("inject_time", serialized)
        self.assertNotIn("/", pack["incident_id"])

    def test_rmg_accepts_only_adjacent_windows_and_preserves_immutable_pack(self) -> None:
        first, second = _packs()
        rmg = RCAMemoryGraph(first["incident_id"])
        rmg.add_window(first)
        rmg.add_window(second)
        self.assertEqual(set(rmg.windows), {"W00", "W01"})
        self.assertTrue(any(item["type"] == "entity-continuity" for item in rmg.relations))
        changed = json.loads(json.dumps(first))
        changed["quality_notes"]["tampered"] = True
        with self.assertRaisesRegex(ValueError, "immutable"):
            rmg.add_window(changed)

    def test_steward_contract_rejects_unknown_shape(self) -> None:
        with self.assertRaises(ValueError):
            validate_steward_response({"selected_entity": "frontend"})

    def test_investigator_contract_normalizes_scalar_limitation(self) -> None:
        result = validate_investigator_response(
            {
                "verdict": "SUPPORTED",
                "supporting_evidence_ids": ["E1"],
                "contradicting_evidence_ids": [],
                "verified_relations": [],
                "limitations": "score is not causal proof",
            }
        )
        self.assertEqual(result["limitations"], ["score is not causal proof"])

    def test_default_protocol_exhausts_history_and_synthesizes_after_validation(self) -> None:
        case_root = next(path for path in PACK_ROOT.iterdir() if path.is_dir())
        result, rmg = run_case_default(
            case_root, Module2Config(), steward_agent=None, investigator=None
        )
        self.assertEqual(result["protocol"], "default-exhaustive")
        self.assertEqual(result["default_exhaustive"]["activated_windows"], 24)
        self.assertGreaterEqual(len(result["hierarchical_rca_loop"]), 1)
        first_cycle = result["hierarchical_rca_loop"][0]
        self.assertTrue(first_cycle["inspection_tasks"])
        self.assertIn("steward_after_inspection", first_cycle)
        self.assertTrue(rmg["validation_results"])
        self.assertTrue(rmg["diagnostic_abstracts"])
        self.assertGreaterEqual(len(rmg["synthesis_history"]), 2)
        self.assertTrue(
            any(
                relation["type"] == "semantic-association"
                and relation["final_support_eligible"] is False
                for relation in rmg["relations"]
            )
        )


if __name__ == "__main__":
    unittest.main()
