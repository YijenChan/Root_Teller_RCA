from __future__ import annotations

import unittest

import torch

from root_teller.module1.data import modality_quality
from root_teller.module1.evaluate import opaque_case_id


class ModelContractTest(unittest.TestCase):
    def test_unavailable_quality_is_zero(self) -> None:
        mask = torch.zeros((2, 4), dtype=torch.bool)
        quality = modality_quality(mask)
        self.assertTrue(torch.equal(quality, torch.zeros_like(quality)))

    def test_complete_quality_is_one(self) -> None:
        mask = torch.ones((2, 4), dtype=torch.bool)
        quality = modality_quality(mask)
        self.assertTrue(torch.allclose(quality, torch.ones_like(quality)))

    def test_downstream_case_id_is_opaque_and_stable(self) -> None:
        raw = "checkoutservice_cpu/3"
        opaque = opaque_case_id(raw)
        self.assertEqual(opaque, opaque_case_id(raw))
        self.assertTrue(opaque.startswith("re2ob-"))
        self.assertNotIn("checkoutservice", opaque)
        self.assertNotIn("cpu", opaque)


if __name__ == "__main__":
    unittest.main()
