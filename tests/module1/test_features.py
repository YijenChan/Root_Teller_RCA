from __future__ import annotations

import unittest

from root_teller.module1.config import SERVICES, canonical_service
from root_teller.module1.features import normalize_template


class FeatureHelpersTest(unittest.TestCase):
    def test_service_aliases(self) -> None:
        self.assertEqual(canonical_service("frontendservice"), "frontend")
        self.assertEqual(canonical_service("redis-cart"), "redis")
        self.assertIn("checkoutservice", SERVICES)

    def test_template_normalization(self) -> None:
        left = normalize_template("request 123 failed for 0xabcdef1234")
        right = normalize_template("request 987 failed for 0x9999999999")
        self.assertEqual(left, right)


if __name__ == "__main__":
    unittest.main()
