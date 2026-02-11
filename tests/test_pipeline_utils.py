from __future__ import annotations

import datetime as dt
import unittest

from src.common.pipeline_utils import (
    jaccard_similarity,
    make_property_key,
    normalize_text,
    parse_date_any,
    time_decay_weight,
)


class TestPipelineUtils(unittest.TestCase):
    def test_normalize_text(self) -> None:
        self.assertEqual(normalize_text("123 Example Street Apt #2"), "123 example st 2")

    def test_property_key_stable(self) -> None:
        a = make_property_key("123 Example St", "District 1")
        b = make_property_key("123 example street", "District 1")
        self.assertEqual(a, b)

    def test_jaccard_similarity(self) -> None:
        score = jaccard_similarity("123 example st", "123 example street")
        self.assertGreaterEqual(score, 0.9)

    def test_parse_date_any(self) -> None:
        self.assertEqual(parse_date_any("2025-01-15"), dt.date(2025, 1, 15))
        self.assertEqual(parse_date_any("01/15/2025"), dt.date(2025, 1, 15))

    def test_time_decay_weight(self) -> None:
        old_date = dt.date(2020, 1, 1)
        recent_date = dt.date(2025, 1, 1)
        today = dt.date(2026, 1, 1)
        self.assertLess(time_decay_weight(old_date, today=today), time_decay_weight(recent_date, today=today))


if __name__ == "__main__":
    unittest.main()
