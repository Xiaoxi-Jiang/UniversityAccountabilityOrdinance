from __future__ import annotations

import csv
import subprocess
import unittest
from pathlib import Path


class TestRiskPipeline(unittest.TestCase):
    def test_integrated_risk_model_outputs(self) -> None:
        root = Path(__file__).resolve().parents[1]

        subprocess.run(
            [
                "python3",
                "src/data/prepare_student_housing.py",
                "--input",
                "data/raw/student_housing_template.csv",
                "--output",
                "data/processed/student_housing_clean.csv",
            ],
            cwd=root,
            check=True,
        )
        subprocess.run(["python3", "src/data/build_property_registry.py"], cwd=root, check=True)
        subprocess.run(["python3", "src/analysis/integrated_risk_model.py"], cwd=root, check=True)

        out = root / "data/processed/landlord_risk_model.csv"
        self.assertTrue(out.exists())

        with out.open("r", encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))

        self.assertGreaterEqual(len(rows), 1)
        self.assertIn("risk_score", rows[0])


if __name__ == "__main__":
    unittest.main()
