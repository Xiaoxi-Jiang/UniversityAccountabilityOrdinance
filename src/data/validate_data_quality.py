#!/usr/bin/env python3
"""Data quality checks for cleaned and modeled outputs."""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate core datasets for required fields and null rates.")
    parser.add_argument("--student-clean", default="data/processed/student_housing_clean.csv")
    parser.add_argument("--property-risk", default="data/processed/property_risk_model.csv")
    return parser.parse_args()


def check_required(path: pathlib.Path, required: list[str]) -> list[str]:
    issues: list[str] = []
    if not path.exists():
        return [f"Missing file: {path}"]

    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            return [f"No header row: {path}"]

        missing = [f for f in required if f not in reader.fieldnames]
        if missing:
            issues.append(f"{path}: missing required columns: {', '.join(missing)}")

        rows = list(reader)
        if not rows:
            issues.append(f"{path}: no data rows")
            return issues

        for col in required:
            empty = sum(1 for r in rows if not (r.get(col, "") or "").strip())
            ratio = empty / len(rows)
            if ratio > 0.2:
                issues.append(f"{path}: column '{col}' empty ratio too high ({ratio:.1%})")

    return issues


def main() -> int:
    args = parse_args()
    student_issues = check_required(pathlib.Path(args.student_clean), ["address", "district", "year", "student_count"])
    risk_issues = check_required(pathlib.Path(args.property_risk), ["property_key", "address", "district", "risk_score"])

    issues = student_issues + risk_issues
    if issues:
        print("Data quality checks failed:")
        for i in issues:
            print(f"- {i}")
        return 1

    print("Data quality checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
