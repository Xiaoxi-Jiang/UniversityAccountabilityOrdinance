#!/usr/bin/env python3
"""Create yearly district trend table from cleaned student housing data."""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys
from collections import defaultdict


def parse_int(value: str) -> int:
    if not value:
        return 0
    try:
        return int(value)
    except ValueError:
        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate yearly trend table for district student housing.")
    parser.add_argument(
        "--student-housing",
        default="data/processed/student_housing_clean.csv",
        help="Processed student housing CSV.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/district_yearly_trend.csv",
        help="Output yearly trend CSV.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = pathlib.Path(args.student_housing)
    output_path = pathlib.Path(args.output)

    if not input_path.exists():
        print(f"Input not found: {input_path}")
        return 1

    with input_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            print("Input CSV has no header row.")
            return 1
        rows = list(reader)

    agg: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {"students": 0, "units": 0, "records": 0})

    for row in rows:
        district = (row.get("district", "") or "UNKNOWN").strip() or "UNKNOWN"
        year = (row.get("year", "") or "UNKNOWN").strip() or "UNKNOWN"
        key = (year, district)

        agg[key]["students"] += parse_int(row.get("student_count", ""))
        agg[key]["units"] += parse_int(row.get("units", ""))
        agg[key]["records"] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        fields = ["year", "district", "records", "students", "units", "students_per_unit"]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()

        for (year, district), values in sorted(agg.items(), key=lambda x: (x[0][0], x[0][1])):
            units = values["units"]
            students = values["students"]
            ratio = round(students / units, 2) if units else 0.0
            writer.writerow(
                {
                    "year": year,
                    "district": district,
                    "records": str(values["records"]),
                    "students": str(students),
                    "units": str(units),
                    "students_per_unit": f"{ratio:.2f}",
                }
            )

    print(f"Wrote yearly trend CSV: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
