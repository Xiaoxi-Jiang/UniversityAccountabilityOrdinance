#!/usr/bin/env python3
"""Generate district-level summary outputs for student housing accountability."""

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
    parser = argparse.ArgumentParser(description="Create summary tables and markdown report.")
    parser.add_argument(
        "--student-housing",
        default="data/processed/student_housing_clean.csv",
        help="Processed student housing CSV path.",
    )
    parser.add_argument(
        "--output-csv",
        default="data/processed/district_summary.csv",
        help="Output district summary CSV.",
    )
    parser.add_argument(
        "--output-md",
        default="reports/student_housing_summary.md",
        help="Output markdown report.",
    )
    return parser.parse_args()


def compute_district_summary(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    agg: dict[str, dict[str, int]] = defaultdict(lambda: {"records": 0, "students": 0, "units": 0})

    for row in rows:
        district = row.get("district", "").strip() or "UNKNOWN"
        agg[district]["records"] += 1
        agg[district]["students"] += parse_int(row.get("student_count", ""))
        agg[district]["units"] += parse_int(row.get("units", ""))

    summary: list[dict[str, str]] = []
    for district, totals in sorted(agg.items()):
        units = totals["units"]
        students = totals["students"]
        students_per_unit = round(students / units, 2) if units else 0.0
        summary.append(
            {
                "district": district,
                "records": str(totals["records"]),
                "students": str(students),
                "units": str(units),
                "students_per_unit": f"{students_per_unit:.2f}",
            }
        )

    return summary


def write_csv(rows: list[dict[str, str]], output_path: pathlib.Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["district", "records", "students", "units", "students_per_unit"]
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, str]], output_path: pathlib.Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_students = sum(parse_int(r["students"]) for r in rows)
    total_units = sum(parse_int(r["units"]) for r in rows)
    overall_ratio = round(total_students / total_units, 2) if total_units else 0.0

    lines = [
        "# Student Housing Accountability Summary",
        "",
        "## Topline Metrics",
        f"- Districts covered: {len(rows)}",
        f"- Total students (reported): {total_students}",
        f"- Total units (reported): {total_units}",
        f"- Students per unit (overall): {overall_ratio:.2f}",
        "",
        "## District Table",
        "",
        "| District | Records | Students | Units | Students/Unit |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    for row in rows:
        lines.append(
            f"| {row['district']} | {row['records']} | {row['students']} | {row['units']} | {row['students_per_unit']} |"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_path = pathlib.Path(args.student_housing)
    output_csv = pathlib.Path(args.output_csv)
    output_md = pathlib.Path(args.output_md)

    if not input_path.exists():
        print(f"Input not found: {input_path}")
        print("Run src/data/prepare_student_housing.py first.")
        return 1

    with input_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    if not rows:
        print("Input has no data rows.")
        return 1

    summary = compute_district_summary(rows)
    write_csv(summary, output_csv)
    write_markdown(summary, output_md)

    print(f"Wrote district summary CSV: {output_csv}")
    print(f"Wrote markdown report: {output_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
