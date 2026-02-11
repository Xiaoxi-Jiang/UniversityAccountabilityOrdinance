#!/usr/bin/env python3
"""Normalize student housing data into a clean processed CSV."""

from __future__ import annotations

import argparse
import csv
import pathlib
import re
import sys

COLUMN_ALIASES = {
    "address": {"address", "street_address", "property_address"},
    "district": {"district", "city_council_district", "council_district"},
    "year": {"year", "report_year", "academic_year"},
    "student_count": {"student_count", "students", "num_students", "students_total"},
    "units": {"units", "unit_count", "num_units", "housing_units"},
    "landlord": {"landlord", "owner", "owner_name", "property_owner"},
    "latitude": {"latitude", "lat", "y", "y_coord"},
    "longitude": {"longitude", "lon", "lng", "long", "x", "x_coord"},
}

OUTPUT_FIELDS = ["address", "district", "year", "student_count", "units", "landlord", "latitude", "longitude"]


def normalize_header(header: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", header.strip().lower()).strip("_")


def build_column_map(headers: list[str]) -> dict[str, str]:
    normalized = {normalize_header(h): h for h in headers}
    result: dict[str, str] = {}

    for canonical, candidates in COLUMN_ALIASES.items():
        for alias in candidates:
            if alias in normalized:
                result[canonical] = normalized[alias]
                break
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize student housing CSV into a standard schema.")
    parser.add_argument(
        "--input",
        default="data/raw/student_housing.csv",
        help="Raw student housing CSV path.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/student_housing_clean.csv",
        help="Output processed CSV path.",
    )
    return parser.parse_args()


def normalize_value(field: str, value: str) -> str:
    if value is None:
        return ""
    text = value.strip()

    if field in {"year", "student_count", "units"}:
        digits = re.sub(r"[^0-9]", "", text)
        return digits
    if field in {"latitude", "longitude"}:
        return re.sub(r"[^0-9.\\-]", "", text)

    return text


def main() -> int:
    args = parse_args()
    input_path = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output)

    if not input_path.exists():
        print(f"Input not found: {input_path}")
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8-sig", newline="") as infile:
        reader = csv.DictReader(infile)
        if reader.fieldnames is None:
            print("Input CSV has no header row.")
            return 1

        column_map = build_column_map(reader.fieldnames)
        missing_required = [field for field in ["district", "year", "student_count"] if field not in column_map]
        if missing_required:
            print("Missing required fields in input: " + ", ".join(missing_required))
            print("Detected headers: " + ", ".join(reader.fieldnames))
            return 1

        rows_written = 0
        with output_path.open("w", encoding="utf-8", newline="") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()

            for row in reader:
                out_row: dict[str, str] = {}
                for field in OUTPUT_FIELDS:
                    source_col = column_map.get(field)
                    raw_value = row.get(source_col, "") if source_col else ""
                    out_row[field] = normalize_value(field, raw_value)
                writer.writerow(out_row)
                rows_written += 1

    print(f"Wrote {rows_written} rows to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
