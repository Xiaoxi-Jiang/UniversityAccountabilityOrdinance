#!/usr/bin/env python3
"""Link student housing records to violations by normalized address and derive landlord risk metrics."""

from __future__ import annotations

import argparse
import csv
import pathlib
import re
import sys
from collections import defaultdict


VIOLATION_ADDRESS_CANDIDATES = [
    "address",
    "location",
    "full_address",
    "street_address",
    "violation_address",
]

VIOLATION_OWNER_CANDIDATES = [
    "owner",
    "owner_name",
    "property_owner",
    "name",
    "landlord",
]

SEVERITY_CANDIDATES = [
    "severity",
    "code_severity",
    "violation_level",
    "violation_type",
    "description",
]

HIGH_SEVERITY_KEYWORDS = {"unsafe", "fire", "hazard", "emergency", "critical", "severe"}


def normalize_text(value: str) -> str:
    value = (value or "").strip().lower()
    value = value.replace("#", " ")
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\b(street|st)\b", " st ", value)
    value = re.sub(r"\b(avenue|ave)\b", " ave ", value)
    value = re.sub(r"\b(road|rd)\b", " rd ", value)
    value = re.sub(r"\b(boulevard|blvd)\b", " blvd ", value)
    value = re.sub(r"\b(apartment|apt|unit)\b", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def choose_column(fieldnames: list[str], candidates: list[str]) -> str | None:
    lookup = {f.strip().lower(): f for f in fieldnames}
    for key in candidates:
        if key in lookup:
            return lookup[key]
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Link student housing rows to violation records.")
    parser.add_argument(
        "--student-housing",
        default="data/processed/student_housing_clean.csv",
        help="Processed student housing CSV.",
    )
    parser.add_argument(
        "--violations",
        default="data/raw/violations_sample.csv",
        help="Violations CSV (raw source or prepared sample).",
    )
    parser.add_argument(
        "--output-linked",
        default="data/processed/student_housing_with_violations.csv",
        help="Output merged CSV.",
    )
    parser.add_argument(
        "--output-landlord",
        default="data/processed/landlord_risk_summary.csv",
        help="Output landlord risk summary CSV.",
    )
    parser.add_argument(
        "--min-violations-bad-landlord",
        type=int,
        default=3,
        help="Threshold for labeling landlord as bad_landlord.",
    )
    return parser.parse_args()


def load_violations(path: pathlib.Path) -> tuple[dict[str, list[dict[str, str]]], dict[str, list[dict[str, str]]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError("Violations CSV has no header row.")

        addr_col = choose_column(reader.fieldnames, VIOLATION_ADDRESS_CANDIDATES)
        owner_col = choose_column(reader.fieldnames, VIOLATION_OWNER_CANDIDATES)
        sev_col = choose_column(reader.fieldnames, SEVERITY_CANDIDATES)

        if not addr_col:
            raise ValueError("Could not find address column in violations CSV.")

        by_address: dict[str, list[dict[str, str]]] = defaultdict(list)
        by_owner: dict[str, list[dict[str, str]]] = defaultdict(list)

        for row in reader:
            norm_addr = normalize_text(row.get(addr_col, ""))
            if norm_addr:
                by_address[norm_addr].append(row)

            if owner_col:
                norm_owner = normalize_text(row.get(owner_col, ""))
                if norm_owner:
                    by_owner[norm_owner].append(row)

            if sev_col and "_severity_text" not in row:
                row["_severity_text"] = (row.get(sev_col, "") or "").lower()

        return by_address, by_owner


def severity_count(rows: list[dict[str, str]]) -> int:
    severe = 0
    for row in rows:
        text = (row.get("_severity_text", "") or "").lower()
        if any(k in text for k in HIGH_SEVERITY_KEYWORDS):
            severe += 1
    return severe


def main() -> int:
    args = parse_args()
    student_path = pathlib.Path(args.student_housing)
    violations_path = pathlib.Path(args.violations)
    linked_path = pathlib.Path(args.output_linked)
    landlord_path = pathlib.Path(args.output_landlord)

    if not student_path.exists():
        print(f"Input not found: {student_path}")
        return 1
    if not violations_path.exists():
        print(f"Input not found: {violations_path}")
        return 1

    by_address, by_owner = load_violations(violations_path)

    with student_path.open("r", encoding="utf-8", newline="") as fh:
        student_reader = csv.DictReader(fh)
        if student_reader.fieldnames is None:
            print("Student housing CSV has no header row.")
            return 1
        student_rows = list(student_reader)

    linked_fields = student_reader.fieldnames + [
        "matched_by",
        "matched_violation_count",
        "matched_high_severity_count",
        "bad_landlord",
    ]

    landlord_agg: dict[str, dict[str, int]] = defaultdict(lambda: {"properties": 0, "violations": 0, "high_severity": 0})

    linked_path.parent.mkdir(parents=True, exist_ok=True)
    with linked_path.open("w", encoding="utf-8", newline="") as out_fh:
        writer = csv.DictWriter(out_fh, fieldnames=linked_fields)
        writer.writeheader()

        for row in student_rows:
            norm_addr = normalize_text(row.get("address", ""))
            norm_owner = normalize_text(row.get("landlord", ""))

            matched = []
            matched_by = "none"

            if norm_addr and norm_addr in by_address:
                matched = by_address[norm_addr]
                matched_by = "address"
            elif norm_owner and norm_owner in by_owner:
                matched = by_owner[norm_owner]
                matched_by = "landlord"

            violation_n = len(matched)
            severe_n = severity_count(matched)
            is_bad = violation_n >= args.min_violations_bad_landlord

            if norm_owner:
                landlord_agg[norm_owner]["properties"] += 1
                landlord_agg[norm_owner]["violations"] += violation_n
                landlord_agg[norm_owner]["high_severity"] += severe_n

            out_row = dict(row)
            out_row["matched_by"] = matched_by
            out_row["matched_violation_count"] = str(violation_n)
            out_row["matched_high_severity_count"] = str(severe_n)
            out_row["bad_landlord"] = "1" if is_bad else "0"
            writer.writerow(out_row)

    landlord_path.parent.mkdir(parents=True, exist_ok=True)
    with landlord_path.open("w", encoding="utf-8", newline="") as out_fh:
        fields = ["landlord", "properties", "violations", "high_severity", "bad_landlord"]
        writer = csv.DictWriter(out_fh, fieldnames=fields)
        writer.writeheader()

        for landlord, stats in sorted(landlord_agg.items(), key=lambda item: item[1]["violations"], reverse=True):
            writer.writerow(
                {
                    "landlord": landlord,
                    "properties": str(stats["properties"]),
                    "violations": str(stats["violations"]),
                    "high_severity": str(stats["high_severity"]),
                    "bad_landlord": "1" if stats["violations"] >= args.min_violations_bad_landlord else "0",
                }
            )

    print(f"Wrote linked output: {linked_path}")
    print(f"Wrote landlord summary: {landlord_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
