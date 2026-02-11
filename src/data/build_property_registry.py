#!/usr/bin/env python3
"""Build unified property registry (master keys) from student housing, SAM, and assessment data."""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.pipeline_utils import make_property_key, normalize_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create property master key registry.")
    parser.add_argument("--student-housing", default="data/processed/student_housing_clean.csv")
    parser.add_argument("--sam", default="data/raw/sam_addresses.csv")
    parser.add_argument("--assessment", default="data/raw/property_assessment.csv")
    parser.add_argument("--output", default="data/processed/property_registry.csv")
    return parser.parse_args()


def choose_column(fieldnames: list[str], candidates: list[str]) -> str | None:
    lookup = {f.strip().lower(): f for f in fieldnames}
    for c in candidates:
        if c in lookup:
            return lookup[c]
    return None


def load_optional_csv(path: pathlib.Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        return [], []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            return [], []
        return list(reader), list(reader.fieldnames)


def main() -> int:
    args = parse_args()
    student_path = pathlib.Path(args.student_housing)
    sam_path = pathlib.Path(args.sam)
    assess_path = pathlib.Path(args.assessment)
    output_path = pathlib.Path(args.output)

    if not student_path.exists():
        print(f"Input not found: {student_path}")
        return 1

    with student_path.open("r", encoding="utf-8", newline="") as fh:
        student_rows = list(csv.DictReader(fh))

    sam_rows, sam_fields = load_optional_csv(sam_path)
    assess_rows, assess_fields = load_optional_csv(assess_path)

    registry: dict[str, dict[str, str]] = {}
    sources: dict[str, set[str]] = defaultdict(set)

    for row in student_rows:
        address = row.get("address", "")
        district = row.get("district", "")
        key = make_property_key(address, district)
        rec = registry.setdefault(
            key,
            {
                "property_key": key,
                "normalized_address": normalize_text(address),
                "address": address,
                "district": district,
                "latitude": row.get("latitude", ""),
                "longitude": row.get("longitude", ""),
                "landlord": row.get("landlord", ""),
            },
        )
        if not rec.get("latitude"):
            rec["latitude"] = row.get("latitude", "")
        if not rec.get("longitude"):
            rec["longitude"] = row.get("longitude", "")
        if not rec.get("landlord"):
            rec["landlord"] = row.get("landlord", "")
        sources[key].add("student")

    sam_addr_col = choose_column(sam_fields, ["address", "full_address", "street_address"])
    sam_lat_col = choose_column(sam_fields, ["latitude", "lat", "y"])
    sam_lon_col = choose_column(sam_fields, ["longitude", "lon", "lng", "x"])
    sam_district_col = choose_column(sam_fields, ["district", "city_council_district", "council_district"])

    for row in sam_rows:
        address = row.get(sam_addr_col, "") if sam_addr_col else ""
        district = row.get(sam_district_col, "") if sam_district_col else ""
        if not address:
            continue
        key = make_property_key(address, district)
        rec = registry.setdefault(
            key,
            {
                "property_key": key,
                "normalized_address": normalize_text(address),
                "address": address,
                "district": district,
                "latitude": row.get(sam_lat_col, "") if sam_lat_col else "",
                "longitude": row.get(sam_lon_col, "") if sam_lon_col else "",
                "landlord": "",
            },
        )
        if not rec.get("latitude") and sam_lat_col:
            rec["latitude"] = row.get(sam_lat_col, "")
        if not rec.get("longitude") and sam_lon_col:
            rec["longitude"] = row.get(sam_lon_col, "")
        if not rec.get("district") and sam_district_col:
            rec["district"] = row.get(sam_district_col, "")
        sources[key].add("sam")

    assess_addr_col = choose_column(assess_fields, ["address", "property_address", "street_address"])
    assess_owner_col = choose_column(assess_fields, ["owner_name", "owner", "landlord", "property_owner"])
    assess_district_col = choose_column(assess_fields, ["district", "city_council_district", "council_district"])

    for row in assess_rows:
        address = row.get(assess_addr_col, "") if assess_addr_col else ""
        district = row.get(assess_district_col, "") if assess_district_col else ""
        if not address:
            continue
        key = make_property_key(address, district)
        rec = registry.setdefault(
            key,
            {
                "property_key": key,
                "normalized_address": normalize_text(address),
                "address": address,
                "district": district,
                "latitude": "",
                "longitude": "",
                "landlord": row.get(assess_owner_col, "") if assess_owner_col else "",
            },
        )
        if not rec.get("district") and assess_district_col:
            rec["district"] = row.get(assess_district_col, "")
        if not rec.get("landlord") and assess_owner_col:
            rec["landlord"] = row.get(assess_owner_col, "")
        sources[key].add("assessment")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "property_key",
        "normalized_address",
        "address",
        "district",
        "latitude",
        "longitude",
        "landlord",
        "sources",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for key, rec in sorted(registry.items()):
            row = dict(rec)
            row["sources"] = "|".join(sorted(sources.get(key, set())))
            writer.writerow(row)

    print(f"Wrote property registry: {output_path}")
    print(f"Properties: {len(registry)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
