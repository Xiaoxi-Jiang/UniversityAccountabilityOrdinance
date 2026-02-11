#!/usr/bin/env python3
"""Integrate violations + 311 against property registry and compute weighted risk scores."""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.pipeline_utils import (
    DATE_CANDIDATES,
    find_best_address_match,
    make_property_key,
    normalize_text,
    weighted_events_score,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute weighted risk model for properties and landlords.")
    parser.add_argument("--registry", default="data/processed/property_registry.csv")
    parser.add_argument("--violations", default="data/raw/violations.csv")
    parser.add_argument("--service-311", default="data/raw/service_requests_311.csv")
    parser.add_argument("--output-property", default="data/processed/property_risk_model.csv")
    parser.add_argument("--output-landlord", default="data/processed/landlord_risk_model.csv")
    parser.add_argument("--address-match-threshold", type=float, default=0.6)
    parser.add_argument("--bad-landlord-threshold", type=float, default=6.0)
    parser.add_argument("--decay-lambda", type=float, default=0.25)
    return parser.parse_args()


def choose_column(fieldnames: list[str], candidates: list[str]) -> str | None:
    lookup = {f.strip().lower(): f for f in fieldnames}
    for c in candidates:
        if c in lookup:
            return lookup[c]
    return None


def choose_date_column(fieldnames: list[str]) -> str | None:
    return choose_column(fieldnames, DATE_CANDIDATES)


def load_csv(path: pathlib.Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        return [], []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            return [], []
        return list(reader), list(reader.fieldnames)


def build_address_index(registry_rows: list[dict[str, str]]) -> dict[str, str]:
    index: dict[str, str] = {}
    for row in registry_rows:
        address = row.get("address", "")
        district = row.get("district", "")
        norm = normalize_text(address)
        if norm:
            index[norm] = row["property_key"]
        composite = normalize_text(f"{address} {district}")
        if composite:
            index[composite] = row["property_key"]
    return index


def match_property_key(address: str, district: str, index: dict[str, str], threshold: float) -> tuple[str | None, str, float]:
    norm_addr = normalize_text(address)
    exact = index.get(norm_addr)
    if exact:
        return exact, "exact", 1.0

    composite = normalize_text(f"{address} {district}")
    exact = index.get(composite)
    if exact:
        return exact, "exact_composite", 1.0

    best_addr, score = find_best_address_match(norm_addr, index.keys(), threshold=threshold)
    if best_addr:
        return index[best_addr], "fuzzy", score

    key = make_property_key(address, district)
    return key, "generated", 0.0


def main() -> int:
    args = parse_args()
    registry_path = pathlib.Path(args.registry)
    violations_path = pathlib.Path(args.violations)
    service311_path = pathlib.Path(args.service_311)

    if not registry_path.exists():
        print(f"Input not found: {registry_path}")
        print("Run src/data/build_property_registry.py first.")
        return 1

    registry_rows, _ = load_csv(registry_path)
    address_index = build_address_index(registry_rows)
    property_meta = {r["property_key"]: r for r in registry_rows}

    violations_rows, violations_fields = load_csv(violations_path)
    requests_rows, requests_fields = load_csv(service311_path)

    v_addr_col = choose_column(violations_fields, ["address", "location", "street_address", "full_address", "violation_address"])
    v_district_col = choose_column(violations_fields, ["district", "city_council_district", "council_district"])
    v_sev_col = choose_column(violations_fields, ["severity", "code_severity", "violation_level", "description"])
    v_date_col = choose_date_column(violations_fields)

    r_addr_col = choose_column(requests_fields, ["address", "location", "street_address", "full_address"])
    r_district_col = choose_column(requests_fields, ["district", "city_council_district", "council_district"])
    r_sev_col = choose_column(requests_fields, ["case_title", "subject", "reason", "type"])
    r_date_col = choose_date_column(requests_fields)

    violations_by_prop: dict[str, list[dict[str, str]]] = defaultdict(list)
    requests_by_prop: dict[str, list[dict[str, str]]] = defaultdict(list)

    for row in violations_rows:
        address = row.get(v_addr_col, "") if v_addr_col else ""
        district = row.get(v_district_col, "") if v_district_col else ""
        key, _, _ = match_property_key(address, district, address_index, args.address_match_threshold)
        row["_severity"] = row.get(v_sev_col, "") if v_sev_col else ""
        row["_date"] = row.get(v_date_col, "") if v_date_col else ""
        violations_by_prop[key].append(row)

    for row in requests_rows:
        address = row.get(r_addr_col, "") if r_addr_col else ""
        district = row.get(r_district_col, "") if r_district_col else ""
        key, _, _ = match_property_key(address, district, address_index, args.address_match_threshold)
        row["_severity"] = row.get(r_sev_col, "") if r_sev_col else ""
        row["_date"] = row.get(r_date_col, "") if r_date_col else ""
        requests_by_prop[key].append(row)

    property_rows: list[dict[str, str]] = []
    landlord_agg: dict[str, dict[str, float]] = defaultdict(lambda: {"properties": 0.0, "risk": 0.0, "violations": 0.0, "service_311": 0.0})

    for key, meta in property_meta.items():
        events_v = violations_by_prop.get(key, [])
        events_r = requests_by_prop.get(key, [])

        v_score = weighted_events_score(events_v, "_severity", "_date", decay_lambda=args.decay_lambda)
        r_score = weighted_events_score(events_r, "_severity", "_date", decay_lambda=args.decay_lambda)

        total_score = round(v_score + 0.4 * r_score, 4)
        landlord = meta.get("landlord", "").strip() or "UNKNOWN"

        property_rows.append(
            {
                "property_key": key,
                "address": meta.get("address", ""),
                "district": meta.get("district", ""),
                "landlord": landlord,
                "latitude": meta.get("latitude", ""),
                "longitude": meta.get("longitude", ""),
                "violation_events": str(len(events_v)),
                "service_311_events": str(len(events_r)),
                "violation_score": f"{v_score:.4f}",
                "service_311_score": f"{r_score:.4f}",
                "risk_score": f"{total_score:.4f}",
                "bad_landlord": "0",
            }
        )

        landlord_agg[landlord]["properties"] += 1
        landlord_agg[landlord]["risk"] += total_score
        landlord_agg[landlord]["violations"] += len(events_v)
        landlord_agg[landlord]["service_311"] += len(events_r)

    landlord_rows: list[dict[str, str]] = []
    for landlord, values in sorted(landlord_agg.items(), key=lambda x: x[1]["risk"], reverse=True):
        risk = round(values["risk"], 4)
        bad = 1 if risk >= args.bad_landlord_threshold else 0
        landlord_rows.append(
            {
                "landlord": landlord,
                "properties": str(int(values["properties"])),
                "risk_score": f"{risk:.4f}",
                "violation_events": str(int(values["violations"])),
                "service_311_events": str(int(values["service_311"])),
                "bad_landlord": str(bad),
            }
        )

    bad_landlords = {r["landlord"] for r in landlord_rows if r["bad_landlord"] == "1"}
    for row in property_rows:
        row["bad_landlord"] = "1" if row["landlord"] in bad_landlords else "0"

    out_prop = pathlib.Path(args.output_property)
    out_landlord = pathlib.Path(args.output_landlord)
    out_prop.parent.mkdir(parents=True, exist_ok=True)

    prop_fields = [
        "property_key",
        "address",
        "district",
        "landlord",
        "latitude",
        "longitude",
        "violation_events",
        "service_311_events",
        "violation_score",
        "service_311_score",
        "risk_score",
        "bad_landlord",
    ]
    with out_prop.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=prop_fields)
        writer.writeheader()
        for row in sorted(property_rows, key=lambda r: float(r["risk_score"]), reverse=True):
            writer.writerow(row)

    landlord_fields = ["landlord", "properties", "risk_score", "violation_events", "service_311_events", "bad_landlord"]
    with out_landlord.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=landlord_fields)
        writer.writeheader()
        writer.writerows(landlord_rows)

    print(f"Wrote property risk model: {out_prop}")
    print(f"Wrote landlord risk model: {out_landlord}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
