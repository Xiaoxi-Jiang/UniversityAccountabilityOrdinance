#!/usr/bin/env python3
"""Compute district-level spatial risk metrics using GeoJSON boundaries and property coordinates."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from collections import defaultdict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spatial aggregation for property risk by district polygon.")
    parser.add_argument("--property-risk", default="data/processed/property_risk_model.csv")
    parser.add_argument("--district-geojson", default="data/raw/city_council_districts.geojson")
    parser.add_argument("--output-csv", default="data/processed/spatial_district_risk.csv")
    parser.add_argument("--output-md", default="reports/spatial_district_summary.md")
    parser.add_argument("--district-property", default="district")
    return parser.parse_args()


def point_in_ring(x: float, y: float, ring: list[list[float]]) -> bool:
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        intersect = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi)
        if intersect:
            inside = not inside
        j = i
    return inside


def point_in_polygon(x: float, y: float, coords: list) -> bool:
    if not coords:
        return False
    if point_in_ring(x, y, coords[0]) is False:
        return False
    for hole in coords[1:]:
        if point_in_ring(x, y, hole):
            return False
    return True


def find_district(lon: float, lat: float, features: list[dict]) -> str | None:
    for feat in features:
        geom = feat.get("geometry", {})
        props = feat.get("properties", {})
        district = props.get("district") or props.get("name") or props.get("DISTRICT") or "UNKNOWN"
        gtype = geom.get("type")
        coords = geom.get("coordinates", [])

        if gtype == "Polygon":
            if point_in_polygon(lon, lat, coords):
                return str(district)
        elif gtype == "MultiPolygon":
            for poly in coords:
                if point_in_polygon(lon, lat, poly):
                    return str(district)
    return None


def main() -> int:
    args = parse_args()
    prop_path = pathlib.Path(args.property_risk)
    geo_path = pathlib.Path(args.district_geojson)
    out_csv = pathlib.Path(args.output_csv)
    out_md = pathlib.Path(args.output_md)

    if not prop_path.exists():
        print(f"Input not found: {prop_path}")
        return 1

    features: list[dict] = []
    if geo_path.exists():
        data = json.loads(geo_path.read_text(encoding="utf-8"))
        features = data.get("features", [])

    with prop_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    agg: dict[str, dict[str, float]] = defaultdict(lambda: {"properties": 0.0, "risk": 0.0, "bad": 0.0})
    spatial_used = 0

    for row in rows:
        district = (row.get(args.district_property, "") or "").strip()
        lat_raw = (row.get("latitude", "") or "").strip()
        lon_raw = (row.get("longitude", "") or "").strip()

        if features and lat_raw and lon_raw:
            try:
                lat = float(lat_raw)
                lon = float(lon_raw)
                resolved = find_district(lon, lat, features)
                if resolved:
                    district = resolved
                    spatial_used += 1
            except ValueError:
                pass

        district = district or "UNKNOWN"
        agg[district]["properties"] += 1
        agg[district]["risk"] += float(row.get("risk_score", "0") or 0)
        agg[district]["bad"] += 1 if row.get("bad_landlord", "0") == "1" else 0

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        fields = ["district", "properties", "total_risk", "avg_risk", "bad_landlord_properties"]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for district, vals in sorted(agg.items(), key=lambda x: x[1]["risk"], reverse=True):
            props = int(vals["properties"])
            total_risk = round(vals["risk"], 4)
            avg_risk = round(total_risk / props, 4) if props else 0.0
            writer.writerow(
                {
                    "district": district,
                    "properties": str(props),
                    "total_risk": f"{total_risk:.4f}",
                    "avg_risk": f"{avg_risk:.4f}",
                    "bad_landlord_properties": str(int(vals["bad"])),
                }
            )

    lines = [
        "# Spatial District Risk Summary",
        "",
        f"- Districts covered: {len(agg)}",
        f"- Spatial polygon assignment used: {spatial_used} properties",
        "",
        "| District | Properties | Total Risk | Avg Risk | Bad Landlord Properties |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    for district, vals in sorted(agg.items(), key=lambda x: x[1]["risk"], reverse=True):
        props = int(vals["properties"])
        total_risk = round(vals["risk"], 4)
        avg_risk = round(total_risk / props, 4) if props else 0.0
        lines.append(f"| {district} | {props} | {total_risk:.4f} | {avg_risk:.4f} | {int(vals['bad'])} |")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote spatial district CSV: {out_csv}")
    print(f"Wrote spatial district summary: {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
