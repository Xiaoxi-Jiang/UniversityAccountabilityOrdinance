#!/usr/bin/env python3
"""Render district yearly trend chart as SVG without external plotting dependencies."""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys
from collections import defaultdict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot district yearly students_per_unit trend to SVG.")
    parser.add_argument("--input", default="data/processed/district_yearly_trend.csv")
    parser.add_argument("--output", default="reports/figures/district_yearly_trend.svg")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output)

    if not input_path.exists():
        print(f"Input not found: {input_path}")
        return 1

    with input_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    if not rows:
        print("No rows in trend input.")
        return 1

    data: dict[str, list[tuple[int, float]]] = defaultdict(list)
    years = set()
    values = []

    for row in rows:
        try:
            year = int(row.get("year", "0"))
            val = float(row.get("students_per_unit", "0"))
        except ValueError:
            continue
        district = row.get("district", "UNKNOWN")
        data[district].append((year, val))
        years.add(year)
        values.append(val)

    if not years or not values:
        print("No valid numeric rows in trend input.")
        return 1

    min_year, max_year = min(years), max(years)
    min_val, max_val = min(values), max(values)
    if min_val == max_val:
        max_val = min_val + 1.0

    width, height = 960, 540
    left, right, top, bottom = 80, 40, 50, 70
    plot_w = width - left - right
    plot_h = height - top - bottom

    palette = ["#0b4f6c", "#7a1f5c", "#1b6b3a", "#b3541e", "#3f3d9e", "#8e2a2a"]

    def x_scale(year: int) -> float:
        span = max(max_year - min_year, 1)
        return left + (year - min_year) / span * plot_w

    def y_scale(v: float) -> float:
        return top + (max_val - v) / (max_val - min_val) * plot_h

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f6f4ef"/>',
        f'<text x="{left}" y="30" font-size="22" font-family="Georgia, serif" fill="#222">District Yearly Students/Unit Trend</text>',
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#444"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#444"/>',
    ]

    for year in sorted(years):
        x = x_scale(year)
        svg.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+plot_h}" stroke="#ddd"/>')
        svg.append(f'<text x="{x:.1f}" y="{height-35}" text-anchor="middle" font-size="12" fill="#333">{year}</text>')

    y_ticks = 5
    for i in range(y_ticks + 1):
        v = min_val + i * (max_val - min_val) / y_ticks
        y = y_scale(v)
        svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#e5e5e5"/>')
        svg.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-size="12" fill="#333">{v:.2f}</text>')

    legend_x = left + plot_w - 180
    legend_y = top + 20

    for idx, (district, points) in enumerate(sorted(data.items())):
        color = palette[idx % len(palette)]
        points = sorted(points)
        poly = " ".join(f"{x_scale(y):.1f},{y_scale(v):.1f}" for y, v in points)
        svg.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{poly}"/>')
        for year, val in points:
            svg.append(f'<circle cx="{x_scale(year):.1f}" cy="{y_scale(val):.1f}" r="3.5" fill="{color}"/>')
        ly = legend_y + idx * 22
        svg.append(f'<line x1="{legend_x}" y1="{ly}" x2="{legend_x+20}" y2="{ly}" stroke="{color}" stroke-width="3"/>')
        svg.append(f'<text x="{legend_x+28}" y="{ly+4}" font-size="12" fill="#222">{district}</text>')

    svg.append("</svg>")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(svg) + "\n", encoding="utf-8")

    print(f"Wrote trend chart SVG: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
