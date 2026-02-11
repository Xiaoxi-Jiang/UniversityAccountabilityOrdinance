#!/usr/bin/env python3
"""Download core Boston open datasets for the project."""

from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import sys
import urllib.error
import urllib.request

DATASETS = {
    "violations": "https://data.boston.gov/dataset/800a2663-1d6a-46e7-9356-bedb70f5332c/resource/800a2663-1d6a-46e7-9356-bedb70f5332c/download/tmpufnraytw.csv",
    "service_requests_311": "https://data.boston.gov/dataset/6ff6a6fd-3141-4440-a880-6f60a37fe789/resource/e8e2fca7-0f1a-4a44-889f-3b2f58d27f0a/download/tmph4v4ws7m.csv",
    "sam_addresses": "https://data.boston.gov/dataset/c42f2d06-d6eb-4f93-8e25-bd5adffb805f/resource/9fdbdcad-67c8-4b23-b6c2-8618e77f602f/download/tmpm0v4f3h7.csv",
    "property_assessment": "https://data.boston.gov/dataset/062fc6fa-c88d-4241-b0e3-30db35f0fbaf/resource/e5c7f3d8-57ac-4c8f-bf0b-8fd13af69f95/download/tmp3i4a90ih.csv",
}


def fetch_one(name: str, url: str, output_dir: pathlib.Path, stamp: str) -> tuple[bool, str]:
    destination = output_dir / f"{name}_{stamp}.csv"
    try:
        urllib.request.urlretrieve(url, destination)
        return True, str(destination)
    except urllib.error.URLError as exc:
        return False, f"{name}: download failed ({exc})"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Boston open datasets.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        help="Dataset keys to fetch. Use 'all' or a subset of: " + ", ".join(DATASETS.keys()),
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw",
        help="Directory where raw CSV files are stored.",
    )
    return parser.parse_args()


def resolve_targets(dataset_args: list[str]) -> dict[str, str]:
    if "all" in dataset_args:
        return DATASETS

    selected: dict[str, str] = {}
    for key in dataset_args:
        if key not in DATASETS:
            valid = ", ".join(DATASETS.keys())
            raise ValueError(f"Unknown dataset '{key}'. Valid keys: {valid}")
        selected[key] = DATASETS[key]
    return selected


def main() -> int:
    args = parse_args()
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        targets = resolve_targets(args.datasets)
    except ValueError as exc:
        print(exc)
        return 2

    stamp = dt.datetime.now().strftime("%Y%m%d")
    failures: list[str] = []

    for name, url in targets.items():
        ok, info = fetch_one(name, url, output_dir, stamp)
        if ok:
            print(f"[ok] {name} -> {info}")
        else:
            failures.append(info)
            print(f"[error] {info}")

    if failures:
        print("\nSome datasets failed to download. Check network access and dataset URLs.")
        return 1

    print("\nAll requested datasets downloaded successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
