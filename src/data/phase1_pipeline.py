"""Phase 1 pipeline: collect and clean Boston building/property violations data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd


VIOLATIONS_CSV_URL = (
    "https://data.boston.gov/datastore/dump/"
    "800a2663-1d6a-46e7-9356-bedb70f5332c?format=csv"
)


@dataclass(frozen=True)
class Phase1Config:
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    raw_filename: str = "violations.csv"
    cleaned_filename: str = "violations_clean.csv"


def _standardize_columns(columns: list[str]) -> list[str]:
    clean_cols = []
    for col in columns:
        col = col.strip().lower()
        col = re.sub(r"[^a-z0-9]+", "_", col)
        col = re.sub(r"_+", "_", col).strip("_")
        clean_cols.append(col)
    return clean_cols


def download_violations_csv(url: str, output_path: Path, timeout: int = 30) -> None:
    import requests

    output_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    output_path.write_bytes(response.content)


def clean_violations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = _standardize_columns(df.columns.tolist())

    # Keep a stable subset when available.
    expected_cols = [
        "case_no",
        "status",
        "description",
        "violationtype",
        "violationtype_descr",
        "violation_st",
        "violation_zip",
        "violator_name",
        "violdttm",
    ]
    cols = [c for c in expected_cols if c in df.columns]
    if cols:
        df = df.loc[:, cols].copy()

    if "status" in df.columns:
        status = df["status"].astype("string").str.strip().str.lower()
        status = status.fillna("unknown")
        df.loc[:, "status"] = status
    if "violdttm" in df.columns:
        df.loc[:, "violdttm"] = pd.to_datetime(df["violdttm"], errors="coerce")
    if "case_no" in df.columns:
        df = df.drop_duplicates(subset=["case_no"], keep="first")

    if "status" in df.columns:
        df.loc[:, "is_open_violation"] = df["status"].isin(
            ["open", "pending", "active"]
        ).astype(int)

    return df


def run_phase1(config: Phase1Config) -> tuple[Path, Path]:
    raw_path = config.raw_dir / config.raw_filename
    cleaned_path = config.processed_dir / config.cleaned_filename

    download_violations_csv(VIOLATIONS_CSV_URL, raw_path)

    df = pd.read_csv(raw_path)
    cleaned = clean_violations(df)

    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(cleaned_path, index=False)

    return raw_path, cleaned_path


def main() -> None:
    raw_path, cleaned_path = run_phase1(Phase1Config())
    print(f"Raw data saved to: {raw_path}")
    print(f"Cleaned data saved to: {cleaned_path}")


if __name__ == "__main__":
    main()
