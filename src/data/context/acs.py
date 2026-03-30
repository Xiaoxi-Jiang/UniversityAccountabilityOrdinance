"""ACS ZIP-level contextual features for interpretive neighborhood context."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

from src.data.features import normalize_zip
from src.data.violations import _standardize_columns

from .common import find_local_file, load_existing_clean_output, load_local_tabular, save_clean_output


ACS_API_URL = "https://api.census.gov/data/{year}/acs/acs5"
ACS_VARIABLES = [
    "B19013_001E",
    "B25003_001E",
    "B25003_003E",
    "B25002_001E",
    "B25002_003E",
    "B25064_001E",
    "B01003_001E",
    "B01001_007E",
    "B01001_008E",
    "B01001_009E",
    "B01001_010E",
    "B01001_031E",
    "B01001_032E",
    "B01001_033E",
    "B01001_034E",
]


@dataclass(frozen=True)
class ACSContextConfig:
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    candidates: tuple[str, ...] = ("acs_context.csv", "acs_context.xlsx")
    year: int = 2022
    clean_output_path: Path = Path("data/processed/acs_context_clean.csv")


def _coerce_numeric(df: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")


def clean_acs_context(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize ACS ZIP-level context and derive interpretable features."""
    cleaned = df.copy()
    cleaned.columns = _standardize_columns(cleaned.columns.tolist())

    rename_map = {
        "zip": "acs_zip",
        "zcta5": "acs_zip",
        "zcta5ce10": "acs_zip",
        "zip_code": "acs_zip",
        "zip_code_tabulation_area": "acs_zip",
        "median_household_income": "b19013_001e",
        "total_tenure": "b25003_001e",
        "renter_occupied": "b25003_003e",
        "total_housing_units": "b25002_001e",
        "vacant_housing_units": "b25002_003e",
        "median_gross_rent": "b25064_001e",
        "total_population": "b01003_001e",
    }
    cleaned = cleaned.rename(columns={src: dst for src, dst in rename_map.items() if src in cleaned.columns})

    if "name" in cleaned.columns and "acs_zip" not in cleaned.columns:
        extracted = cleaned["name"].astype("string").str.extract(r"(\d{5})")
        cleaned["acs_zip"] = extracted.iloc[:, 0]

    numeric_cols = [column.lower() for column in ACS_VARIABLES]
    _coerce_numeric(cleaned, numeric_cols)
    if "acs_zip" in cleaned.columns:
        cleaned["acs_zip"] = cleaned["acs_zip"].map(normalize_zip).astype("string")

    young_adult_cols = [
        "b01001_007e",
        "b01001_008e",
        "b01001_009e",
        "b01001_010e",
        "b01001_031e",
        "b01001_032e",
        "b01001_033e",
        "b01001_034e",
    ]
    cleaned["acs_median_household_income"] = cleaned.get("b19013_001e")
    cleaned["acs_median_gross_rent"] = cleaned.get("b25064_001e")
    cleaned["acs_renter_occupied_share"] = (
        cleaned.get("b25003_003e", 0) / cleaned.get("b25003_001e", 1)
    )
    cleaned["acs_vacancy_rate"] = cleaned.get("b25002_003e", 0) / cleaned.get("b25002_001e", 1)
    cleaned["acs_young_adult_share"] = (
        cleaned.reindex(columns=young_adult_cols).fillna(0).sum(axis=1) / cleaned.get("b01003_001e", 1)
    )

    keep_cols = [
        "acs_zip",
        "acs_median_household_income",
        "acs_median_gross_rent",
        "acs_renter_occupied_share",
        "acs_vacancy_rate",
        "acs_young_adult_share",
    ]
    available = [column for column in keep_cols if column in cleaned.columns]
    return cleaned.loc[:, available].dropna(subset=["acs_zip"], how="all").reset_index(drop=True)


def _download_acs_context(config: ACSContextConfig, timeout: int = 60) -> pd.DataFrame | None:
    """Query the ACS API for ZIP-level context features."""
    try:
        response = requests.get(
            ACS_API_URL.format(year=config.year),
            params={
                "get": ",".join(ACS_VARIABLES),
                "for": "zip code tabulation area:*",
            },
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        print(f"Skipping ACS download for {config.year}: {exc}")
        return None

    if not payload or len(payload) < 2:
        print(f"Skipping ACS download for {config.year}: no records returned.")
        return None

    header, rows = payload[0], payload[1:]
    return pd.DataFrame(rows, columns=header)


def load_acs_context(config: ACSContextConfig) -> pd.DataFrame | None:
    """Load ACS ZIP-level context from cache, local files, or the Census API."""
    existing = load_existing_clean_output(config.clean_output_path)
    if existing is not None:
        return existing

    raw_path = find_local_file(config.raw_dir, config.candidates)
    if raw_path is not None:
        cleaned = clean_acs_context(load_local_tabular(raw_path))
        save_clean_output(cleaned, config.clean_output_path)
        return cleaned

    downloaded = _download_acs_context(config)
    if downloaded is None:
        print("ACS context unavailable: no local extract or API response was found.")
        return None

    cleaned = clean_acs_context(downloaded)
    save_clean_output(cleaned, config.clean_output_path)
    return cleaned
