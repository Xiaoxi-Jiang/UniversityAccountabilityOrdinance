"""Building permit loading, cleaning, and aggregation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd

from src.data.features import normalize_address, normalize_string, normalize_zip
from src.data.violations import _standardize_columns

from .common import (
    build_address_zip_key,
    download_arcgis_layer,
    find_local_file,
    load_existing_clean_output,
    load_local_tabular,
    save_clean_output,
)


PERMITS_URL = (
    "https://gisportal.boston.gov/arcgis/rest/services/ISD/"
    "approved_building_permits/MapServer/0/query"
)
MAJOR_PERMIT_TOKENS = {"addition", "alteration", "renovation", "foundation", "structural", "demolition"}
OCCUPANCY_PERMIT_TOKENS = {"occupancy", "certificate", "dwelling", "residential", "use change"}


@dataclass(frozen=True)
class PermitContextConfig:
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    candidates: tuple[str, ...] = (
        "building_permits.csv",
        "building_permits.xlsx",
        "building_permits.geojson",
    )
    permits_url: str | None = PERMITS_URL
    clean_output_path: Path = Path("data/processed/building_permits_clean.csv")
    download_timeout: int = 60
    download_chunk_size: int = 1000
    download_workers: int = 4
    download_max_attempts: int = 3


def _normalize_identifier(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    if re.fullmatch(r"\d+\.0+", text):
        return text.split(".", 1)[0]
    return normalize_string(text).replace(" ", "")


def _coerce_permit_datetime(series: pd.Series) -> pd.Series:
    """Parse permit timestamps from either ISO strings or epoch milliseconds."""
    numeric = pd.to_numeric(series, errors="coerce")
    parsed_numeric = pd.to_datetime(numeric, unit="ms", errors="coerce")
    parsed_text = pd.to_datetime(series, errors="coerce")
    return parsed_numeric.where(parsed_numeric.notna(), parsed_text)


def clean_permits(df: pd.DataFrame) -> pd.DataFrame:
    """Clean permit exports for property-level enrichment."""
    cleaned = df.copy()
    cleaned.columns = _standardize_columns(cleaned.columns.tolist())
    if "parcel_id" in cleaned.columns and "map_par_id" not in cleaned.columns:
        cleaned["map_par_id"] = cleaned["parcel_id"]
    if "property_id" in cleaned.columns and "pid" not in cleaned.columns:
        cleaned["pid"] = cleaned["property_id"]

    address_col = next(
        (
            column
            for column in [
                "full_address",
                "address",
                "street_address",
                "work_address",
                "site_address",
            ]
            if column in cleaned.columns
        ),
        None,
    )
    zip_col = next((column for column in ["zip_code", "zip", "postal_code"] if column in cleaned.columns), None)
    issued_col = next(
        (column for column in ["issued_date", "issue_date", "permit_issued_date", "approval_date"] if column in cleaned.columns),
        None,
    )
    permit_text_cols = [
        column
        for column in [
            "permit_type",
            "permittypedescr",
            "description",
            "worktype",
            "declared_valuation_description",
        ]
        if column in cleaned.columns
    ]

    if address_col and zip_col:
        cleaned["address_zip_key"] = build_address_zip_key(cleaned, address_col, zip_col)
    elif address_col:
        cleaned["address_zip_key"] = cleaned[address_col].map(normalize_address).astype("string")

    if address_col:
        cleaned["permit_address"] = cleaned[address_col].map(normalize_address).astype("string")
    if zip_col:
        cleaned["permit_zip"] = cleaned[zip_col].map(normalize_zip).astype("string")
    for column in ["map_par_id", "loc_id", "gis_id", "pid"]:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].map(_normalize_identifier).astype("string")
    if issued_col:
        cleaned["permit_issue_date"] = _coerce_permit_datetime(cleaned[issued_col])

    permit_text = (
        cleaned[permit_text_cols].fillna("").astype("string").agg(" ".join, axis=1).map(normalize_string)
        if permit_text_cols
        else pd.Series("", index=cleaned.index, dtype="string")
    )
    cleaned["major_permit_flag"] = permit_text.apply(
        lambda text: int(any(token in text for token in MAJOR_PERMIT_TOKENS))
    )
    cleaned["occupancy_related_permit_flag"] = permit_text.apply(
        lambda text: int(any(token in text for token in OCCUPANCY_PERMIT_TOKENS))
    )
    cleaned["permit_record_count"] = 1
    return cleaned


def aggregate_permits(
    permits_df: pd.DataFrame,
    *,
    join_key: str,
    reference_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Aggregate permit records into property-level context features."""
    if join_key not in permits_df.columns:
        return pd.DataFrame(columns=[join_key])

    working = permits_df.dropna(subset=[join_key]).copy()
    if working.empty:
        return pd.DataFrame(columns=[join_key])

    if "permit_issue_date" in working.columns:
        working["permit_issue_date"] = pd.to_datetime(working["permit_issue_date"], errors="coerce")
    if reference_date is not None:
        reference_date = pd.to_datetime(reference_date, errors="coerce")

    if reference_date is None and "permit_issue_date" in working.columns:
        reference_date = working["permit_issue_date"].max()

    if pd.notna(reference_date) and "permit_issue_date" in working.columns:
        working["permits_730d"] = (
            working["permit_issue_date"]
            .ge(reference_date - pd.Timedelta(days=730))
            .fillna(False)
            .astype(int)
        )
        recency = (reference_date - working["permit_issue_date"]).dt.days
        working["days_since_permit"] = recency.where(recency.ge(0))
    else:
        working["permits_730d"] = 0
        working["days_since_permit"] = pd.NA

    aggregated = (
        working.groupby(join_key)
        .agg(
            permit_count=("permit_record_count", "sum"),
            major_permit_count=("major_permit_flag", "sum"),
            occupancy_related_permit_count=("occupancy_related_permit_flag", "sum"),
            permits_730d=("permits_730d", "sum"),
            last_permit_date=("permit_issue_date", "max"),
            min_days_since_permit=("days_since_permit", "min"),
        )
        .reset_index()
    )
    aggregated["major_permit_flag"] = aggregated["major_permit_count"].fillna(0).gt(0).astype(int)
    aggregated["occupancy_related_permit_flag"] = (
        aggregated["occupancy_related_permit_count"].fillna(0).gt(0).astype(int)
    )
    return aggregated


def load_permits(config: PermitContextConfig) -> pd.DataFrame | None:
    """Load a cleaned permit table from cache, local files, or the public endpoint."""
    raw_path = find_local_file(config.raw_dir, config.candidates)
    if raw_path is None:
        existing = load_existing_clean_output(config.clean_output_path)
        if existing is not None:
            cleaned_existing = clean_permits(existing)
            save_clean_output(cleaned_existing, config.clean_output_path)
            return cleaned_existing
        fallback_path = config.raw_dir / config.candidates[0]
        downloaded = download_arcgis_layer(
            config.permits_url,
            fallback_path,
            timeout=config.download_timeout,
            use_object_id_chunks=True,
            chunk_size=config.download_chunk_size,
            max_workers=config.download_workers,
            max_attempts=config.download_max_attempts,
        )
        raw_path = downloaded or (fallback_path if fallback_path.exists() else None)
    if raw_path is None:
        print("Building permit data unavailable: no local extract or public endpoint response was found.")
        return None

    cleaned = clean_permits(load_local_tabular(raw_path))
    save_clean_output(cleaned, config.clean_output_path)
    return cleaned
