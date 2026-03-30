"""SAM/geocoder address context used to stabilize cross-dataset joins."""

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


SAM_ADDRESSES_URL = (
    "https://gis.boston.gov/arcgis/rest/services/SAM/"
    "Live_SAM_Address/FeatureServer/1/query"
)


@dataclass(frozen=True)
class AddressContextConfig:
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    sam_candidates: tuple[str, ...] = ("sam_addresses.csv", "sam_addresses.geojson")
    sam_url: str | None = SAM_ADDRESSES_URL
    sam_clean_output: Path = Path("data/processed/sam_addresses_clean.csv")


def _normalize_identifier(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    if re.fullmatch(r"\d+\.0+", text):
        return text.split(".", 1)[0]
    return normalize_string(text).replace(" ", "")


def _address_from_parts(df: pd.DataFrame) -> pd.Series:
    pieces = []
    for column in ["st_no", "street_number", "house_number", "st_name", "street_name", "street", "st_suffix", "street_suffix"]:
        if column in df.columns:
            pieces.append(df[column].fillna("").astype("string").str.strip())
    if not pieces:
        return pd.Series("", index=df.index, dtype="string")
    address = pieces[0]
    for piece in pieces[1:]:
        address = address.str.cat(piece, sep=" ")
    return address.str.replace(r"\s+", " ", regex=True).str.strip()


def clean_sam_addresses(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a SAM/geocoder address table without assuming one exact schema."""
    cleaned = df.copy()
    cleaned.columns = _standardize_columns(cleaned.columns.tolist())
    if "parcel" in cleaned.columns and "map_par_id" not in cleaned.columns:
        cleaned["map_par_id"] = cleaned["parcel"]
    if "mailing_neighborhood" in cleaned.columns and "neighborhood" not in cleaned.columns:
        cleaned["neighborhood"] = cleaned["mailing_neighborhood"]
    if "sam_address_id" in cleaned.columns and "address_id" not in cleaned.columns:
        cleaned["address_id"] = cleaned["sam_address_id"]

    address_col = next(
        (
            column
            for column in [
                "full_address",
                "address",
                "street_address",
                "site_address",
                "sam_address",
                "address_line_1",
            ]
            if column in cleaned.columns
        ),
        None,
    )
    zip_col = next(
        (column for column in ["zip_code", "zip", "postal_code", "postcode"] if column in cleaned.columns),
        None,
    )

    keep_candidates = [
        "sam_id",
        "address_id",
        "map_par_id",
        "loc_id",
        "gis_id",
        "pid",
        "full_address",
        "address",
        "street_address",
        "site_address",
        "address_line_1",
        "zip_code",
        "zip",
        "postal_code",
        "postcode",
        "neighborhood",
        "planning_district",
        "district",
        "ward",
        "latitude",
        "longitude",
        "x_coord",
        "y_coord",
    ]
    working = cleaned.loc[:, [column for column in keep_candidates if column in cleaned.columns]].copy()
    if address_col and address_col not in working.columns:
        working[address_col] = cleaned[address_col]
    if zip_col and zip_col not in working.columns:
        working[zip_col] = cleaned[zip_col]

    if address_col is not None:
        working["sam_address"] = cleaned[address_col].astype("string")
    else:
        working["sam_address"] = _address_from_parts(cleaned)

    if zip_col is not None:
        working["sam_zip"] = cleaned[zip_col].map(normalize_zip).astype("string")
    else:
        working["sam_zip"] = pd.Series(pd.NA, index=working.index, dtype="string")

    working["sam_address_clean"] = working["sam_address"].map(normalize_address).astype("string")
    working["sam_address_zip_key"] = build_address_zip_key(working, "sam_address", "sam_zip")
    working["sam_address_only_key"] = working["sam_address"].map(normalize_address).astype("string")

    for column in ["map_par_id", "loc_id", "gis_id", "pid"]:
        if column in working.columns:
            working[column] = working[column].map(_normalize_identifier).astype("string")

    for column in ["neighborhood", "planning_district", "district", "ward"]:
        if column in working.columns:
            working[column] = working[column].map(normalize_string).astype("string")

    dedupe_cols = [
        column
        for column in ["sam_address_zip_key", "sam_address_only_key", "map_par_id", "loc_id", "gis_id", "pid"]
        if column in working.columns
    ]
    if dedupe_cols:
        for column in dedupe_cols:
            non_empty = working[column].fillna("").astype("string").str.len().gt(0)
            deduped = working.loc[non_empty].drop_duplicates(subset=[column], keep="first")
            if not deduped.empty:
                working = deduped.reset_index(drop=True)
                break

    return working.reset_index(drop=True)


def load_sam_addresses(config: AddressContextConfig) -> pd.DataFrame | None:
    """Load SAM addresses from cache, local raw data, or the configured public endpoint."""
    raw_path = find_local_file(config.raw_dir, config.sam_candidates)
    if raw_path is None:
        existing = load_existing_clean_output(config.sam_clean_output)
        if existing is not None:
            cleaned_existing = clean_sam_addresses(existing)
            save_clean_output(cleaned_existing, config.sam_clean_output)
            return cleaned_existing
        fallback_path = config.raw_dir / config.sam_candidates[0]
        downloaded = download_arcgis_layer(config.sam_url, fallback_path)
        raw_path = downloaded or (fallback_path if fallback_path.exists() else None)
    if raw_path is None:
        print("SAM address data unavailable: no local extract or public endpoint response was found.")
        return None

    cleaned = clean_sam_addresses(load_local_tabular(raw_path))
    save_clean_output(cleaned, config.sam_clean_output)
    return cleaned
