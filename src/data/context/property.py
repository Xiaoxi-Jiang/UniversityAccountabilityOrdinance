"""Optional property-data loaders and enrichment helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import pandas as pd

from src.data.features import normalize_address, normalize_string, normalize_zip
from src.data.violations import _standardize_columns

from .acs import ACSContextConfig, load_acs_context
from .address import AddressContextConfig, load_sam_addresses
from .common import (
    build_address_zip_key,
    find_local_file,
    load_existing_clean_output,
    load_local_tabular,
    save_clean_output,
)
from .permits import PermitContextConfig, aggregate_permits, load_permits
from .service_requests import (
    ServiceRequestConfig,
    aggregate_service_requests,
    load_service_requests,
)


PROPERTY_ASSESSMENT_URL = (
    "https://gis.boston.gov/arcgis/rest/services/Assessing/"
    "properties_boston_gov/FeatureServer/0/query"
)
PARCELS_CURRENT_URL = (
    "https://gis.boston.gov/arcgis/rest/services/Parcels/"
    "Parcels_current/FeatureServer/0/query"
)


@dataclass(frozen=True)
class PropertyDataConfig:
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    property_assessment_candidates: tuple[str, ...] = ("property_assessment_fy25.csv",)
    parcels_candidates: tuple[str, ...] = ("parcels_current.csv", "parcels_current.geojson")
    rentsmart_candidates: tuple[str, ...] = ("rentsmart.csv", "rentsmart.xlsx", "rentsmart.geojson")
    property_assessment_url: str | None = PROPERTY_ASSESSMENT_URL
    parcels_url: str | None = PARCELS_CURRENT_URL
    property_assessment_clean_output: Path = Path("data/processed/property_assessment_clean.csv")
    parcels_clean_output: Path = Path("data/processed/parcels_clean.csv")
    rentsmart_clean_output: Path = Path("data/processed/rentsmart_clean.csv")
    property_risk_output: Path = Path("data/processed/property_risk_table_v1.csv")


def normalize_owner_text(value: object) -> str:
    """Normalize owner text while carefully trimming common legal suffix noise."""
    text = normalize_string(value)
    if not text:
        return ""
    text = re_sub_legal_suffixes(text)
    return " ".join(text.split())


def re_sub_legal_suffixes(text: str) -> str:
    """Remove a small set of common trailing legal suffixes conservatively."""
    suffixes = [
        "llc",
        "inc",
        "corp",
        "corporation",
        "co",
        "company",
        "lp",
        "l p",
        "llp",
        "trust",
        "trs",
    ]
    pattern = rf"(?:\b(?:{'|'.join(suffixes)})\b\s*)+$"
    return pd.Series([text]).str.replace(pattern, "", regex=True).iloc[0].strip()


def normalize_property_identifier(value: object) -> str:
    """Normalize parcel/property identifiers for cross-source joins."""
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    if re.fullmatch(r"\d+\.0+", text):
        return text.split(".", 1)[0]
    return normalize_string(text).replace(" ", "")


def _load_remote_csv_or_layer(query_url: str | None, output_path: Path) -> Path | None:
    from .common import download_arcgis_layer

    return download_arcgis_layer(query_url, output_path)


def _load_context_source(
    *,
    raw_dir: Path,
    candidates: tuple[str, ...],
    clean_output_path: Path,
    remote_url: str | None,
    clean_fn,
    missing_message: str | None = None,
) -> pd.DataFrame | None:
    existing = load_existing_clean_output(clean_output_path)
    if existing is not None:
        cleaned_existing = clean_fn(existing)
        save_clean_output(cleaned_existing, clean_output_path)
        return cleaned_existing

    raw_path = find_local_file(raw_dir, candidates)
    if raw_path is None and remote_url:
        fallback_path = raw_dir / candidates[0]
        downloaded = _load_remote_csv_or_layer(remote_url, fallback_path)
        raw_path = downloaded or (fallback_path if fallback_path.exists() else None)
    if raw_path is None:
        if missing_message:
            print(missing_message)
        return None

    cleaned = clean_fn(load_local_tabular(raw_path))
    save_clean_output(cleaned, clean_output_path)
    return cleaned


def _coalesce_lookup(
    base_df: pd.DataFrame,
    join_df: pd.DataFrame,
    *,
    source_column: str,
    join_candidates: list[tuple[str, str]],
) -> pd.Series:
    values = pd.Series(pd.NA, index=base_df.index, dtype="object")
    if source_column not in join_df.columns:
        return values

    for left_col, right_col in join_candidates:
        if left_col not in base_df.columns or right_col not in join_df.columns:
            continue
        if source_column == right_col:
            valid_keys = (
                join_df[right_col]
                .dropna()
                .drop_duplicates()
                .astype("string")
            )
            candidate = base_df[left_col].astype("string")
            matched = candidate.where(candidate.isin(valid_keys))
            fill_mask = values.isna() & matched.notna()
            values.loc[fill_mask] = matched.loc[fill_mask]
            continue
        lookup = (
            join_df.dropna(subset=[right_col])
            .drop_duplicates(subset=[right_col], keep="first")
            .set_index(right_col)[source_column]
        )
        matched = base_df[left_col].map(lookup)
        fill_mask = values.isna() & matched.notna()
        values.loc[fill_mask] = matched.loc[fill_mask]
    return values


def _assign_lookup_columns(
    base_df: pd.DataFrame,
    join_df: pd.DataFrame | None,
    *,
    join_candidates: list[tuple[str, str]],
    source_columns: list[str],
    prefix: str = "",
) -> None:
    if join_df is None:
        return
    for column in source_columns:
        if column not in join_df.columns:
            continue
        base_df[f"{prefix}{column}"] = _coalesce_lookup(
            base_df,
            join_df,
            source_column=column,
            join_candidates=join_candidates,
        )


def _normalize_street_number(value: object) -> str:
    """Format a street number without float artifacts such as ``195.0``."""
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    if text.endswith(".0") and text.replace(".", "", 1).isdigit():
        return text[:-2]
    return text


def _positive_flag(series: pd.Series | None, index: pd.Index) -> pd.Series:
    """Convert an optional numeric-like series into a 0/1 availability flag."""
    if series is None:
        return pd.Series(0, index=index, dtype=int)
    numeric = pd.to_numeric(series, errors="coerce").fillna(0)
    return numeric.gt(0).astype(int)


def clean_property_assessment(df: pd.DataFrame) -> pd.DataFrame:
    """Clean Boston property assessment data for conservative downstream joins."""
    cleaned = df.copy()
    cleaned.columns = _standardize_columns(cleaned.columns.tolist())

    keep_cols = [
        "map_par_id",
        "loc_id",
        "gis_id",
        "pid",
        "full_address",
        "st_num",
        "st_name",
        "zip_code",
        "lu",
        "lu_desc",
        "bldg_type",
        "own_occ",
        "owner",
        "mail_addressee",
        "mail_street_address",
        "res_units",
        "land_sf",
        "gross_area",
        "living_area",
        "total_value",
        "gross_tax",
        "yr_built",
        "yr_remodel",
        "structure_class",
        "overall_cond",
    ]
    cleaned = cleaned.loc[:, [column for column in keep_cols if column in cleaned.columns]].copy()

    for column in ["map_par_id", "loc_id", "gis_id", "pid"]:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].map(normalize_property_identifier).astype("string")
    if "owner" in cleaned.columns:
        cleaned["owner_clean"] = cleaned["owner"].map(normalize_owner_text).astype("string")
        cleaned["owner_available_flag"] = cleaned["owner_clean"].fillna("").str.len().gt(0).astype(int)
    if {"st_num", "st_name"}.issubset(cleaned.columns):
        cleaned["property_address"] = (
            cleaned["st_num"].map(_normalize_street_number).astype("string").str.strip()
            + " "
            + cleaned["st_name"].fillna("").astype("string").str.strip()
        ).str.replace(r"\s+", " ", regex=True).str.strip()
    elif "full_address" in cleaned.columns:
        cleaned["property_address"] = cleaned["full_address"].astype("string")
    if "property_address" in cleaned.columns:
        cleaned["address_only_key"] = cleaned["property_address"].map(normalize_address).astype("string")
    if "property_address" in cleaned.columns and "zip_code" in cleaned.columns:
        cleaned["address_zip_key"] = build_address_zip_key(cleaned, "property_address", "zip_code")
    return cleaned


def clean_parcels(df: pd.DataFrame) -> pd.DataFrame:
    """Clean Boston parcels data while keeping geometry optional."""
    cleaned = df.copy()
    cleaned.columns = _standardize_columns(cleaned.columns.tolist())
    keep_cols = [
        "map_par_id",
        "loc_id",
        "poly_type",
        "map_no",
        "source",
        "plan_id",
        "shape_area",
        "shape_len",
    ]
    cleaned = cleaned.loc[:, [column for column in keep_cols if column in cleaned.columns]].copy()
    for column in ["map_par_id", "loc_id"]:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].map(normalize_property_identifier).astype("string")
    return cleaned


def clean_rentsmart(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a local RentSmart export when one is available."""
    cleaned = df.copy()
    cleaned.columns = _standardize_columns(cleaned.columns.tolist())

    owner_like = [column for column in cleaned.columns if "owner" in column or "landlord" in column]
    for column in owner_like:
        cleaned[f"{column}_clean"] = cleaned[column].map(normalize_owner_text).astype("string")

    address_col = next(
        (
            column
            for column in [
                "full_address",
                "address",
                "property_address",
                "street_address",
            ]
            if column in cleaned.columns
        ),
        None,
    )
    zip_col = next((column for column in ["zip_code", "zip", "postal_code"] if column in cleaned.columns), None)
    if address_col:
        cleaned["address_only_key"] = cleaned[address_col].map(normalize_address).astype("string")
    if address_col and zip_col:
        cleaned["address_zip_key"] = build_address_zip_key(cleaned, address_col, zip_col)
    for column in ["map_par_id", "loc_id", "gis_id", "pid"]:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].map(normalize_property_identifier).astype("string")
    return cleaned


def load_property_assessment(config: PropertyDataConfig) -> pd.DataFrame | None:
    """Load property assessment data from local cache or the configured public URL."""
    return _load_context_source(
        raw_dir=config.raw_dir,
        candidates=config.property_assessment_candidates,
        clean_output_path=config.property_assessment_clean_output,
        remote_url=config.property_assessment_url,
        clean_fn=clean_property_assessment,
    )


def load_parcels(config: PropertyDataConfig) -> pd.DataFrame | None:
    """Load parcel context from local cache or the configured public URL."""
    return _load_context_source(
        raw_dir=config.raw_dir,
        candidates=config.parcels_candidates,
        clean_output_path=config.parcels_clean_output,
        remote_url=config.parcels_url,
        clean_fn=clean_parcels,
    )


def load_rentsmart(config: PropertyDataConfig) -> pd.DataFrame | None:
    """Load a local RentSmart extract if the project team has one."""
    return _load_context_source(
        raw_dir=config.raw_dir,
        candidates=config.rentsmart_candidates,
        clean_output_path=config.rentsmart_clean_output,
        remote_url=None,
        clean_fn=clean_rentsmart,
        missing_message="RentSmart data unavailable: no local CSV/XLSX/GeoJSON extract was found.",
    )


def _aggregate_rentsmart_join(rentsmart_df: pd.DataFrame, join_key: str) -> pd.DataFrame:
    """Aggregate RentSmart records to a join key with simple risk indicators."""
    working = rentsmart_df.dropna(subset=[join_key]).copy()
    if working.empty:
        return pd.DataFrame(columns=[join_key, "rentsmart_record_count"])

    complaint_cols = [
        column
        for column in working.columns
        if any(token in column for token in ["complaint", "violation", "inspection", "issue"])
    ]
    complaint_signal = working[complaint_cols].notna().any(axis=1).astype(int) if complaint_cols else 0

    aggregated = (
        working.assign(
            rentsmart_record_count=1,
            rentsmart_history_flag=1,
            rentsmart_complaint_indicator=complaint_signal,
        )
        .groupby(join_key)
        .agg(
            rentsmart_record_count=("rentsmart_record_count", "sum"),
            rentsmart_history_flag=("rentsmart_history_flag", "max"),
            rentsmart_complaint_indicator=("rentsmart_complaint_indicator", "max"),
        )
        .reset_index()
    )
    return aggregated


def _context_configs(config: PropertyDataConfig) -> tuple[AddressContextConfig, ServiceRequestConfig, PermitContextConfig, ACSContextConfig]:
    processed_dir = config.processed_dir
    return (
        AddressContextConfig(
            raw_dir=config.raw_dir,
            processed_dir=processed_dir,
            sam_clean_output=processed_dir / "sam_addresses_clean.csv",
        ),
        ServiceRequestConfig(
            raw_dir=config.raw_dir,
            processed_dir=processed_dir,
            clean_output_path=processed_dir / "service_requests_311_clean.csv",
        ),
        PermitContextConfig(
            raw_dir=config.raw_dir,
            processed_dir=processed_dir,
            clean_output_path=processed_dir / "building_permits_clean.csv",
        ),
        ACSContextConfig(
            raw_dir=config.raw_dir,
            processed_dir=processed_dir,
            clean_output_path=processed_dir / "acs_context_clean.csv",
        ),
    )


def build_property_risk_table(
    feature_df: pd.DataFrame,
    property_assessment_df: pd.DataFrame | None,
    parcels_df: pd.DataFrame | None,
    rentsmart_df: pd.DataFrame | None,
    sam_df: pd.DataFrame | None = None,
    service_requests_df: pd.DataFrame | None = None,
    permits_df: pd.DataFrame | None = None,
    acs_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join violations features to optional property-risk context layers."""
    risk_df = feature_df.copy()
    if {"violation_st", "violation_zip"}.issubset(risk_df.columns):
        risk_df["address_zip_key"] = build_address_zip_key(risk_df, "violation_st", "violation_zip")
        risk_df["address_only_key"] = risk_df["violation_st"].map(normalize_address).astype("string")
    else:
        risk_df["address_zip_key"] = pd.Series(pd.NA, index=risk_df.index, dtype="string")
        risk_df["address_only_key"] = pd.Series(pd.NA, index=risk_df.index, dtype="string")

    diagnostics: dict[str, Any] = {
        "sam_loaded": sam_df is not None,
        "property_assessment_loaded": property_assessment_df is not None,
        "parcels_loaded": parcels_df is not None,
        "rentsmart_loaded": rentsmart_df is not None,
        "service_requests_loaded": service_requests_df is not None,
        "permits_loaded": permits_df is not None,
        "acs_loaded": acs_df is not None,
        "assessment_join_type": "identifier_exact_via_sam_or_address_fallback",
        "parcels_join_type": "identifier_exact_via_assessment_or_sam",
        "rentsmart_join_type": "identifier_or_address_fallback",
        "service_requests_join_type": "identifier_or_address_fallback",
        "permits_join_type": "identifier_or_address_fallback",
        "acs_join_type": "zip_exact",
    }

    if sam_df is not None:
        sam_cols = [
            column
            for column in [
                "sam_address_zip_key",
                "sam_address_only_key",
                "map_par_id",
                "loc_id",
                "gis_id",
                "pid",
                "neighborhood",
                "planning_district",
                "district",
                "ward",
                "latitude",
                "longitude",
            ]
            if column in sam_df.columns
        ]
        _assign_lookup_columns(
            risk_df,
            sam_df.loc[:, sam_cols].copy(),
            join_candidates=[
                ("address_zip_key", "sam_address_zip_key"),
                ("address_only_key", "sam_address_only_key"),
            ],
            source_columns=[column for column in sam_cols if column not in {"sam_address_zip_key", "sam_address_only_key"}],
            prefix="sam_",
        )

    if property_assessment_df is not None:
        assessment_cols = [
            column
            for column in [
                "map_par_id",
                "loc_id",
                "gis_id",
                "pid",
                "lu",
                "lu_desc",
                "bldg_type",
                "own_occ",
                "owner",
                "owner_clean",
                "owner_available_flag",
                "res_units",
                "gross_area",
                "living_area",
                "total_value",
                "gross_tax",
                "yr_built",
                "overall_cond",
            ]
            if column in property_assessment_df.columns
        ]
        _assign_lookup_columns(
            risk_df,
            property_assessment_df.loc[:, [column for column in ["address_zip_key", "address_only_key", *assessment_cols] if column in property_assessment_df.columns]].copy(),
            join_candidates=[
                ("sam_map_par_id", "map_par_id"),
                ("sam_loc_id", "loc_id"),
                ("sam_gis_id", "gis_id"),
                ("sam_pid", "pid"),
                ("address_zip_key", "address_zip_key"),
                ("address_only_key", "address_only_key"),
            ],
            source_columns=assessment_cols,
            prefix="assessment_",
        )

    if parcels_df is not None:
        parcel_cols = [column for column in ["map_par_id", "loc_id", "poly_type", "source", "plan_id", "shape_area"] if column in parcels_df.columns]
        _assign_lookup_columns(
            risk_df,
            parcels_df.loc[:, parcel_cols].copy(),
            join_candidates=[
                ("assessment_loc_id", "loc_id"),
                ("assessment_map_par_id", "map_par_id"),
                ("sam_loc_id", "loc_id"),
                ("sam_map_par_id", "map_par_id"),
            ],
            source_columns=[column for column in parcel_cols if column not in {"map_par_id", "loc_id"}],
            prefix="parcel_",
        )

    reference_date = None
    if "reference_date" in risk_df.columns and risk_df["reference_date"].notna().any():
        reference_date = pd.to_datetime(risk_df["reference_date"], errors="coerce").max()
    if reference_date is None and "last_violation_date" in risk_df.columns:
        reference_date = pd.to_datetime(risk_df["last_violation_date"], errors="coerce").max()

    if rentsmart_df is not None:
        rentsmart_candidates = [
            ("assessment_map_par_id", "map_par_id"),
            ("assessment_loc_id", "loc_id"),
            ("assessment_gis_id", "gis_id"),
            ("assessment_pid", "pid"),
            ("sam_map_par_id", "map_par_id"),
            ("sam_loc_id", "loc_id"),
            ("address_zip_key", "address_zip_key"),
            ("address_only_key", "address_only_key"),
        ]
        aggregated_frames: list[tuple[str, pd.DataFrame]] = []
        for left_col, right_col in rentsmart_candidates:
            if left_col in risk_df.columns and right_col in rentsmart_df.columns:
                aggregated_frames.append(
                    (
                        left_col,
                        _aggregate_rentsmart_join(rentsmart_df, right_col).rename(columns={right_col: left_col}),
                    )
                )
        joined = False
        for left_col, aggregated in aggregated_frames:
            if left_col not in aggregated.columns or aggregated.empty:
                continue
            risk_df = risk_df.merge(aggregated, on=left_col, how="left")
            joined = True
            if left_col.startswith("assessment_") or left_col.startswith("sam_"):
                diagnostics["rentsmart_join_type"] = "identifier_exact"
            elif left_col == "address_zip_key":
                diagnostics["rentsmart_join_type"] = "address_zip_approximate"
            break
        if not joined:
            diagnostics["rentsmart_join_type"] = "skipped"

    if service_requests_df is not None:
        service_candidates = [
            ("sam_map_par_id", "map_par_id"),
            ("sam_loc_id", "loc_id"),
            ("address_zip_key", "address_zip_key"),
        ]
        _assign_lookup_columns(
            risk_df,
            pd.concat(
                [
                    aggregate_service_requests(service_requests_df, join_key=right_col, reference_date=reference_date).rename(columns={right_col: left_col})
                    for left_col, right_col in service_candidates
                    if right_col in service_requests_df.columns and left_col in risk_df.columns
                ],
                ignore_index=False,
            )
            if any(right_col in service_requests_df.columns and left_col in risk_df.columns for left_col, right_col in service_candidates)
            else None,
            join_candidates=[(left_col, left_col) for left_col, right_col in service_candidates if left_col in risk_df.columns],
            source_columns=[
                "service_request_count",
                "housing_related_service_request_count",
                "service_requests_365d",
                "last_service_request_date",
                "min_days_since_service_request",
                "housing_related_service_request_flag",
            ],
        )
        if "service_request_count" not in risk_df.columns:
            diagnostics["service_requests_join_type"] = "skipped"
        elif pd.to_numeric(risk_df["service_request_count"], errors="coerce").fillna(0).gt(0).any():
            diagnostics["service_requests_join_type"] = "identifier_or_address_joined"

    if permits_df is not None:
        permit_candidates = [
            ("assessment_map_par_id", "map_par_id"),
            ("assessment_pid", "pid"),
            ("sam_map_par_id", "map_par_id"),
            ("sam_loc_id", "loc_id"),
            ("address_zip_key", "address_zip_key"),
        ]
        _assign_lookup_columns(
            risk_df,
            pd.concat(
                [
                    aggregate_permits(permits_df, join_key=right_col, reference_date=reference_date).rename(columns={right_col: left_col})
                    for left_col, right_col in permit_candidates
                    if right_col in permits_df.columns and left_col in risk_df.columns
                ],
                ignore_index=False,
            )
            if any(right_col in permits_df.columns and left_col in risk_df.columns for left_col, right_col in permit_candidates)
            else None,
            join_candidates=[(left_col, left_col) for left_col, right_col in permit_candidates if left_col in risk_df.columns],
            source_columns=[
                "permit_count",
                "major_permit_count",
                "occupancy_related_permit_count",
                "permits_730d",
                "last_permit_date",
                "min_days_since_permit",
                "major_permit_flag",
                "occupancy_related_permit_flag",
            ],
        )
        if "permit_count" not in risk_df.columns:
            diagnostics["permits_join_type"] = "skipped"
        elif pd.to_numeric(risk_df["permit_count"], errors="coerce").fillna(0).gt(0).any():
            diagnostics["permits_join_type"] = "identifier_or_address_joined"

    if acs_df is not None and {"violation_zip", "acs_zip"}.issubset(risk_df.columns.union(acs_df.columns)):
        risk_df = risk_df.merge(acs_df, left_on="violation_zip", right_on="acs_zip", how="left")

    risk_df["sam_match_flag"] = (
        risk_df.filter(regex=r"^sam_(map_par_id|loc_id|gis_id|pid|neighborhood)$").notna().any(axis=1).astype(int)
        if not risk_df.filter(regex=r"^sam_(map_par_id|loc_id|gis_id|pid|neighborhood)$").empty
        else 0
    )
    risk_df["assessment_match_flag"] = (
        risk_df.filter(regex=r"^assessment_(map_par_id|loc_id|gis_id|pid)$").notna().any(axis=1).astype(int)
        if not risk_df.filter(regex=r"^assessment_(map_par_id|loc_id|gis_id|pid)$").empty
        else 0
    )
    risk_df["parcel_context_flag"] = (
        risk_df.filter(regex=r"^parcel_").notna().any(axis=1).astype(int)
        if not risk_df.filter(regex=r"^parcel_").empty
        else 0
    )
    risk_df["owner_data_available_flag"] = (
        risk_df["assessment_owner_clean"].fillna("").astype("string").str.len().gt(0).astype(int)
        if "assessment_owner_clean" in risk_df.columns
        else 0
    )
    risk_df["rentsmart_match_flag"] = _positive_flag(risk_df.get("rentsmart_record_count"), risk_df.index)
    risk_df["service_request_context_flag"] = _positive_flag(risk_df.get("service_request_count"), risk_df.index)
    risk_df["permit_context_flag"] = _positive_flag(risk_df.get("permit_count"), risk_df.index)
    risk_df["acs_context_flag"] = (
        risk_df.filter(regex=r"^acs_").notna().any(axis=1).astype(int)
        if not risk_df.filter(regex=r"^acs_").empty
        else 0
    )

    diagnostics["sam_match_rate_pct"] = round(float(risk_df["sam_match_flag"].mean() * 100), 1)
    diagnostics["assessment_match_rate_pct"] = round(float(risk_df["assessment_match_flag"].mean() * 100), 1)
    diagnostics["parcel_match_rate_pct"] = round(float(risk_df["parcel_context_flag"].mean() * 100), 1)
    diagnostics["rentsmart_match_rate_pct"] = round(float(risk_df["rentsmart_match_flag"].mean() * 100), 1)
    diagnostics["service_request_context_rate_pct"] = round(float(risk_df["service_request_context_flag"].mean() * 100), 1)
    diagnostics["permit_context_rate_pct"] = round(float(risk_df["permit_context_flag"].mean() * 100), 1)
    diagnostics["acs_context_rate_pct"] = round(float(risk_df["acs_context_flag"].mean() * 100), 1)
    diagnostics["owner_data_rate_pct"] = round(float(risk_df["owner_data_available_flag"].mean() * 100), 1)
    diagnostics["parcel_context_rate_pct"] = diagnostics["parcel_match_rate_pct"]
    return risk_df, diagnostics


def save_property_risk_table(config: PropertyDataConfig, feature_table_path: Path) -> tuple[Path, dict[str, Any]]:
    """Build and save the enriched property-risk table."""
    feature_df = pd.read_csv(feature_table_path, low_memory=False)
    property_assessment_df = load_property_assessment(config)
    parcels_df = load_parcels(config)
    rentsmart_df = load_rentsmart(config)
    address_config, service_config, permit_config, acs_config = _context_configs(config)
    sam_df = load_sam_addresses(address_config)
    service_requests_df = load_service_requests(service_config)
    permits_df = load_permits(permit_config)
    acs_df = load_acs_context(acs_config)

    risk_df, diagnostics = build_property_risk_table(
        feature_df,
        property_assessment_df=property_assessment_df,
        parcels_df=parcels_df,
        rentsmart_df=rentsmart_df,
        sam_df=sam_df,
        service_requests_df=service_requests_df,
        permits_df=permits_df,
        acs_df=acs_df,
    )
    config.property_risk_output.parent.mkdir(parents=True, exist_ok=True)
    risk_df.to_csv(config.property_risk_output, index=False)
    print(f"Property risk table saved to: {config.property_risk_output}")
    return config.property_risk_output, diagnostics
