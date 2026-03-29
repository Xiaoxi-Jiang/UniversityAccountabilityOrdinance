"""Optional property-data loaders and enrichment helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.data.features import normalize_address, normalize_string, normalize_zip
from src.data.violations import _standardize_columns


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
    text = normalize_string(value).replace(" ", "")
    return text


def _download_arcgis_layer(
    query_url: str | None,
    output_path: Path,
    timeout: int = 60,
) -> Path | None:
    """Download an ArcGIS layer to CSV and cache it locally."""
    if output_path.exists():
        return output_path
    if not query_url:
        return None

    features: list[dict[str, Any]] = []
    result_offset = 0
    result_record_count = 2000

    try:
        while True:
            response = requests.get(
                query_url,
                params={
                    "where": "1=1",
                    "outFields": "*",
                    "returnGeometry": "false",
                    "f": "json",
                    "resultOffset": result_offset,
                    "resultRecordCount": result_record_count,
                },
                timeout=timeout,
            )
            response.raise_for_status()
            payload = response.json()
            batch = payload.get("features", [])
            features.extend(feature.get("attributes", {}) for feature in batch)
            if len(batch) < result_record_count:
                break
            result_offset += result_record_count
    except Exception as exc:
        print(f"Skipping remote property-data download for {output_path.name}: {exc}")
        return None

    if not features:
        print(f"Skipping remote property-data download for {output_path.name}: no records returned.")
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(features).to_csv(output_path, index=False)
    return output_path


def _load_local_tabular(path: Path) -> pd.DataFrame:
    """Load CSV, XLSX, JSON, or GeoJSON tabular data."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix in {".json", ".geojson"}:
        payload = pd.read_json(path)
        if "features" in payload.columns:
            features = payload["features"].tolist()
            rows = []
            for feature in features:
                rows.append(feature.get("properties", {}))
            return pd.DataFrame(rows)
        return payload
    raise ValueError(f"Unsupported file format for {path}")


def _find_local_file(raw_dir: Path, candidates: tuple[str, ...]) -> Path | None:
    for candidate in candidates:
        path = raw_dir / candidate
        if path.exists():
            return path
    return None


def _save_clean_output(df: pd.DataFrame | None, output_path: Path) -> Path | None:
    if df is None:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved clean table: {output_path}")
    return output_path


def _load_existing_clean_output(output_path: Path) -> pd.DataFrame | None:
    if output_path.exists():
        return pd.read_csv(output_path)
    return None


def _build_address_zip_key(df: pd.DataFrame, address_col: str, zip_col: str) -> pd.Series:
    address = df[address_col].map(normalize_address).astype("string")
    zip_code = df[zip_col].map(normalize_zip).astype("string")
    key = address.str.cat(zip_code, sep="|")
    return key.str.strip("|").astype("string")


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
            cleaned["st_num"].fillna("").astype("string").str.strip()
            + " "
            + cleaned["st_name"].fillna("").astype("string").str.strip()
        ).str.replace(r"\s+", " ", regex=True).str.strip()
    elif "full_address" in cleaned.columns:
        cleaned["property_address"] = cleaned["full_address"].astype("string")
    if "property_address" in cleaned.columns and "zip_code" in cleaned.columns:
        cleaned["address_zip_key"] = _build_address_zip_key(cleaned, "property_address", "zip_code")
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
    if address_col and zip_col:
        cleaned["address_zip_key"] = _build_address_zip_key(cleaned, address_col, zip_col)
    for column in ["map_par_id", "loc_id", "gis_id", "pid"]:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].map(normalize_property_identifier).astype("string")
    return cleaned


def load_property_assessment(config: PropertyDataConfig) -> pd.DataFrame | None:
    """Load property assessment data from local cache or the configured public URL."""
    existing = _load_existing_clean_output(config.property_assessment_clean_output)
    if existing is not None:
        return existing

    raw_path = _find_local_file(config.raw_dir, config.property_assessment_candidates)
    if raw_path is None:
        fallback_path = config.raw_dir / config.property_assessment_candidates[0]
        downloaded = _download_arcgis_layer(config.property_assessment_url, fallback_path)
        raw_path = downloaded or (fallback_path if fallback_path.exists() else None)
    if raw_path is None:
        return None
    cleaned = clean_property_assessment(_load_local_tabular(raw_path))
    _save_clean_output(cleaned, config.property_assessment_clean_output)
    return cleaned


def load_parcels(config: PropertyDataConfig) -> pd.DataFrame | None:
    """Load parcel context from local cache or the configured public URL."""
    existing = _load_existing_clean_output(config.parcels_clean_output)
    if existing is not None:
        return existing

    raw_path = _find_local_file(config.raw_dir, config.parcels_candidates)
    if raw_path is None:
        fallback_path = config.raw_dir / config.parcels_candidates[0]
        downloaded = _download_arcgis_layer(config.parcels_url, fallback_path)
        raw_path = downloaded or (fallback_path if fallback_path.exists() else None)
    if raw_path is None:
        return None
    cleaned = clean_parcels(_load_local_tabular(raw_path))
    _save_clean_output(cleaned, config.parcels_clean_output)
    return cleaned


def load_rentsmart(config: PropertyDataConfig) -> pd.DataFrame | None:
    """Load a local RentSmart extract if the project team has one."""
    existing = _load_existing_clean_output(config.rentsmart_clean_output)
    if existing is not None:
        return existing

    raw_path = _find_local_file(config.raw_dir, config.rentsmart_candidates)
    if raw_path is None:
        print("RentSmart data unavailable: no local CSV/XLSX/GeoJSON extract was found.")
        return None
    cleaned = clean_rentsmart(_load_local_tabular(raw_path))
    _save_clean_output(cleaned, config.rentsmart_clean_output)
    return cleaned


def _deduplicate_join_frame(df: pd.DataFrame, key_columns: list[str]) -> pd.DataFrame:
    valid_keys = [column for column in key_columns if column in df.columns]
    if not valid_keys:
        return df
    for column in valid_keys:
        deduped = df.dropna(subset=[column]).drop_duplicates(subset=[column], keep="first")
        if not deduped.empty:
            return deduped
    return df.drop_duplicates(keep="first")


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

    if complaint_cols:
        complaint_signal = working[complaint_cols].notna().any(axis=1).astype(int)
    else:
        complaint_signal = pd.Series(0, index=working.index)

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


def build_property_risk_table(
    feature_df: pd.DataFrame,
    property_assessment_df: pd.DataFrame | None,
    parcels_df: pd.DataFrame | None,
    rentsmart_df: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join violations features to optional property-risk context layers."""
    risk_df = feature_df.copy()
    if {"violation_st", "violation_zip"}.issubset(risk_df.columns):
        risk_df["address_zip_key"] = _build_address_zip_key(risk_df, "violation_st", "violation_zip")
    else:
        risk_df["address_zip_key"] = pd.Series(pd.NA, index=risk_df.index, dtype="string")

    diagnostics: dict[str, Any] = {
        "property_assessment_loaded": property_assessment_df is not None,
        "parcels_loaded": parcels_df is not None,
        "rentsmart_loaded": rentsmart_df is not None,
        "assessment_join_type": "address_zip_approximate",
        "parcels_join_type": "identifier_exact_via_assessment" if parcels_df is not None else "skipped",
        "rentsmart_join_type": "identifier_or_address_fallback",
    }

    if property_assessment_df is not None:
        assessment_cols = [
            "address_zip_key",
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
        available = [column for column in assessment_cols if column in property_assessment_df.columns]
        assessment_join = _deduplicate_join_frame(
            property_assessment_df.loc[:, available].copy(),
            ["address_zip_key", "map_par_id", "loc_id", "gis_id", "pid"],
        )
        assessment_join = assessment_join.rename(
            columns={column: f"assessment_{column}" for column in assessment_join.columns if column != "address_zip_key"}
        )
        risk_df = risk_df.merge(assessment_join, on="address_zip_key", how="left")

    if parcels_df is not None and "assessment_loc_id" in risk_df.columns:
        parcel_cols = [
            column
            for column in ["loc_id", "map_par_id", "poly_type", "source", "plan_id", "shape_area"]
            if column in parcels_df.columns
        ]
        parcel_join = _deduplicate_join_frame(parcels_df.loc[:, parcel_cols].copy(), ["loc_id", "map_par_id"])
        parcel_join = parcel_join.rename(
            columns={column: f"parcel_{column}" for column in parcel_join.columns if column not in {"loc_id", "map_par_id"}}
        )
        risk_df = risk_df.merge(
            parcel_join,
            left_on="assessment_loc_id",
            right_on="loc_id",
            how="left",
        )

    if rentsmart_df is not None:
        joined = False
        for candidate in ["map_par_id", "loc_id", "gis_id", "pid"]:
            assessment_candidate = f"assessment_{candidate}"
            if candidate in rentsmart_df.columns and assessment_candidate in risk_df.columns:
                rentsmart_agg = _aggregate_rentsmart_join(rentsmart_df, candidate).rename(
                    columns={candidate: assessment_candidate}
                )
                risk_df = risk_df.merge(rentsmart_agg, on=assessment_candidate, how="left")
                diagnostics["rentsmart_join_type"] = "identifier_exact"
                joined = True
                break
        if not joined and "address_zip_key" in rentsmart_df.columns:
            rentsmart_agg = _aggregate_rentsmart_join(rentsmart_df, "address_zip_key")
            risk_df = risk_df.merge(rentsmart_agg, on="address_zip_key", how="left")
            diagnostics["rentsmart_join_type"] = "address_zip_approximate"

    risk_df["assessment_match_flag"] = (
        risk_df.filter(regex=r"^assessment_(map_par_id|loc_id|gis_id|pid)$").notna().any(axis=1).astype(int)
        if not risk_df.filter(regex=r"^assessment_(map_par_id|loc_id|gis_id|pid)$").empty
        else 0
    )
    risk_df["parcel_context_flag"] = (
        risk_df.filter(regex=r"^(loc_id|parcel_)").notna().any(axis=1).astype(int)
        if not risk_df.filter(regex=r"^(loc_id|parcel_)").empty
        else 0
    )
    risk_df["owner_data_available_flag"] = (
        risk_df["assessment_owner_clean"].fillna("").astype("string").str.len().gt(0).astype(int)
        if "assessment_owner_clean" in risk_df.columns
        else 0
    )
    if "rentsmart_record_count" in risk_df.columns:
        risk_df["rentsmart_match_flag"] = risk_df["rentsmart_record_count"].fillna(0).gt(0).astype(int)
    else:
        risk_df["rentsmart_match_flag"] = 0

    diagnostics["assessment_match_rate_pct"] = round(float(risk_df["assessment_match_flag"].mean() * 100), 1)
    diagnostics["parcel_match_rate_pct"] = round(float(risk_df["parcel_context_flag"].mean() * 100), 1)
    diagnostics["rentsmart_match_rate_pct"] = round(float(risk_df["rentsmart_match_flag"].mean() * 100), 1)
    diagnostics["owner_data_rate_pct"] = round(float(risk_df["owner_data_available_flag"].mean() * 100), 1)
    diagnostics["parcel_context_rate_pct"] = diagnostics["parcel_match_rate_pct"]

    return risk_df, diagnostics


def save_property_risk_table(config: PropertyDataConfig, feature_table_path: Path) -> tuple[Path, dict[str, Any]]:
    """Build and save the enriched property-risk table."""
    feature_df = pd.read_csv(feature_table_path)
    property_assessment_df = load_property_assessment(config)
    parcels_df = load_parcels(config)
    rentsmart_df = load_rentsmart(config)
    risk_df, diagnostics = build_property_risk_table(
        feature_df,
        property_assessment_df=property_assessment_df,
        parcels_df=parcels_df,
        rentsmart_df=rentsmart_df,
    )
    config.property_risk_output.parent.mkdir(parents=True, exist_ok=True)
    risk_df.to_csv(config.property_risk_output, index=False)
    print(f"Property risk table saved to: {config.property_risk_output}")
    return config.property_risk_output, diagnostics
