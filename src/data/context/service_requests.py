"""311 service-request loading, cleaning, and aggregation helpers."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import io
from pathlib import Path
import re
from typing import Any

import pandas as pd
import requests

from src.data.features import normalize_address, normalize_string, normalize_zip
from src.data.violations import _standardize_columns

from .common import (
    build_address_zip_key,
    find_local_file,
    load_existing_clean_output,
    load_local_tabular,
    save_clean_output,
)


SERVICE_REQUESTS_PACKAGE_URL = "https://data.boston.gov/api/3/action/package_show?id=311-service-requests"
SERVICE_REQUESTS_SQL_API_URL = "https://data.boston.gov/api/3/action/datastore_search_sql"
HOUSING_RELATED_TOKENS = {
    "housing",
    "heat",
    "unsafe",
    "inspection",
    "rodent",
    "trash",
    "sanitation",
    "graffiti",
    "maintenance",
    "abandoned",
    "short term rental",
    "occupancy",
    "building",
}
SERVICE_REQUEST_USECOLS = {
    "case_enquiry_id",
    "case_id",
    "open_dt",
    "open_date",
    "closed_dt",
    "close_date",
    "case_status",
    "closure_reason",
    "closure_comments",
    "case_title",
    "subject",
    "reason",
    "type",
    "department",
    "assigned_department",
    "case_topic",
    "service_name",
    "location",
    "full_address",
    "street_number",
    "location_street_name",
    "street_name",
    "location_zipcode",
    "zip_code",
    "neighborhood",
    "ward",
    "precinct",
    "police_district",
    "fire_district",
    "city_council_district",
    "latitude",
    "longitude",
    "source",
    "report_source",
}
HISTORICAL_HOUSING_SERVICE_REQUEST_COLUMNS = (
    "case_enquiry_id",
    "open_dt",
    "case_status",
    "case_title",
    "subject",
    "reason",
    "type",
    "department",
    "neighborhood",
    "ward",
    "location_street_name",
    "location_zipcode",
    "latitude",
    "longitude",
)
HISTORICAL_HOUSING_FILTER_TERMS: dict[str, tuple[str, ...]] = {
    "subject": ("Housing", "Inspectional"),
    "reason": ("Housing", "Sanitation"),
    "type": ("Pest", "Rodent", "Heat", "Bed Bug", "Lead", "Illegal"),
}


@dataclass(frozen=True)
class ServiceRequestConfig:
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    candidates: tuple[str, ...] = (
        "service_requests_311.csv",
        "service_requests_311.xlsx",
        "service_requests_311.geojson",
    )
    historical_candidates: tuple[str, ...] = ("311_historical_housing.csv",)
    service_requests_metadata_url: str = SERVICE_REQUESTS_PACKAGE_URL
    clean_output_path: Path = Path("data/processed/service_requests_311_clean.csv")
    historical_clean_output_path: Path = Path("data/processed/service_requests_311_historical.csv")
    years_back: int = 1
    historical_start_year: int = 2015
    historical_end_year: int | None = None
    historical_page_size: int = 5000


def _normalize_identifier(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    if re.fullmatch(r"\d+\.0+", text):
        return text.split(".", 1)[0]
    return normalize_string(text).replace(" ", "")


def _service_request_usecol(column_name: str) -> bool:
    return _standardize_columns([column_name])[0] in SERVICE_REQUEST_USECOLS


def _select_service_request_resources(
    resources: list[dict[str, Any]],
    years: set[int],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for resource in resources:
        if str(resource.get("format", "")).upper() != "CSV":
            continue
        name = str(resource.get("name") or "").strip().upper()
        if "NEW SYSTEM" in name:
            selected.append(resource)
            continue
        if any(str(year) in name for year in years):
            selected.append(resource)
    return selected


def _service_request_resource_year(resource: dict[str, Any]) -> int | None:
    text = " ".join(
        str(resource.get(key) or "")
        for key in ["name", "description"]
    )
    match = re.search(r"\b(20\d{2})\b", text)
    if match is None:
        return None
    return int(match.group(1))


def _select_historical_service_request_resources(
    resources: list[dict[str, Any]],
    *,
    start_year: int,
    end_year: int,
) -> list[dict[str, Any]]:
    selected_by_year: dict[int, dict[str, Any]] = {}
    for resource in resources:
        if str(resource.get("format", "")).upper() != "CSV":
            continue
        if not bool(resource.get("datastore_active")):
            continue
        year = _service_request_resource_year(resource)
        if year is None or year < start_year or year > end_year:
            continue
        selected_by_year.setdefault(year, resource)
    return [selected_by_year[year] for year in sorted(selected_by_year)]


def _download_service_requests_bulk(
    config: ServiceRequestConfig,
    timeout: int = 60,
    *,
    force_refresh: bool = False,
) -> Path | None:
    target_path = config.raw_dir / config.candidates[0]
    if target_path.exists() and not force_refresh:
        return target_path

    try:
        response = requests.get(config.service_requests_metadata_url, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        resources = payload.get("result", {}).get("resources", [])
    except Exception as exc:
        print(f"Skipping 311 bulk download: {exc}")
        return None

    current_year = pd.Timestamp.today().year
    years = {current_year - offset for offset in range(config.years_back + 1)}
    selected = _select_service_request_resources(resources, years)
    if not selected:
        print("Skipping 311 bulk download: no matching CSV resources were found.")
        return None

    frames: list[pd.DataFrame] = []
    for resource in selected:
        url = resource.get("url")
        if not url:
            continue
        try:
            response = requests.get(
                str(url),
                timeout=timeout,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            response.raise_for_status()
            frame = pd.read_csv(
                io.StringIO(response.text),
                low_memory=False,
                usecols=_service_request_usecol,
            )
        except Exception as exc:
            print(f"Skipping 311 resource {resource.get('name') or resource.get('id')}: {exc}")
            continue
        if frame.empty:
            continue
        frame["source_resource_name"] = resource.get("name")
        frames.append(frame)

    if not frames:
        print("Skipping 311 bulk download: all matching resources failed to load.")
        return None

    combined = pd.concat(frames, ignore_index=True)
    dedupe_candidates = [column for column in ["case_enquiry_id", "case_id"] if column in combined.columns]
    if dedupe_candidates:
        for column in dedupe_candidates:
            non_null = combined[column].notna()
            combined = pd.concat(
                [
                    combined.loc[non_null].drop_duplicates(subset=[column], keep="first"),
                    combined.loc[~non_null],
                ],
                ignore_index=True,
            )

    target_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(target_path, index=False)
    return target_path


def _historical_housing_where_sql() -> str:
    clauses: list[str] = []
    for column, terms in HISTORICAL_HOUSING_FILTER_TERMS.items():
        for term in terms:
            escaped = term.replace("'", "''")
            clauses.append(f"{column} ILIKE '%{escaped}%'")
    return " OR ".join(clauses)


def _download_historical_housing_service_requests(
    config: ServiceRequestConfig,
    timeout: int = 60,
    *,
    force_refresh: bool = False,
) -> Path | None:
    target_path = config.raw_dir / config.historical_candidates[0]
    if target_path.exists() and not force_refresh:
        return target_path

    try:
        response = requests.get(config.service_requests_metadata_url, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        resources = payload.get("result", {}).get("resources", [])
    except Exception as exc:
        print(f"Skipping historical 311 download: {exc}")
        return None

    end_year = config.historical_end_year
    if end_year is None:
        end_year = pd.Timestamp.today().year - 1
    selected = _select_historical_service_request_resources(
        resources,
        start_year=config.historical_start_year,
        end_year=end_year,
    )
    if not selected:
        print("Skipping historical 311 download: no annual Datastore-backed CSV resources were found.")
        return None

    where_sql = _historical_housing_where_sql()
    target_path.parent.mkdir(parents=True, exist_ok=True)

    total_saved = 0
    with target_path.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(
            fout,
            fieldnames=[*HISTORICAL_HOUSING_SERVICE_REQUEST_COLUMNS, "source_year"],
            extrasaction="ignore",
        )
        writer.writeheader()

        for resource in selected:
            resource_id = resource.get("id")
            year = _service_request_resource_year(resource)
            if not resource_id or year is None:
                continue

            year_saved = 0
            offset = 0
            while True:
                sql = (
                    f'SELECT {", ".join(HISTORICAL_HOUSING_SERVICE_REQUEST_COLUMNS)} '
                    f'FROM "{resource_id}" '
                    f"WHERE ({where_sql}) "
                    f"ORDER BY open_dt "
                    f"LIMIT {config.historical_page_size} OFFSET {offset}"
                )
                try:
                    response = requests.get(
                        SERVICE_REQUESTS_SQL_API_URL,
                        params={"sql": sql},
                        timeout=timeout,
                        headers={"User-Agent": "Mozilla/5.0"},
                    )
                    response.raise_for_status()
                    payload = response.json()
                except Exception as exc:
                    print(f"Skipping historical 311 page for {year} at offset {offset}: {exc}")
                    break

                if not payload.get("success"):
                    print(f"Skipping historical 311 page for {year} at offset {offset}: API returned success=false")
                    break

                records = payload.get("result", {}).get("records", [])
                if not records:
                    break

                for record in records:
                    row = {column: record.get(column) for column in HISTORICAL_HOUSING_SERVICE_REQUEST_COLUMNS}
                    row["source_year"] = year
                    writer.writerow(row)

                batch_size = len(records)
                year_saved += batch_size
                total_saved += batch_size
                offset += config.historical_page_size
                if batch_size < config.historical_page_size:
                    break

            print(f"Saved {year_saved:,} housing-related 311 rows for {year}.")

    if total_saved == 0:
        target_path.unlink(missing_ok=True)
        print("Skipping historical 311 download: no housing-related rows were returned.")
        return None
    return target_path


def _should_refresh_service_request_raw(path: Path) -> bool:
    """Detect obviously broken bulk snapshots so they can be refreshed automatically."""
    try:
        preview = pd.read_csv(path, low_memory=False)
    except Exception:
        return True
    return "source_resource_name" in preview.columns and len(preview) < 1000


def _best_service_request_address(df: pd.DataFrame) -> pd.Series:
    """Build the cleanest available street address across legacy and new 311 schemas."""
    if "location_street_name" in df.columns:
        return df["location_street_name"].astype("string")
    if {"street_number", "street_name"}.issubset(df.columns):
        return (
            df["street_number"].fillna("").astype("string").str.strip().str.cat(
                df["street_name"].fillna("").astype("string").str.strip(),
                sep=" ",
            )
        ).str.replace(r"\s+", " ", regex=True).str.strip()
    for column in ["address", "street_address", "address_line_1", "full_address", "location"]:
        if column in df.columns:
            values = df[column].astype("string")
            if column == "full_address":
                return values.str.split(",", n=1).str[0].str.strip()
            return values
    return pd.Series(pd.NA, index=df.index, dtype="string")


def clean_service_requests(df: pd.DataFrame) -> pd.DataFrame:
    """Clean Boston 311/service-request exports for property-level aggregation."""
    cleaned = df.copy()
    cleaned.columns = _standardize_columns(cleaned.columns.tolist())

    zip_col = next(
        (
            column
            for column in [
                "location_zipcode",
                "zip_code",
                "zip",
                "postal_code",
            ]
            if column in cleaned.columns
        ),
        None,
    )
    created_col = next(
        (
            column
            for column in [
                "open_dt",
                "open_date",
                "submitted_date",
                "created_date",
                "case_created_date",
                "creation_date",
            ]
            if column in cleaned.columns
        ),
        None,
    )
    closed_col = next(
        (
            column
            for column in ["closed_dt", "close_date", "closure_date", "closed_date", "case_closed_date"]
            if column in cleaned.columns
        ),
        None,
    )

    for column in ["map_par_id", "loc_id", "gis_id", "pid"]:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].map(_normalize_identifier).astype("string")

    cleaned["service_request_address"] = _best_service_request_address(cleaned).map(normalize_address).astype("string")

    if zip_col:
        cleaned["address_zip_key"] = build_address_zip_key(cleaned, "service_request_address", zip_col)
    else:
        cleaned["address_zip_key"] = cleaned["service_request_address"].astype("string")

    if zip_col:
        cleaned["service_request_zip"] = cleaned[zip_col].map(normalize_zip).astype("string")

    if created_col:
        cleaned["service_request_open_date"] = pd.to_datetime(cleaned[created_col], errors="coerce")
    if closed_col:
        cleaned["service_request_closed_date"] = pd.to_datetime(cleaned[closed_col], errors="coerce")

    issue_text_cols = [
        column
        for column in [
            "reason",
            "type",
            "case_type",
            "case_topic",
            "service_name",
            "subject",
            "department",
            "assigned_department",
            "closure_reason",
            "closure_comments",
        ]
        if column in cleaned.columns
    ]
    if issue_text_cols:
        combined_text = cleaned[issue_text_cols].fillna("").astype("string").agg(" ".join, axis=1)
        normalized_text = combined_text.map(normalize_string)
        cleaned["housing_related_request_flag"] = normalized_text.apply(
            lambda text: int(any(token in text for token in HOUSING_RELATED_TOKENS))
        )
    else:
        cleaned["housing_related_request_flag"] = 0

    cleaned["service_request_record_count"] = 1
    return cleaned


def aggregate_service_requests(
    service_requests_df: pd.DataFrame,
    *,
    join_key: str,
    reference_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Aggregate 311 requests into compact property-level context features."""
    if join_key not in service_requests_df.columns:
        return pd.DataFrame(columns=[join_key])

    working = service_requests_df.dropna(subset=[join_key]).copy()
    if working.empty:
        return pd.DataFrame(columns=[join_key])

    if "service_request_open_date" in working.columns:
        working["service_request_open_date"] = pd.to_datetime(
            working["service_request_open_date"],
            errors="coerce",
        )
    if reference_date is not None:
        reference_date = pd.to_datetime(reference_date, errors="coerce")

    if reference_date is None and "service_request_open_date" in working.columns:
        reference_date = working["service_request_open_date"].max()

    if pd.notna(reference_date) and "service_request_open_date" in working.columns:
        recent_mask = working["service_request_open_date"].ge(reference_date - pd.Timedelta(days=365))
        working["service_requests_365d"] = recent_mask.fillna(False).astype(int)
        recency = (reference_date - working["service_request_open_date"]).dt.days
        working["days_since_service_request"] = recency.where(recency.ge(0))
    else:
        working["service_requests_365d"] = 0
        working["days_since_service_request"] = pd.NA

    aggregated = (
        working.groupby(join_key)
        .agg(
            service_request_count=("service_request_record_count", "sum"),
            housing_related_service_request_count=("housing_related_request_flag", "sum"),
            service_requests_365d=("service_requests_365d", "sum"),
            last_service_request_date=("service_request_open_date", "max"),
            min_days_since_service_request=("days_since_service_request", "min"),
        )
        .reset_index()
    )
    aggregated["housing_related_service_request_flag"] = (
        aggregated["housing_related_service_request_count"].fillna(0).gt(0).astype(int)
    )
    return aggregated


def load_service_requests(config: ServiceRequestConfig) -> pd.DataFrame | None:
    """Load a cleaned 311/service-request table when data is available."""
    raw_path = find_local_file(config.raw_dir, config.candidates)
    if raw_path is not None and raw_path.name == config.candidates[0] and _should_refresh_service_request_raw(raw_path):
        raw_path = _download_service_requests_bulk(config, force_refresh=True)
    if raw_path is None:
        raw_path = _download_service_requests_bulk(config)
    if raw_path is None:
        existing = load_existing_clean_output(config.clean_output_path)
        if existing is not None:
            cleaned_existing = clean_service_requests(existing)
            save_clean_output(cleaned_existing, config.clean_output_path)
            return cleaned_existing
    if raw_path is None:
        print("311 service request data unavailable: no local extract or public endpoint was found.")
        return None

    cleaned = clean_service_requests(load_local_tabular(raw_path))
    save_clean_output(cleaned, config.clean_output_path)
    return cleaned


def load_historical_service_requests(config: ServiceRequestConfig) -> pd.DataFrame | None:
    """Load a housing-focused historical 311 table for leak-safe temporal modeling."""
    raw_path = find_local_file(config.raw_dir, config.historical_candidates)
    if raw_path is None:
        raw_path = _download_historical_housing_service_requests(config)
    if raw_path is None:
        existing = load_existing_clean_output(config.historical_clean_output_path)
        if existing is not None:
            cleaned_existing = clean_service_requests(existing)
            save_clean_output(cleaned_existing, config.historical_clean_output_path)
            return cleaned_existing
        print("Historical 311 service request data unavailable: no local filtered extract or CKAN Datastore export was found.")
        return None

    cleaned = clean_service_requests(load_local_tabular(raw_path))
    save_clean_output(cleaned, config.historical_clean_output_path)
    return cleaned
