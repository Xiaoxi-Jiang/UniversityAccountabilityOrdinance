"""Shared loaders and join helpers for optional context datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.data.features import normalize_address, normalize_zip


def download_arcgis_layer(
    query_url: str | None,
    output_path: Path,
    timeout: int = 60,
) -> Path | None:
    """Download a paginated ArcGIS layer to CSV and cache it locally."""
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
        print(f"Skipping remote context-data download for {output_path.name}: {exc}")
        return None

    if not features:
        print(f"Skipping remote context-data download for {output_path.name}: no records returned.")
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(features).to_csv(output_path, index=False)
    return output_path


def load_local_tabular(path: Path) -> pd.DataFrame:
    """Load CSV, XLSX, JSON, or GeoJSON tabular data."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix in {".json", ".geojson"}:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "features" in payload:
            rows = [feature.get("properties", {}) for feature in payload.get("features", [])]
            return pd.DataFrame(rows)
        return pd.DataFrame(payload)
    raise ValueError(f"Unsupported file format for {path}")


def find_local_file(raw_dir: Path, candidates: tuple[str, ...]) -> Path | None:
    """Return the first candidate file that exists under ``raw_dir``."""
    for candidate in candidates:
        path = raw_dir / candidate
        if path.exists():
            return path
    return None


def save_clean_output(df: pd.DataFrame | None, output_path: Path) -> Path | None:
    """Persist a cleaned table when one is available."""
    if df is None:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved clean table: {output_path}")
    return output_path


def load_existing_clean_output(output_path: Path) -> pd.DataFrame | None:
    """Load a cached cleaned table when it already exists."""
    if output_path.exists():
        return pd.read_csv(output_path, low_memory=False)
    return None


def build_address_zip_key_from_series(
    address: pd.Series,
    zip_code: pd.Series,
) -> pd.Series:
    """Create a normalized address+ZIP join key from two series."""
    normalized_address = address.map(normalize_address).astype("string")
    normalized_zip = zip_code.map(normalize_zip).astype("string")
    key = normalized_address.str.cat(normalized_zip, sep="|")
    return key.str.strip("|").astype("string")


def build_address_zip_key(df: pd.DataFrame, address_col: str, zip_col: str) -> pd.Series:
    """Create a normalized address+ZIP join key from DataFrame columns."""
    return build_address_zip_key_from_series(df[address_col], df[zip_col])
