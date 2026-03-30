"""Shared loaders and join helpers for optional context datasets."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import time
from typing import Any

import pandas as pd
import requests

from src.data.features import normalize_address, normalize_zip


def _retry_delay_seconds(attempt: int) -> int:
    return min(2 ** max(attempt - 1, 0), 8)


def _chunk_values(values: list[int], chunk_size: int) -> list[list[int]]:
    return [values[index : index + chunk_size] for index in range(0, len(values), chunk_size)]


def _request_arcgis_json(
    query_url: str,
    *,
    params: dict[str, Any],
    timeout: int,
    label: str,
) -> dict[str, Any]:
    response = requests.get(query_url, params=params, timeout=timeout)
    print(f"  Response status for {label}: {response.status_code}", flush=True)
    response.raise_for_status()
    return response.json()


def _normalize_object_ids(payload: dict[str, Any]) -> list[int]:
    object_ids = payload.get("objectIds", [])
    normalized: list[int] = []
    for object_id in object_ids:
        if object_id is None:
            continue
        normalized.append(int(object_id))
    return sorted(set(normalized))


def _download_arcgis_chunk_by_ids(
    query_url: str,
    *,
    object_ids: list[int],
    timeout: int,
    chunk_index: int,
    total_chunks: int,
    max_attempts: int,
    object_id_field: str | None,
) -> tuple[int, list[dict[str, Any]]]:
    params = {
        "objectIds": ",".join(str(object_id) for object_id in object_ids),
        "outFields": "*",
        "returnGeometry": "false",
        "f": "json",
    }

    last_problem: str | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.post(query_url, data=params, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            features = payload.get("features", [])
            rows = [feature.get("attributes", {}) for feature in features]

            if object_id_field is not None:
                returned_object_ids = {
                    int(row[object_id_field])
                    for row in rows
                    if row.get(object_id_field) is not None
                }
                missing_object_ids = set(object_ids) - returned_object_ids
                if not missing_object_ids:
                    return response.status_code, features
                last_problem = (
                    f"missing {len(missing_object_ids)} of {len(object_ids)} requested object IDs"
                )
            else:
                if rows:
                    return response.status_code, features
                last_problem = f"received 0 rows for {len(object_ids)} requested object IDs"

            if attempt < max_attempts:
                delay = _retry_delay_seconds(attempt)
                print(
                    f"  Chunk {chunk_index}/{total_chunks} returned an incomplete payload "
                    f"on attempt {attempt}/{max_attempts} ({last_problem}); retrying in {delay}s...",
                    flush=True,
                )
                time.sleep(delay)
                continue
        except requests.exceptions.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else "unknown"
            retryable = status_code in {429, 500, 502, 503, 504}
            if attempt >= max_attempts or not retryable:
                last_problem = f"HTTP status {status_code}"
                break
            delay = _retry_delay_seconds(attempt)
            print(
                f"  Chunk {chunk_index}/{total_chunks} returned HTTP {status_code} "
                f"on attempt {attempt}/{max_attempts}; retrying in {delay}s...",
                flush=True,
            )
            time.sleep(delay)
        except requests.exceptions.Timeout:
            if attempt >= max_attempts:
                last_problem = f"timeout after {timeout}s"
                break
            delay = _retry_delay_seconds(attempt)
            print(
                f"  Chunk {chunk_index}/{total_chunks} timed out on attempt "
                f"{attempt}/{max_attempts}; retrying in {delay}s...",
                flush=True,
            )
            time.sleep(delay)
        except requests.exceptions.RequestException as exc:
            if attempt >= max_attempts:
                last_problem = str(exc)
                break
            delay = _retry_delay_seconds(attempt)
            print(
                f"  Chunk {chunk_index}/{total_chunks} failed on attempt "
                f"{attempt}/{max_attempts}; retrying in {delay}s...",
                flush=True,
            )
            time.sleep(delay)

    if len(object_ids) == 1:
        raise RuntimeError(
            f"Chunk {chunk_index}/{total_chunks} could not recover object ID {object_ids[0]} "
            f"after {max_attempts} attempt(s): {last_problem or 'unknown error'}"
        )

    midpoint = max(1, len(object_ids) // 2)
    left_ids = object_ids[:midpoint]
    right_ids = object_ids[midpoint:]
    print(
        f"  Chunk {chunk_index}/{total_chunks} remained incomplete after {max_attempts} attempt(s) "
        f"({last_problem or 'unknown error'}); splitting into subchunks of "
        f"{len(left_ids)} and {len(right_ids)} IDs...",
        flush=True,
    )
    left_status, left_features = _download_arcgis_chunk_by_ids(
        query_url,
        object_ids=left_ids,
        timeout=timeout,
        chunk_index=chunk_index,
        total_chunks=total_chunks,
        max_attempts=max_attempts,
        object_id_field=object_id_field,
    )
    right_status, right_features = _download_arcgis_chunk_by_ids(
        query_url,
        object_ids=right_ids,
        timeout=timeout,
        chunk_index=chunk_index,
        total_chunks=total_chunks,
        max_attempts=max_attempts,
        object_id_field=object_id_field,
    )
    return max(left_status, right_status), [*left_features, *right_features]


def _download_arcgis_layer_by_object_ids(
    query_url: str,
    output_path: Path,
    *,
    timeout: int,
    chunk_size: int,
    max_workers: int,
    max_attempts: int,
) -> Path | None:
    print("  Requesting object ID inventory...", flush=True)
    inventory = _request_arcgis_json(
        query_url,
        params={
            "where": "1=1",
            "returnIdsOnly": "true",
            "f": "json",
        },
        timeout=timeout,
        label="object ID inventory",
    )

    object_id_field = inventory.get("objectIdFieldName")
    object_ids = _normalize_object_ids(inventory)
    if not object_ids:
        print(
            f"Skipping remote context-data download for {output_path.name}: no object IDs returned.",
            flush=True,
        )
        return None

    chunks = _chunk_values(object_ids, chunk_size)
    worker_count = max(1, min(max_workers, len(chunks)))
    print(
        f"  Received {len(object_ids)} object IDs. "
        f"Downloading in {len(chunks)} chunk(s) with {worker_count} worker(s)...",
        flush=True,
    )

    downloaded_rows: list[dict[str, Any]] = []
    downloaded_object_ids: set[int] = set()
    total_chunks = len(chunks)
    total_downloaded = 0

    try:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    _download_arcgis_chunk_by_ids,
                    query_url,
                    object_ids=chunk,
                    timeout=timeout,
                    chunk_index=index,
                    total_chunks=total_chunks,
                    max_attempts=max_attempts,
                    object_id_field=object_id_field,
                ): index
                for index, chunk in enumerate(chunks, start=1)
            }

            for future in as_completed(futures):
                chunk_index = futures[future]
                status_code, features = future.result()
                print(
                    f"  Chunk {chunk_index}/{total_chunks} completed with status {status_code}; "
                    f"received {len(features)} records",
                    flush=True,
                )
                rows = [feature.get("attributes", {}) for feature in features]
                downloaded_rows.extend(rows)
                total_downloaded += len(rows)
                if object_id_field:
                    for row in rows:
                        object_id = row.get(object_id_field)
                        if object_id is not None:
                            downloaded_object_ids.add(int(object_id))
                print(
                    f"  Total downloaded so far: {total_downloaded} records",
                    flush=True,
                )
    except Exception as exc:
        print(
            "Skipping remote context-data download for "
            f"{output_path.name} during object-ID chunk download "
            f"after {total_downloaded} records: {exc}",
            flush=True,
        )
        return None

    expected_count = len(object_ids)
    if object_id_field:
        missing_object_ids = set(object_ids) - downloaded_object_ids
        duplicate_count = total_downloaded - len(downloaded_object_ids)
        if missing_object_ids:
            print(
                "Incomplete remote context-data download for "
                f"{output_path.name}: expected {expected_count} unique object IDs, "
                f"downloaded {len(downloaded_object_ids)}. Missing {len(missing_object_ids)} IDs. "
                "Partial snapshot will not be saved.",
                flush=True,
            )
            return None
        if duplicate_count > 0:
            print(
                f"  Detected {duplicate_count} duplicate row(s) while downloading {output_path.name}; "
                "deduplicating on object ID before save.",
                flush=True,
            )
            frame = pd.DataFrame(downloaded_rows)
            frame = frame.drop_duplicates(subset=[object_id_field], keep="first")
        else:
            frame = pd.DataFrame(downloaded_rows)
    else:
        frame = pd.DataFrame(downloaded_rows)
        if len(frame) < expected_count:
            print(
                "Incomplete remote context-data download for "
                f"{output_path.name}: expected at least {expected_count} rows from object ID inventory, "
                f"downloaded {len(frame)} rows. Partial snapshot will not be saved.",
                flush=True,
            )
            return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    print(
        f"Saved raw context-data snapshot: {output_path} "
        f"({len(frame)} records validated against object ID inventory)",
        flush=True,
    )
    return output_path


def download_arcgis_layer(
    query_url: str | None,
    output_path: Path,
    timeout: int = 60,
    *,
    use_object_id_chunks: bool = False,
    chunk_size: int = 1000,
    max_workers: int = 4,
    max_attempts: int = 3,
) -> Path | None:
    """Download a paginated ArcGIS layer to CSV and cache it locally."""
    if output_path.exists():
        print(f"Using cached remote context-data file: {output_path}", flush=True)
        return output_path
    if not query_url:
        return None

    features: list[dict[str, Any]] = []
    result_offset = 0
    result_record_count = 2000
    page_number = 0

    print(
        "Starting remote context-data download: "
        f"{output_path.name} "
        f"(page_size={result_record_count}, timeout={timeout}s)",
        flush=True,
    )
    if use_object_id_chunks:
        print(
            f"  Download mode: object-ID chunks (chunk_size={chunk_size}, max_workers={max_workers}, "
            f"max_attempts={max_attempts})",
            flush=True,
        )
        return _download_arcgis_layer_by_object_ids(
            query_url,
            output_path,
            timeout=timeout,
            chunk_size=chunk_size,
            max_workers=max_workers,
            max_attempts=max_attempts,
        )

    try:
        while True:
            page_number += 1
            print(
                f"  Requesting page {page_number} (offset={result_offset})...",
                flush=True,
            )
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
            print(
                f"  Response status for page {page_number}: {response.status_code}",
                flush=True,
            )
            response.raise_for_status()
            payload = response.json()
            batch = payload.get("features", [])
            features.extend(feature.get("attributes", {}) for feature in batch)
            print(
                f"  Received {len(batch)} records on page {page_number}; "
                f"total downloaded: {len(features)}",
                flush=True,
            )
            if len(batch) < result_record_count:
                break
            result_offset += result_record_count
    except requests.exceptions.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else "unknown"
        print(
            "Skipping remote context-data download for "
            f"{output_path.name} after {page_number - 1 if page_number else 0} "
            f"successful page(s) and {len(features)} records: "
            f"HTTP status {status_code}: {exc}",
            flush=True,
        )
        return None
    except requests.exceptions.Timeout as exc:
        print(
            "Skipping remote context-data download for "
            f"{output_path.name} after {page_number - 1 if page_number else 0} "
            f"successful page(s) and {len(features)} records: "
            f"timeout after {timeout}s; no HTTP response status was received. {exc}",
            flush=True,
        )
        return None
    except Exception as exc:
        print(
            "Skipping remote context-data download for "
            f"{output_path.name} after {page_number - 1 if page_number else 0} "
            f"successful page(s) and {len(features)} records: {exc}",
            flush=True,
        )
        return None

    if not features:
        print(
            f"Skipping remote context-data download for {output_path.name}: no records returned.",
            flush=True,
        )
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(features).to_csv(output_path, index=False)
    print(
        f"Saved raw context-data snapshot: {output_path} ({len(features)} records)",
        flush=True,
    )
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
