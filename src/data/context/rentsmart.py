from __future__ import annotations

import argparse
import gzip
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from urllib.request import Request, urlopen

import pandas as pd


QUERYDATA_URL = "https://wabi-us-gov-virginia-api.analysis.usgovcloudapi.net/public/reports/querydata"
RESOURCE_KEY = "7ae67bd7-1185-4984-b10e-8ce6599a08b0"
MODEL_ID = 1179503
PAGE_URL = "https://www.boston.gov/departments/analytics-team/rentsmart-boston"
DEFAULT_OUTPUT_PATH = Path("data/raw/rentsmart.csv")
DEFAULT_PAGE_SIZE = 50000

DEFAULT_EXPORT_COLUMNS: tuple[str, ...] = (
    "full_address",
    "address",
    "zip_code",
    "neighborhood",
    "owner",
    "property_type",
    "year_built",
    "year_remodeled",
    "sam_id",
    "parcel",
    "latitude",
    "longitude",
    "date",
    "type",
    "description",
)


@dataclass(frozen=True)
class RentSmartDownloadConfig:
    output_path: Path = DEFAULT_OUTPUT_PATH
    querydata_url: str = QUERYDATA_URL
    resource_key: str = RESOURCE_KEY
    model_id: int = MODEL_ID
    user_preferred_locale: str = "en-US"
    page_size: int = DEFAULT_PAGE_SIZE
    timeout_seconds: int = 120


class RentSmartDownloadError(RuntimeError):
    """Raised when the public RentSmart export cannot be retrieved."""


def _column_select(column: str, source_name: str = "r") -> dict[str, Any]:
    return {
        "Column": {
            "Expression": {"SourceRef": {"Source": source_name}},
            "Property": column,
        },
        "Name": f"rentsmart.{column}",
        "NativeReferenceName": column,
    }


def _column_order(column: str, direction: int, source_name: str = "r") -> dict[str, Any]:
    return {
        "Direction": direction,
        "Expression": {
            "Column": {
                "Expression": {"SourceRef": {"Source": source_name}},
                "Property": column,
            }
        },
    }


def build_query_payload(
    columns: Sequence[str],
    *,
    model_id: int = MODEL_ID,
    page_size: int = DEFAULT_PAGE_SIZE,
    restart_tokens: list[list[str]] | None = None,
    user_preferred_locale: str = "en-US",
) -> dict[str, Any]:
    projections = list(range(len(columns)))
    window: dict[str, Any] = {"Count": page_size}
    if restart_tokens:
        window["RestartTokens"] = restart_tokens

    query = {
        "Commands": [
            {
                "SemanticQueryDataShapeCommand": {
                    "Query": {
                        "Version": 2,
                        "From": [{"Name": "r", "Entity": "rentsmart", "Type": 0}],
                        "Select": [_column_select(column) for column in columns],
                        "OrderBy": [_column_order(column, 1) for column in columns],
                    },
                    "Binding": {
                        "Primary": {"Groupings": [{"Projections": projections}]},
                        "DataReduction": {
                            "DataVolume": 3,
                            "Primary": {"Window": window},
                        },
                        "Version": 1,
                    },
                    "ExecutionMetricsKind": 1,
                }
            }
        ]
    }
    return {
        "version": "1.0.0",
        "queries": [{"Query": query, "QueryId": "rentsmart-export"}],
        "cancelQueries": [],
        "modelId": model_id,
        "userPreferredLocale": user_preferred_locale,
    }


def _decode_raw_bytes(raw_bytes: bytes, content_encoding: str | None) -> dict[str, Any]:
    if content_encoding and "gzip" in content_encoding.lower():
        raw_bytes = gzip.decompress(raw_bytes)
    return json.loads(raw_bytes.decode("utf-8"))


def _coerce_datetime(raw_value: Any, fmt: str | None) -> Any:
    if raw_value is None or not isinstance(raw_value, (int, float)):
        return raw_value
    dt = datetime.fromtimestamp(raw_value / 1000, tz=timezone.utc)
    if fmt == "MM/dd/yyyy":
        return dt.date().isoformat()
    return dt.isoformat().replace("+00:00", "Z")


def _resolve_display_value(
    raw_value: Any,
    column_spec: dict[str, Any] | None,
    header_spec: dict[str, Any] | None,
    value_dicts: dict[str, list[Any]] | None,
) -> Any:
    if raw_value is None:
        return None

    dict_name = column_spec.get("DN") if column_spec else None
    if dict_name and value_dicts and isinstance(raw_value, int):
        lookup_values = value_dicts.get(dict_name)
        if lookup_values and 0 <= raw_value < len(lookup_values):
            raw_value = lookup_values[raw_value]

    if column_spec and column_spec.get("T") == 7:
        fmt = header_spec.get("Format") if header_spec else None
        return _coerce_datetime(raw_value, fmt)
    return raw_value


def unpack_powerbi_rows(
    headers: dict[str, dict[str, Any]],
    rows: list[dict[str, Any]],
    value_dicts: dict[str, list[Any]] | None = None,
) -> list[dict[str, Any]]:
    if not rows:
        return []

    first_row = rows[0]
    column_count = len(first_row.get("S", [])) or len(first_row.get("C", []))
    if column_count == 0:
        return []

    previous_row: list[Any] = [None] * column_count
    column_names = [f"column_{idx}" for idx in range(column_count)]
    column_specs: list[dict[str, Any] | None] = [None] * column_count
    header_specs: list[dict[str, Any] | None] = [None] * column_count
    unpacked_rows: list[dict[str, Any]] = []

    for row in rows:
        if "S" in row:
            specs = row["S"]
            for idx, spec in enumerate(specs):
                column_specs[idx] = spec
                header_spec = headers.get(spec.get("N", ""))
                header_specs[idx] = header_spec
                if header_spec:
                    column_names[idx] = header_spec.get("Name", column_names[idx])

        current_row: list[Any] = [None] * column_count
        copy_mask = int(row.get("R", 0) or 0)
        null_mask = int(row.get("Ø", 0) or 0)
        values = row.get("C", [])
        value_index = 0
        column_mask = 1

        for idx in range(column_count):
            if copy_mask & column_mask:
                current_row[idx] = previous_row[idx]
            elif null_mask & column_mask:
                current_row[idx] = None
            else:
                raw_value = values[value_index] if value_index < len(values) else None
                current_row[idx] = _resolve_display_value(
                    raw_value,
                    column_specs[idx],
                    header_specs[idx],
                    value_dicts,
                )
                value_index += 1

            previous_row[idx] = current_row[idx]
            column_mask <<= 1

        unpacked_rows.append(dict(zip(column_names, current_row)))

    return unpacked_rows


def parse_powerbi_response(response_json: dict[str, Any]) -> dict[str, Any]:
    data = response_json["results"][0]["result"]["data"]
    headers = {column["Value"]: column for column in data["descriptor"]["Select"]}
    dataset = data["dsr"]["DS"][0]
    primary_hierarchy = dataset["PH"][0]
    row_key = next(iter(primary_hierarchy))
    rows = unpack_powerbi_rows(headers, primary_hierarchy[row_key], dataset.get("ValueDicts"))

    row_count = None
    for event in data.get("metrics", {}).get("Events", []):
        metrics = event.get("Metrics")
        if isinstance(metrics, dict) and "RowCount" in metrics:
            row_count = metrics["RowCount"]
            break

    return {
        "rows": rows,
        "restart_tokens": dataset.get("RT", []),
        "is_complete": bool(dataset.get("IC")),
        "row_count": row_count,
    }


def _request_page(payload: dict[str, Any], config: RentSmartDownloadConfig) -> dict[str, Any]:
    request = Request(
        config.querydata_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "X-PowerBI-ResourceKey": config.resource_key,
            "Accept-Encoding": "gzip",
        },
        method="POST",
    )
    with urlopen(request, timeout=config.timeout_seconds) as response:
        return _decode_raw_bytes(response.read(), response.headers.get("Content-Encoding"))


def _build_export_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    renamed = {
        column: column.split(".", 1)[1]
        for column in frame.columns
        if column.startswith("rentsmart.")
    }
    if renamed:
        frame = frame.rename(columns=renamed)

    if "parcel" in frame.columns and "map_par_id" not in frame.columns:
        frame["map_par_id"] = frame["parcel"]
    if "date" in frame.columns and "violation_date" not in frame.columns:
        frame["violation_date"] = frame["date"]
    if "type" in frame.columns and "violation_type" not in frame.columns:
        frame["violation_type"] = frame["type"]
    if "description" in frame.columns and "violation_description" not in frame.columns:
        frame["violation_description"] = frame["description"]

    preferred_order = [
        "full_address",
        "address",
        "zip_code",
        "neighborhood",
        "owner",
        "property_type",
        "year_built",
        "year_remodeled",
        "sam_id",
        "parcel",
        "map_par_id",
        "latitude",
        "longitude",
        "date",
        "violation_date",
        "type",
        "violation_type",
        "description",
        "violation_description",
    ]
    existing = [column for column in preferred_order if column in frame.columns]
    remaining = [column for column in frame.columns if column not in existing]
    return frame.loc[:, existing + remaining]


def download_rentsmart_csv(config: RentSmartDownloadConfig = RentSmartDownloadConfig()) -> Path:
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    restart_tokens: list[list[str]] | None = None
    page_number = 1
    seen_restart_tokens: set[str] = set()

    print(f"Starting RentSmart download from official City of Boston page: {PAGE_URL}", flush=True)
    while True:
        payload = build_query_payload(
            DEFAULT_EXPORT_COLUMNS,
            model_id=config.model_id,
            page_size=config.page_size,
            restart_tokens=restart_tokens,
            user_preferred_locale=config.user_preferred_locale,
        )
        response_json = _request_page(payload, config)
        parsed = parse_powerbi_response(response_json)
        page_rows = parsed["rows"]
        all_rows.extend(page_rows)

        print(
            f"Downloaded RentSmart page {page_number}: "
            f"{len(page_rows)} decoded rows; total rows so far: {len(all_rows)}"
        , flush=True)
        restart_tokens = parsed["restart_tokens"] or None
        if not restart_tokens:
            break
        restart_key = json.dumps(restart_tokens, sort_keys=True)
        if restart_key in seen_restart_tokens:
            raise RentSmartDownloadError("Encountered a repeated RentSmart restart token; aborting to avoid an infinite loop.")
        seen_restart_tokens.add(restart_key)
        page_number += 1

    frame = _build_export_frame(all_rows).drop_duplicates().reset_index(drop=True)
    if frame.empty:
        raise RentSmartDownloadError("RentSmart export completed, but no rows were decoded.")

    frame.to_csv(output_path, index=False)
    print(f"Saved RentSmart export to: {output_path}", flush=True)
    print(f"Rows written: {len(frame)}", flush=True)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the public RentSmart Boston dataset.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    args = parser.parse_args()

    download_rentsmart_csv(
        RentSmartDownloadConfig(
            output_path=args.output,
            page_size=args.page_size,
        )
    )


if __name__ == "__main__":
    main()
