"""Phase 2 feature engineering for cleaned violations data.

This module keeps Phase 1 outputs intact and optionally enriches them with a
small raw-data supplement keyed by ``case_no``. That supplement is used only to
recover fields that materially improve Phase 2 interpretability, such as:

- a usable date column via ``status_dttm``
- an address-like location field via ``violation_st``

If those fields are unavailable, the feature pipeline falls back gracefully and
reports the limitation explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Iterable

import pandas as pd


DEFAULT_INPUT_PATH = Path("data/processed/violations_clean.csv")
DEFAULT_OUTPUT_PATH = Path("data/processed/violations_feature_table_v1.csv")
DEFAULT_RAW_PATH = Path("data/raw/violations.csv")

DATE_COLUMN_CANDIDATES = ["violdttm", "violation_date", "status_dttm", "date"]
STATUS_COLUMN_CANDIDATES = ["status"]
VIOLATION_TYPE_CANDIDATES = ["violationtype", "violationtype_descr", "description"]
VIOLATOR_COLUMN_CANDIDATES = ["violator_name"]

RAW_SUPPLEMENT_COLUMNS = [
    "case_no",
    "status_dttm",
    "violation_stno",
    "violation_street",
    "violation_suffix",
    "violation_zip",
]


@dataclass(frozen=True)
class Phase2FeatureConfig:
    input_path: Path = DEFAULT_INPUT_PATH
    output_path: Path = DEFAULT_OUTPUT_PATH
    raw_path: Path = DEFAULT_RAW_PATH


def _column_or_default(
    df: pd.DataFrame,
    column: str,
    index: pd.Index,
    normalizer,
) -> pd.Series:
    if column not in df.columns:
        return pd.Series("", index=index, dtype="string")
    return df[column].map(normalizer).astype("string")


def first_available_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    """Return the first column present in the DataFrame."""
    return next((column for column in candidates if column in df.columns), None)


def normalize_string(value: object) -> str:
    """Normalize generic strings for grouping and comparison."""
    if value is None or pd.isna(value):
        return ""

    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_address(value: object) -> str:
    """Normalize an address-like string without pretending to standardize USPS format."""
    text = normalize_string(value)
    replacements = {
        r"\bavenue\b": "ave",
        r"\bav\b": "ave",
        r"\bstreet\b": "st",
        r"\broad\b": "rd",
        r"\bboulevard\b": "blvd",
        r"\bdrive\b": "dr",
        r"\bplace\b": "pl",
        r"\bcourt\b": "ct",
        r"\bterrace\b": "ter",
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_zip(value: object) -> str:
    """Normalize ZIP values to a 5-character string when possible."""
    if value is None or pd.isna(value):
        return ""

    digits = re.sub(r"\D+", "", str(value))
    if not digits:
        return ""
    return digits[:5].zfill(5)


def clean_violator_name(series: pd.Series) -> pd.Series:
    """Create a normalized violator name series."""
    return series.map(normalize_string).astype("string")


def coerce_datetime_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    """Parse the first usable date column in place."""
    for column in candidates:
        if column not in df.columns:
            continue
        parsed = pd.to_datetime(df[column], errors="coerce")
        df[column] = parsed
        if parsed.notna().any():
            return column
    return None


def _build_violation_st_from_raw(raw_df: pd.DataFrame) -> pd.Series:
    """Construct a best-effort location string from raw address components."""
    pieces = []
    for column in ["violation_stno", "violation_street", "violation_suffix"]:
        if column in raw_df.columns:
            pieces.append(raw_df[column].fillna("").astype("string").str.strip())
        else:
            pieces.append(pd.Series("", index=raw_df.index, dtype="string"))

    combined = (pieces[0] + " " + pieces[1] + " " + pieces[2]).str.replace(
        r"\s+",
        " ",
        regex=True,
    )
    return combined.str.strip()


def load_phase2_source_data(config: Phase2FeatureConfig) -> pd.DataFrame:
    """Load the cleaned Phase 1 table and enrich it with small raw-data supplements."""
    if not config.input_path.exists():
        raise FileNotFoundError(
            f"Missing cleaned violations dataset at {config.input_path}. Run Phase 1 first."
        )

    df = pd.read_csv(config.input_path)
    if "violation_zip" in df.columns:
        df["violation_zip"] = df["violation_zip"].map(normalize_zip).astype("string")

    if not config.raw_path.exists() or "case_no" not in df.columns:
        return df

    raw_df = pd.read_csv(config.raw_path, usecols=RAW_SUPPLEMENT_COLUMNS)
    raw_df = raw_df.drop_duplicates(subset=["case_no"], keep="first").copy()
    raw_df.loc[:, "status_dttm"] = pd.to_datetime(raw_df["status_dttm"], errors="coerce")
    raw_df.loc[:, "violation_st"] = _build_violation_st_from_raw(raw_df)
    raw_df.loc[:, "violation_st"] = raw_df["violation_st"].map(normalize_address).astype("string")
    raw_df["violation_zip"] = raw_df["violation_zip"].map(normalize_zip).astype("string")
    supplement = raw_df.loc[:, ["case_no", "status_dttm", "violation_st", "violation_zip"]]

    merged = df.merge(supplement, on="case_no", how="left", suffixes=("", "_raw"))
    if "status_dttm" not in merged.columns and "status_dttm_raw" in merged.columns:
        merged = merged.rename(columns={"status_dttm_raw": "status_dttm"})
    if "violation_st" not in merged.columns and "violation_st_raw" in merged.columns:
        merged = merged.rename(columns={"violation_st_raw": "violation_st"})
    if "violation_zip_raw" in merged.columns:
        merged.loc[:, "violation_zip"] = (
            merged["violation_zip"].fillna(merged["violation_zip_raw"]).map(normalize_zip).astype("string")
        )
        merged = merged.drop(columns=["violation_zip_raw"])

    return merged


def generate_property_key_components(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Build property keys and track whether they came from location or fallback logic."""
    index = df.index
    property_key = pd.Series(pd.NA, index=index, dtype="string")
    property_key_source = pd.Series("row_index", index=index, dtype="string")

    address = _column_or_default(df, "violation_st", index, normalize_address)
    zip_code = _column_or_default(df, "violation_zip", index, normalize_zip)
    case_no = _column_or_default(df, "case_no", index, normalize_string)

    has_address = address.fillna("").str.len() > 0
    has_zip = zip_code.fillna("").str.len() > 0
    has_case_no = case_no.fillna("").str.len() > 0

    address_zip_mask = has_address & has_zip
    property_key.loc[address_zip_mask] = (
        address.loc[address_zip_mask].str.cat(zip_code.loc[address_zip_mask], sep="|")
    )
    property_key_source.loc[address_zip_mask] = "address_zip"

    address_only_mask = has_address & ~address_zip_mask
    property_key.loc[address_only_mask] = address.loc[address_only_mask]
    property_key_source.loc[address_only_mask] = "address_only"

    case_no_mask = property_key.isna() & has_case_no
    property_key.loc[case_no_mask] = case_no.loc[case_no_mask]
    property_key_source.loc[case_no_mask] = "case_no"

    row_index_mask = property_key.isna()
    property_key.loc[row_index_mask] = [f"row_{idx}" for idx in range(int(row_index_mask.sum()))]
    property_key_source.loc[row_index_mask] = "row_index"

    return (
        property_key.fillna("unknown_property").astype("string"),
        property_key_source.astype("string"),
    )


def generate_property_key(df: pd.DataFrame) -> pd.Series:
    """Build the most stable property-level key possible from available columns."""
    property_key, _ = generate_property_key_components(df)
    return property_key


def _status_mask(df: pd.DataFrame, statuses: set[str]) -> pd.Series:
    status_col = first_available_column(df, STATUS_COLUMN_CANDIDATES)
    if status_col is None:
        return pd.Series(False, index=df.index)
    return (
        df[status_col]
        .astype("string")
        .str.strip()
        .str.lower()
        .isin(statuses)
        .fillna(False)
    )


def _top_non_empty(series: pd.Series) -> str | pd.NA:
    cleaned = series.dropna().astype("string")
    cleaned = cleaned[cleaned.str.len() > 0]
    if cleaned.empty:
        return pd.NA
    return cleaned.mode().iloc[0]


def _recent_violation_count(
    group: pd.DataFrame,
    date_col: str,
    reference_date: pd.Timestamp,
) -> int:
    dates = group[date_col].dropna()
    if dates.empty:
        return 0
    window_start = reference_date - pd.Timedelta(days=365)
    return int(dates.ge(window_start).sum())


def landlord_features_available(df: pd.DataFrame) -> bool:
    """Return whether landlord/violator features are actually available."""
    if "violator_name_clean" not in df.columns:
        return False
    values = df["violator_name_clean"].astype("string").fillna("")
    return bool(values.str.len().gt(0).any())


def _aligned_group_values(
    grouped: pd.core.groupby.DataFrameGroupBy,
    property_keys: pd.Series,
    column: str,
    aggregator,
    *,
    fill_value=None,
    dtype=None,
) -> pd.Series:
    values = grouped[column].agg(aggregator).reindex(property_keys)
    if fill_value is not None:
        values = values.fillna(fill_value)
    if dtype is not None:
        values = values.astype(dtype)
    return values.reset_index(drop=True)


def _assign_grouped_column(
    feature_table: pd.DataFrame,
    grouped: pd.core.groupby.DataFrameGroupBy,
    source_column: str,
    target_column: str,
    aggregator,
    *,
    fill_value=None,
    dtype=None,
) -> None:
    feature_table.loc[:, target_column] = _aligned_group_values(
        grouped,
        feature_table["property_key"],
        source_column,
        aggregator,
        fill_value=fill_value,
        dtype=dtype,
    ).to_numpy()


def prepare_violations_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """Add normalized columns used by downstream Phase 2 steps."""
    prepared = df.copy()
    date_col = coerce_datetime_column(prepared, DATE_COLUMN_CANDIDATES)
    violator_col = first_available_column(prepared, VIOLATOR_COLUMN_CANDIDATES)

    property_key, property_key_source = generate_property_key_components(prepared)
    prepared.loc[:, "property_key"] = property_key
    prepared.loc[:, "property_key_source"] = property_key_source
    if violator_col is not None:
        prepared.loc[:, "violator_name_clean"] = clean_violator_name(prepared[violator_col])
    else:
        prepared.loc[:, "violator_name_clean"] = pd.Series(
            pd.NA,
            index=prepared.index,
            dtype="string",
        )

    if "violation_zip" in prepared.columns:
        prepared["violation_zip"] = prepared["violation_zip"].map(normalize_zip).astype("string")

    if date_col is not None:
        prepared.loc[:, "year"] = prepared[date_col].dt.year
        prepared.loc[:, "month"] = prepared[date_col].dt.month
        prepared.loc[:, "quarter"] = prepared[date_col].dt.quarter

    return prepared, date_col


def get_property_key_diagnostics(prepared_df: pd.DataFrame) -> dict[str, Any]:
    """Summarize how reliable the current property-key aggregation looks."""
    counts = prepared_df["property_key"].astype("string").value_counts(dropna=False)
    source_counts = (
        prepared_df["property_key_source"].astype("string").value_counts(dropna=False)
        if "property_key_source" in prepared_df.columns
        else pd.Series(dtype="int64")
    )
    top_counts = counts.head(5)

    return {
        "unique_property_keys": int(prepared_df["property_key"].nunique(dropna=True)),
        "singleton_property_key_pct": round(float(counts.eq(1).mean() * 100), 1),
        "address_based_key_pct": round(
            float(
                source_counts.reindex(["address_zip", "address_only"]).fillna(0).sum()
                / max(len(prepared_df), 1)
                * 100
            ),
            1,
        ),
        "case_no_fallback_pct": round(
            float(source_counts.get("case_no", 0) / max(len(prepared_df), 1) * 100),
            1,
        ),
        "top_property_key_counts": [
            f"{index}={value}" for index, value in top_counts.items()
        ],
        "property_key_source_counts": {
            str(index): int(value) for index, value in source_counts.items()
        },
    }


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the cleaned violations table to a property-level feature table."""
    prepared, date_col = prepare_violations_frame(df)
    grouped = prepared.groupby("property_key", dropna=False)

    feature_table = grouped.size().rename("total_violations").reset_index()
    feature_table.loc[:, "open_violations"] = grouped.apply(
        lambda group: int(_status_mask(group, {"open", "pending", "active"}).sum()),
        include_groups=False,
    ).to_numpy()
    feature_table.loc[:, "closed_violations"] = grouped.apply(
        lambda group: int(_status_mask(group, {"closed", "resolved"}).sum()),
        include_groups=False,
    ).to_numpy()

    if "property_key_source" in prepared.columns:
        _assign_grouped_column(
            feature_table,
            grouped,
            "property_key_source",
            "property_key_source",
            _top_non_empty,
            dtype="string",
        )

    violation_type_col = first_available_column(prepared, VIOLATION_TYPE_CANDIDATES)
    if violation_type_col is not None:
        _assign_grouped_column(
            feature_table,
            grouped,
            violation_type_col,
            "distinct_violation_types",
            lambda values: values.nunique(dropna=True),
            fill_value=0,
            dtype=int,
        )
        _assign_grouped_column(
            feature_table,
            grouped,
            violation_type_col,
            "primary_violation_type",
            _top_non_empty,
            dtype="string",
        )
    else:
        feature_table.loc[:, "distinct_violation_types"] = 0

    if "violation_st" in prepared.columns:
        _assign_grouped_column(
            feature_table,
            grouped,
            "violation_st",
            "violation_st",
            _top_non_empty,
            dtype="string",
        )
    if "violation_zip" in prepared.columns:
        _assign_grouped_column(
            feature_table,
            grouped,
            "violation_zip",
            "violation_zip",
            _top_non_empty,
            dtype="string",
        )

    if "violator_name_clean" in prepared.columns:
        _assign_grouped_column(
            feature_table,
            grouped,
            "violator_name_clean",
            "violator_name_clean",
            _top_non_empty,
            dtype="string",
        )
        _assign_grouped_column(
            feature_table,
            grouped,
            "violator_name_clean",
            "distinct_violators",
            lambda values: values.nunique(dropna=True),
            fill_value=0,
            dtype=int,
        )

    if date_col is not None:
        first_violation = grouped[date_col].min()
        last_violation = grouped[date_col].max()
        reference_date = prepared[date_col].max()

        first_violation = first_violation.reindex(feature_table["property_key"])
        last_violation = last_violation.reindex(feature_table["property_key"])
        feature_table.loc[:, "first_violation_date"] = first_violation.to_numpy()
        feature_table.loc[:, "last_violation_date"] = last_violation.to_numpy()
        span_days = (last_violation - first_violation).dt.days.fillna(0).clip(lower=0)
        feature_table.loc[:, "active_span_days"] = span_days.astype(int).to_numpy()
        active_years = (span_days / 365.25).replace(0, 1 / 365.25).clip(lower=1 / 365.25)
        rate = feature_table["total_violations"].to_numpy() / active_years.to_numpy()
        feature_table.loc[:, "violations_per_year"] = pd.Series(rate).round(2)
        if pd.notna(reference_date):
            recency = (reference_date - last_violation).dt.days.fillna(0).clip(lower=0)
            feature_table.loc[:, "days_since_last_violation"] = recency.astype(int).to_numpy()
            recent_counts = grouped.apply(
                lambda group: _recent_violation_count(group, date_col, reference_date),
                include_groups=False,
            )
            feature_table.loc[:, "recent_violation_count_365d"] = (
                recent_counts.reindex(feature_table["property_key"]).fillna(0).astype(int).to_numpy()
            )
            feature_table.loc[:, "reference_date"] = reference_date

    sort_columns = [
        column for column in ["total_violations", "last_violation_date"] if column in feature_table.columns
    ]
    if sort_columns:
        ascending = [False, False][: len(sort_columns)]
        feature_table = feature_table.sort_values(sort_columns, ascending=ascending, na_position="last")

    return feature_table.reset_index(drop=True)


def print_feature_summary(
    df: pd.DataFrame,
    property_key_diagnostics: dict[str, Any] | None = None,
    landlord_available: bool | None = None,
) -> None:
    """Print a short summary of the generated feature table."""
    summary_lines = [
        f"Feature table rows: {len(df)}",
        f"Feature table columns: {len(df.columns)}",
    ]
    if landlord_available is not None:
        summary_lines.append(f"Landlord/violator features available: {landlord_available}")
    if property_key_diagnostics is not None:
        summary_lines.extend(
            [
                f"Unique property keys: {property_key_diagnostics['unique_property_keys']}",
                "Singleton property keys (%): "
                f"{property_key_diagnostics['singleton_property_key_pct']:.1f}",
                "Address-based property keys (%): "
                f"{property_key_diagnostics['address_based_key_pct']:.1f}",
                "Case-no fallback keys (%): "
                f"{property_key_diagnostics['case_no_fallback_pct']:.1f}",
            ]
        )
    for line in summary_lines:
        print(line)
    if property_key_diagnostics is not None:
        print("Top property key counts:")
        for item in property_key_diagnostics["top_property_key_counts"]:
            print(f"  - {item}")
    if not df.empty:
        missingness = (df.isna().mean().sort_values(ascending=False) * 100).head(5)
        print("Top missingness (%):")
        for column, pct in missingness.items():
            print(f"  - {column}: {pct:.1f}%")


def run_phase2_feature_engineering(config: Phase2FeatureConfig) -> Path:
    """Load cleaned violations data, build Phase 2 features, and save them."""
    source_df = load_phase2_source_data(config)
    prepared, _ = prepare_violations_frame(source_df)
    diagnostics = get_property_key_diagnostics(prepared)
    landlord_available = landlord_features_available(prepared)
    feature_table = build_feature_table(source_df)

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_table.to_csv(config.output_path, index=False)
    print_feature_summary(
        feature_table,
        property_key_diagnostics=diagnostics,
        landlord_available=landlord_available,
    )
    if not landlord_available:
        print("Landlord features unavailable: no usable violator_name field exists in the current dataset.")
    print(f"Feature table saved to: {config.output_path}")
    return config.output_path


def main() -> None:
    run_phase2_feature_engineering(Phase2FeatureConfig())


if __name__ == "__main__":
    main()
