"""Optional student housing loader and contextual integration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.features import normalize_string, normalize_zip
from src.data.violations import _standardize_columns

from .common import build_address_zip_key, build_address_zip_key_from_series


@dataclass(frozen=True)
class StudentHousingConfig:
    raw_dir: Path = Path("data/raw")
    candidates: tuple[str, ...] = (
        "student_housing.xlsx",
        "student_housing.csv",
        "uar_fall_2022.xlsx",
        "uar_fall_2023.xlsx",
    )
    default_summary_path: Path | None = Path("data/reference/student_housing_zip_2023.csv")
    clean_output_path: Path = Path("data/processed/student_housing_clean.csv")
    output_path: Path = Path("data/processed/student_housing_context_v1.csv")
    summary_output_path: Path = Path("data/processed/student_housing_summary_v1.csv")


def normalize_school_name(value: object) -> str:
    """Normalize institution names for conservative grouping."""
    return normalize_string(value)


def normalize_year_value(value: object) -> int | None:
    """Convert a year-like value to an integer when possible."""
    if value is None or pd.isna(value):
        return None
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    if len(digits) >= 4:
        return int(digits[:4])
    return None


def load_student_housing_data(config: StudentHousingConfig) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    """Load an optional team-provided or UAR-style student housing spreadsheet."""
    if config.clean_output_path.exists():
        cached = pd.read_csv(config.clean_output_path)
        diagnostics = {
            "student_housing_available": True,
            "source_path": str(config.clean_output_path),
            "grain": detect_student_housing_grain(cached),
        }
        return cached, diagnostics

    for candidate in config.candidates:
        path = config.raw_dir / candidate
        if not path.exists():
            continue
        df = pd.read_excel(path) if path.suffix.lower() == ".xlsx" else pd.read_csv(path)
        cleaned = clean_student_housing(df)
        diagnostics = {
            "student_housing_available": True,
            "source_path": str(path),
            "grain": detect_student_housing_grain(cleaned),
        }
        return cleaned, diagnostics

    if config.default_summary_path is not None and config.default_summary_path.exists():
        fallback = pd.read_csv(config.default_summary_path, low_memory=False)
        cleaned = clean_student_housing(fallback)
        diagnostics = {
            "student_housing_available": True,
            "source_path": str(config.default_summary_path),
            "grain": detect_student_housing_grain(cleaned),
            "student_housing_default_source": True,
        }
        return cleaned, diagnostics

    diagnostics = {
        "student_housing_available": False,
        "source_path": None,
        "grain": None,
    }
    print(
        "Student housing data unavailable: no local student_housing/uar CSV or XLSX file was found."
    )
    return None, diagnostics


def save_clean_student_housing(
    config: StudentHousingConfig,
) -> tuple[Path | None, dict[str, Any]]:
    """Load and persist the cleaned student housing table when an input file exists."""
    student_df, diagnostics = load_student_housing_data(config)
    if student_df is None:
        return None, diagnostics

    if not config.clean_output_path.exists():
        config.clean_output_path.parent.mkdir(parents=True, exist_ok=True)
        student_df.to_csv(config.clean_output_path, index=False)
        print(f"Saved clean table: {config.clean_output_path}")

    return config.clean_output_path, diagnostics


def clean_student_housing(df: pd.DataFrame) -> pd.DataFrame:
    """Clean an optional student housing file without assuming a fixed schema."""
    cleaned = df.copy()
    cleaned.columns = _standardize_columns(cleaned.columns.tolist())

    school_cols = [
        column
        for column in cleaned.columns
        if any(token in column for token in ["school", "institution", "university", "college"])
    ]
    for column in school_cols:
        cleaned[f"{column}_clean"] = cleaned[column].map(normalize_school_name).astype("string")

    for column in [column for column in cleaned.columns if "year" in column or "academic_year" in column]:
        cleaned[column] = cleaned[column].map(normalize_year_value)

    for column in [column for column in cleaned.columns if "zip" in column]:
        cleaned[column] = cleaned[column].map(normalize_zip).astype("string")

    address_col = next(
        (column for column in ["address", "street_address", "full_address", "local_address"] if column in cleaned.columns),
        None,
    )
    zip_col = next((column for column in ["zip", "zip_code", "postal_code"] if column in cleaned.columns), None)
    if address_col and zip_col:
        cleaned["address_zip_key"] = build_address_zip_key(cleaned, address_col, zip_col)
    return cleaned


def detect_student_housing_grain(df: pd.DataFrame) -> str:
    """Infer whether the student housing input is row-level or summary-level."""
    address_cols = {"address", "street_address", "full_address", "local_address"}
    geography_cols = {"zip", "zip_code", "postal_code", "neighborhood", "district"}
    if address_cols.intersection(df.columns):
        return "row_level"
    if geography_cols.intersection(df.columns):
        return "summary_level"
    return "unknown"


def build_student_housing_context(
    base_df: pd.DataFrame,
    student_df: pd.DataFrame | None,
    diagnostics: dict[str, Any],
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    """Create a contextual student-housing view when optional data is available."""
    if student_df is None:
        return None, diagnostics

    grain = diagnostics.get("grain")
    if (
        grain == "row_level"
        and ("address_zip_key" in base_df.columns or {"violation_st", "violation_zip"}.issubset(base_df.columns))
        and "address_zip_key" in student_df.columns
    ):
        context = base_df.copy()
        if "address_zip_key" not in context.columns:
            context["address_zip_key"] = build_address_zip_key_from_series(
                context["violation_st"],
                context["violation_zip"],
            )
        joined = context.merge(student_df, on="address_zip_key", how="left", suffixes=("", "_student"))
        diagnostics["student_context_join"] = "row_level_address_zip_approximate"
        diagnostics["student_output_type"] = "context"
        return joined, diagnostics

    zip_col = next((column for column in ["zip", "zip_code", "postal_code"] if column in student_df.columns), None)
    if zip_col and "violation_zip" in base_df.columns:
        zip_summary = (
            base_df.groupby("violation_zip")
            .agg(
                total_violations=("total_violations", "sum"),
                open_violations=("open_violations", "sum"),
                property_count=("property_key", "nunique"),
            )
            .reset_index()
            .rename(columns={"violation_zip": zip_col})
        )
        merged = zip_summary.merge(student_df, on=[zip_col], how="left")
        diagnostics["student_context_join"] = "zip_level_summary"
        diagnostics["student_output_type"] = "summary"
        return merged, diagnostics

    diagnostics["student_context_join"] = "unavailable"
    diagnostics["student_output_type"] = "skipped"
    return None, diagnostics


def save_student_housing_context(
    config: StudentHousingConfig,
    base_df: pd.DataFrame,
) -> tuple[Path | None, dict[str, Any]]:
    """Build and save student housing context when an optional file exists."""
    student_df, diagnostics = load_student_housing_data(config)
    context_df, diagnostics = build_student_housing_context(base_df, student_df, diagnostics)
    if context_df is None:
        return None, diagnostics

    output_path = (
        config.output_path
        if diagnostics.get("student_output_type") == "context"
        else config.summary_output_path
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    context_df.to_csv(output_path, index=False)
    print(f"Student housing output saved to: {output_path}")
    return output_path, diagnostics
