"""Improved property-level model with static/owner features and XGBoost."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False

from src.data.context.property import normalize_property_identifier
from src.data.features import DEFAULT_INPUT_PATH, DEFAULT_RAW_PATH, Phase2FeatureConfig, load_phase2_source_data, normalize_zip
from src.modeling.baseline_model import build_property_level_modeling_frame


# ── Feature groups ───────────────────────────────────────────────────────────

BEHAVIORAL_FEATURES = [
    "total_violations",
    "open_violations",
    "distinct_violation_types",
    "active_span_days",
    "violations_per_year",
    "days_since_last_violation",
    "recent_violation_count_365d",
    "history_high_risk_violations",
    "history_medium_risk_violations",
    "history_low_risk_violations",
    "history_open_share",
    "history_high_risk_share",
    "recent_high_risk_violation_count_365d",
    "recent_medium_risk_violation_count_365d",
    "recent_medium_or_high_risk_violation_count_365d",
    "recent_open_violation_count_365d",
    "compliance_rate",
]

ACS_FEATURES = [
    "acs_median_household_income",
    "acs_renter_occupied_share",
    "acs_vacancy_rate",
    "acs_young_adult_share",
]

ASSESSMENT_FEATURES = [
    "assessment_yr_built",
    "assessment_gross_area",
    "assessment_total_value",
    "assessment_res_units",
    "assessment_land_sf",
    "assessment_overall_cond_score",
    "assessment_is_residential",
    "assessment_is_large_apartment",
]

RENTSMART_FEATURES = [
    "rentsmart_record_count",
    "rentsmart_complaint_indicator",
    "rentsmart_records_365d",
    "rentsmart_complaint_count",
    "rentsmart_inspection_count",
    "rentsmart_violation_count",
    "rentsmart_issue_type_count",
]

SERVICE_REQUEST_FEATURES = [
    "service_request_count",
    "housing_related_service_request_count",
    "service_requests_90d",
    "service_requests_180d",
    "service_requests_365d",
    "prior_service_requests_365d",
    "service_request_growth_365_vs_prior",
    "heat_service_request_count",
    "pest_service_request_count",
    "sanitation_service_request_count",
    "building_code_service_request_count",
    "distinct_service_request_types",
    "recent_service_request_share_365d",
]

PERMIT_FEATURES = [
    "permit_count",
    "major_permit_count",
    "occupancy_permit_count",
    "permits_365d",
    "permits_730d",
    "major_permits_730d",
    "occupancy_permits_730d",
    "permit_intensity_730d",
    "major_permit_share",
    "occupancy_permit_share",
]

STUDENT_HOUSING_FEATURES = [
    "student_all_students",
    "student_units",
    "student_undergraduates",
    "student_graduates",
    "students_per_modeled_property",
    "student_units_per_modeled_property",
    "student_housing_zip_flag",
]

OWNER_FEATURE_COLS = [
    "owner_property_count",
    "owner_total_violations",
    "owner_avg_violations_per_property",
    "owner_high_risk_share",
    "owner_property_count_with_recent_violations",
    "owner_recent_violation_rate",
    "owner_open_violation_share",
]

LEAKAGE_SAFE_STATIC_FEATURES = ACS_FEATURES + ASSESSMENT_FEATURES + ["is_rental"]
MODEL_CONTEXT_FEATURES = (
    LEAKAGE_SAFE_STATIC_FEATURES
    + OWNER_FEATURE_COLS
    + RENTSMART_FEATURES
    + PERMIT_FEATURES
    + SERVICE_REQUEST_FEATURES
    + STUDENT_HOUSING_FEATURES
)

PERMIT_CONTEXT_COLUMNS = [
    "map_par_id",
    "pid",
    "address_zip_key",
    "permit_issue_date",
    "major_permit_flag",
    "occupancy_related_permit_flag",
    "permit_record_count",
]

SERVICE_REQUEST_CONTEXT_COLUMNS = [
    "address_zip_key",
    "service_request_open_date",
    "housing_related_request_flag",
    "service_request_record_count",
    "case_title",
    "subject",
    "reason",
    "type",
    "department",
]

RENTSMART_CONTEXT_COLUMNS = [
    "map_par_id",
    "address_zip_key",
    "address_only_key",
    "violation_date",
    "date",
    "type",
    "violation_type",
    "description",
    "violation_description",
]

STUDENT_HOUSING_CONTEXT_COLUMNS = [
    "zip",
    "zip_code",
    "postal_code",
    "undergraduates",
    "graduates",
    "all_students",
    "student_units",
    "report_year",
]


@dataclass(frozen=True)
class ImprovedModelConfig:
    input_path: Path = DEFAULT_INPUT_PATH
    raw_path: Path = DEFAULT_RAW_PATH
    property_risk_path: Path = Path("data/processed/property_risk_table_v1.csv")
    permits_context_path: Path = Path("data/processed/building_permits_clean.csv")
    rentsmart_context_path: Path = Path("data/processed/rentsmart_clean.csv")
    service_requests_path: Path = Path("data/processed/service_requests_311_historical.csv")
    student_housing_summary_path: Path = Path("data/processed/student_housing_summary_v1.csv")
    output_path: Path = Path("outputs/tables/improved_model_results.csv")
    feature_importance_path: Path = Path("outputs/tables/improved_model_feature_importance.csv")
    ablation_output_path: Path = Path("outputs/tables/improved_model_ablation.csv")
    grouped_cv_output_path: Path = Path("outputs/tables/improved_model_grouped_cv.csv")
    top_risk_output_path: Path = Path("outputs/tables/top_predicted_risk_properties.csv")
    calibration_output_path: Path = Path("outputs/tables/model_score_calibration.csv")
    prediction_window_days: int = 730
    cv_folds: int = 5
    random_state: int = 42


# ── Owner-level features ──────────────────────────────────────────────────────

def add_owner_level_features(risk_df: pd.DataFrame) -> pd.DataFrame:
    """Compute cross-property landlord signals and join back onto risk_df."""
    df = risk_df.copy()
    owner_col = "assessment_owner_clean"
    if owner_col not in df.columns or "total_violations" not in df.columns:
        for col in OWNER_FEATURE_COLS:
            df[col] = np.nan
        return df

    valid = df.dropna(subset=[owner_col])
    valid = valid.loc[valid[owner_col].astype(str).str.strip().ne("")]

    owner_agg = (
        valid.groupby(owner_col)
        .agg(
            owner_property_count=(owner_col, "count"),
            owner_total_violations=("total_violations", "sum"),
            _owner_total_open_violations=("open_violations", "sum")
            if "open_violations" in valid.columns
            else (owner_col, "count"),
        )
        .reset_index()
    )
    owner_agg["owner_avg_violations_per_property"] = (
        owner_agg["owner_total_violations"] / owner_agg["owner_property_count"]
    )

    if "history_high_risk_violations" in df.columns:
        owner_hr = (
            valid.groupby(owner_col)["history_high_risk_violations"]
            .sum()
            .reset_index()
            .rename(columns={"history_high_risk_violations": "_owner_hr_total"})
        )
        owner_agg = owner_agg.merge(owner_hr, on=owner_col, how="left")
        owner_agg["owner_high_risk_share"] = (
            owner_agg["_owner_hr_total"]
            / owner_agg["owner_total_violations"].replace(0, np.nan)
        ).fillna(0.0)
        owner_agg = owner_agg.drop(columns=["_owner_hr_total"])
    else:
        owner_agg["owner_high_risk_share"] = 0.0

    if "open_violations" in valid.columns:
        owner_agg["owner_open_violation_share"] = (
            owner_agg["_owner_total_open_violations"].astype(float)
            / owner_agg["owner_total_violations"].replace(0, np.nan).astype(float)
        ).fillna(0.0)
    else:
        owner_agg["owner_open_violation_share"] = 0.0
    owner_agg = owner_agg.drop(columns=["_owner_total_open_violations"])

    if "recent_violation_count_365d" in valid.columns:
        recent_owner = (
            valid.assign(_has_recent_violation=valid["recent_violation_count_365d"].fillna(0).gt(0).astype(int))
            .groupby(owner_col)["_has_recent_violation"]
            .sum()
            .reset_index()
            .rename(columns={"_has_recent_violation": "owner_property_count_with_recent_violations"})
        )
        owner_agg = owner_agg.merge(recent_owner, on=owner_col, how="left")
        owner_agg["owner_recent_violation_rate"] = (
            owner_agg["owner_property_count_with_recent_violations"].astype(float)
            / owner_agg["owner_property_count"].replace(0, np.nan).astype(float)
        ).fillna(0.0)
    else:
        owner_agg["owner_property_count_with_recent_violations"] = 0
        owner_agg["owner_recent_violation_rate"] = 0.0

    join_cols = [owner_col] + OWNER_FEATURE_COLS
    df = df.merge(owner_agg[join_cols], on=owner_col, how="left")
    return df


def _read_optional_context_csv(path: Path, columns: list[str]) -> pd.DataFrame | None:
    if not path.exists():
        return None
    available = pd.read_csv(path, nrows=0).columns.tolist()
    usecols = [column for column in columns if column in available]
    if not usecols:
        return None
    return pd.read_csv(path, usecols=usecols, low_memory=False)


def _normalize_join_key(series: pd.Series, *, identifier: bool) -> pd.Series:
    if identifier:
        return series.map(normalize_property_identifier).astype("string").replace("", pd.NA)
    return series.astype("string").str.strip().replace("", pd.NA)


def _combined_lower_text(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    available = [column for column in columns if column in df.columns]
    if not available:
        return pd.Series("", index=df.index, dtype="string")
    return df[available].fillna("").astype("string").agg(" ".join, axis=1).str.lower()


def _merge_context_by_priority(
    property_lookup: pd.DataFrame,
    aggregated_frames: list[tuple[str, str, bool, pd.DataFrame]],
    feature_cols: list[str],
) -> pd.DataFrame:
    output = pd.DataFrame({"property_key": property_lookup["property_key"]})
    for column in feature_cols:
        output[column] = np.nan

    for left_col, right_col, identifier, aggregated in aggregated_frames:
        if left_col not in property_lookup.columns or right_col not in aggregated.columns or aggregated.empty:
            continue

        left = property_lookup[["property_key", left_col]].copy()
        left["_join_key"] = _normalize_join_key(left[left_col], identifier=identifier)
        right = aggregated[[right_col] + feature_cols].copy()
        right["_join_key"] = _normalize_join_key(right[right_col], identifier=identifier)
        right = right.dropna(subset=["_join_key"]).drop_duplicates("_join_key")
        matched = left.merge(right[["_join_key"] + feature_cols], on="_join_key", how="left")

        for column in feature_cols:
            fill_mask = output[column].isna() & matched[column].notna()
            output.loc[fill_mask, column] = matched.loc[fill_mask, column].to_numpy()

    output[feature_cols] = output[feature_cols].fillna(0)
    return output


def _aggregate_permits_before_cutoff(
    permits_df: pd.DataFrame,
    *,
    join_key: str,
    cutoff_date: pd.Timestamp,
) -> pd.DataFrame:
    if join_key not in permits_df.columns or "permit_issue_date" not in permits_df.columns:
        return pd.DataFrame(columns=[join_key] + PERMIT_FEATURES)

    working = permits_df.dropna(subset=[join_key]).copy()
    working["permit_issue_date"] = pd.to_datetime(working["permit_issue_date"], errors="coerce")
    working = working.loc[working["permit_issue_date"].notna() & working["permit_issue_date"].le(cutoff_date)].copy()
    if working.empty:
        return pd.DataFrame(columns=[join_key] + PERMIT_FEATURES)

    if "permit_record_count" not in working.columns:
        working["permit_record_count"] = 1
    if "major_permit_flag" not in working.columns:
        working["major_permit_flag"] = 0
    if "occupancy_related_permit_flag" not in working.columns:
        working["occupancy_related_permit_flag"] = 0
    working["permits_365d"] = (
        working["permit_issue_date"]
        .ge(cutoff_date - pd.Timedelta(days=365))
        .fillna(False)
        .astype(int)
    )
    working["permits_730d"] = (
        working["permit_issue_date"]
        .ge(cutoff_date - pd.Timedelta(days=730))
        .fillna(False)
        .astype(int)
    )
    working["major_permits_730d"] = working["major_permit_flag"].astype(float) * working["permits_730d"]
    working["occupancy_permits_730d"] = (
        working["occupancy_related_permit_flag"].astype(float) * working["permits_730d"]
    )

    aggregated = (
        working.groupby(join_key)
        .agg(
            permit_count=("permit_record_count", "sum"),
            major_permit_count=("major_permit_flag", "sum"),
            occupancy_permit_count=("occupancy_related_permit_flag", "sum"),
            permits_365d=("permits_365d", "sum"),
            permits_730d=("permits_730d", "sum"),
            major_permits_730d=("major_permits_730d", "sum"),
            occupancy_permits_730d=("occupancy_permits_730d", "sum"),
        )
        .reset_index()
    )
    aggregated["permit_intensity_730d"] = aggregated["permits_730d"].astype(float) / 2.0
    aggregated["major_permit_share"] = (
        aggregated["major_permit_count"].astype(float)
        / aggregated["permit_count"].replace(0, np.nan).astype(float)
    ).fillna(0.0)
    aggregated["occupancy_permit_share"] = (
        aggregated["occupancy_permit_count"].astype(float)
        / aggregated["permit_count"].replace(0, np.nan).astype(float)
    ).fillna(0.0)
    return aggregated


def _aggregate_rentsmart_before_cutoff(
    rentsmart_df: pd.DataFrame,
    *,
    join_key: str,
    cutoff_date: pd.Timestamp,
) -> pd.DataFrame:
    if join_key not in rentsmart_df.columns:
        return pd.DataFrame(columns=[join_key] + RENTSMART_FEATURES)

    date = pd.Series(pd.NaT, index=rentsmart_df.index, dtype="datetime64[ns]")
    for column in ["violation_date", "date"]:
        if column in rentsmart_df.columns:
            date = date.fillna(pd.to_datetime(rentsmart_df[column], errors="coerce"))

    working = rentsmart_df.loc[date.notna() & date.le(cutoff_date)].dropna(subset=[join_key]).copy()
    if working.empty:
        return pd.DataFrame(columns=[join_key] + RENTSMART_FEATURES)

    text = _combined_lower_text(
        working,
        ["type", "violation_type", "description", "violation_description"],
    )
    working["rentsmart_records_365d"] = (
        date.loc[working.index].ge(cutoff_date - pd.Timedelta(days=365)).fillna(False).astype(int)
    )
    working["rentsmart_complaint_count"] = text.str.contains("complaint", na=False).astype(int)
    working["rentsmart_inspection_count"] = text.str.contains("inspection|inspect", regex=True, na=False).astype(int)
    working["rentsmart_violation_count"] = (
        text.str.contains("violation|unsafe|code|condition|sanitary|heat|pest|rodent", regex=True, na=False)
        | working[[column for column in ["violation_type", "violation_description"] if column in working.columns]]
        .notna()
        .any(axis=1)
        if any(column in working.columns for column in ["violation_type", "violation_description"])
        else text.str.contains("violation|unsafe|code|condition|sanitary|heat|pest|rodent", regex=True, na=False)
    ).astype(int)
    complaint_signal = (
        working["rentsmart_complaint_count"].gt(0)
        | working["rentsmart_inspection_count"].gt(0)
        | working["rentsmart_violation_count"].gt(0)
    ).astype(int)
    type_cols = [column for column in ["type", "violation_type"] if column in working.columns]
    if type_cols:
        working["_rentsmart_issue_key"] = working[type_cols].fillna("").astype("string").agg("|".join, axis=1)
        working["_rentsmart_issue_key"] = working["_rentsmart_issue_key"].replace("|", pd.NA).replace("", pd.NA)
    else:
        working["_rentsmart_issue_key"] = pd.NA
    working["rentsmart_record_count"] = 1
    working["rentsmart_complaint_indicator"] = complaint_signal

    return (
        working.groupby(join_key)
        .agg(
            rentsmart_record_count=("rentsmart_record_count", "sum"),
            rentsmart_complaint_indicator=("rentsmart_complaint_indicator", "max"),
            rentsmart_records_365d=("rentsmart_records_365d", "sum"),
            rentsmart_complaint_count=("rentsmart_complaint_count", "sum"),
            rentsmart_inspection_count=("rentsmart_inspection_count", "sum"),
            rentsmart_violation_count=("rentsmart_violation_count", "sum"),
            rentsmart_issue_type_count=("_rentsmart_issue_key", "nunique"),
        )
        .reset_index()
    )


def _aggregate_311_before_cutoff(
    sr_df: pd.DataFrame,
    *,
    cutoff_date: pd.Timestamp,
) -> pd.DataFrame:
    """Aggregate 311 service requests per property up to the training cutoff."""
    if "address_zip_key" not in sr_df.columns or "service_request_open_date" not in sr_df.columns:
        return pd.DataFrame(columns=["address_zip_key"] + SERVICE_REQUEST_FEATURES)

    working = sr_df.dropna(subset=["address_zip_key"]).copy()
    working["service_request_open_date"] = pd.to_datetime(working["service_request_open_date"], errors="coerce")
    working = working.loc[working["service_request_open_date"].notna() & working["service_request_open_date"].le(cutoff_date)]
    if working.empty:
        return pd.DataFrame(columns=["address_zip_key"] + SERVICE_REQUEST_FEATURES)

    if "service_request_record_count" not in working.columns:
        working = working.copy()
        working["service_request_record_count"] = 1
    if "housing_related_request_flag" not in working.columns:
        working = working.copy()
        working["housing_related_request_flag"] = 0

    working = working.copy()
    text = _combined_lower_text(
        working,
        ["case_title", "subject", "reason", "type", "department"],
    )
    working["service_requests_90d_flag"] = (
        working["service_request_open_date"].ge(cutoff_date - pd.Timedelta(days=90)).astype(int)
    )
    working["service_requests_180d_flag"] = (
        working["service_request_open_date"].ge(cutoff_date - pd.Timedelta(days=180)).astype(int)
    )
    working["service_requests_365d_flag"] = (
        working["service_request_open_date"].ge(cutoff_date - pd.Timedelta(days=365)).astype(int)
    )
    working["prior_service_requests_365d_flag"] = (
        working["service_request_open_date"].ge(cutoff_date - pd.Timedelta(days=730))
        & working["service_request_open_date"].lt(cutoff_date - pd.Timedelta(days=365))
    ).astype(int)
    working["heat_service_request_flag"] = (
        text.str.contains("heat|heating|hot water", regex=True, na=False).astype(int)
    )
    working["pest_service_request_flag"] = (
        text.str.contains("pest|rodent|bed bug|cockroach|mice|mouse|rat", regex=True, na=False).astype(int)
    )
    working["sanitation_service_request_flag"] = (
        text.str.contains("sanitation|unsanitary|trash|garbage|mold|lead", regex=True, na=False).astype(int)
    )
    working["building_code_service_request_flag"] = (
        text.str.contains(
            "building|code enforcement|unsafe|illegal|occupancy|poor condition|housing",
            regex=True,
            na=False,
        ).astype(int)
    )
    if "type" in working.columns:
        working["_service_request_type_key"] = working["type"].astype("string").str.strip().replace("", pd.NA)
    else:
        working["_service_request_type_key"] = pd.NA

    aggregated = (
        working.groupby("address_zip_key")
        .agg(
            service_request_count=("service_request_record_count", "sum"),
            housing_related_service_request_count=("housing_related_request_flag", "sum"),
            service_requests_90d=("service_requests_90d_flag", "sum"),
            service_requests_180d=("service_requests_180d_flag", "sum"),
            service_requests_365d=("service_requests_365d_flag", "sum"),
            prior_service_requests_365d=("prior_service_requests_365d_flag", "sum"),
            heat_service_request_count=("heat_service_request_flag", "sum"),
            pest_service_request_count=("pest_service_request_flag", "sum"),
            sanitation_service_request_count=("sanitation_service_request_flag", "sum"),
            building_code_service_request_count=("building_code_service_request_flag", "sum"),
            distinct_service_request_types=("_service_request_type_key", "nunique"),
        )
        .reset_index()
    )
    aggregated["recent_service_request_share_365d"] = (
        aggregated["service_requests_365d"].astype(float)
        / aggregated["service_request_count"].replace(0, np.nan).astype(float)
    ).fillna(0.0)
    aggregated["service_request_growth_365_vs_prior"] = (
        (
            aggregated["service_requests_365d"].astype(float)
            - aggregated["prior_service_requests_365d"].astype(float)
        )
        / aggregated["prior_service_requests_365d"].replace(0, 1).astype(float)
    )
    return aggregated


def add_cutoff_safe_temporal_context_features(
    modeling_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    *,
    permits_df: pd.DataFrame | None = None,
    rentsmart_df: pd.DataFrame | None = None,
    service_requests_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add permit/RentSmart/311 features after filtering source records to the model cutoff."""
    if "property_key" not in risk_df.columns or "training_cutoff_date" not in modeling_df.columns:
        return modeling_df

    cutoff_date = pd.to_datetime(modeling_df["training_cutoff_date"].iloc[0], errors="coerce")
    if pd.isna(cutoff_date):
        return modeling_df

    lookup_cols = [
        "property_key",
        "assessment_map_par_id",
        "assessment_pid",
        "sam_map_par_id",
        "address_zip_key",
        "address_only_key",
    ]
    risk_lookup_cols = [column for column in lookup_cols if column in risk_df.columns]
    property_lookup = (
        modeling_df[["property_key"]]
        .merge(risk_df[risk_lookup_cols].drop_duplicates("property_key"), on="property_key", how="left")
        .drop_duplicates("property_key")
        .reset_index(drop=True)
    )
    if "address_zip_key" not in property_lookup.columns:
        property_lookup["address_zip_key"] = property_lookup["property_key"]
    if "address_only_key" not in property_lookup.columns:
        property_lookup["address_only_key"] = property_lookup["property_key"].astype("string").str.split("|").str[0]

    context_frames: list[pd.DataFrame] = []

    if permits_df is not None:
        permit_pairs = [
            ("assessment_map_par_id", "map_par_id", True),
            ("assessment_pid", "pid", True),
            ("sam_map_par_id", "map_par_id", True),
            ("address_zip_key", "address_zip_key", False),
        ]
        permit_aggregated = [
            (left, right, identifier, _aggregate_permits_before_cutoff(permits_df, join_key=right, cutoff_date=cutoff_date))
            for left, right, identifier in permit_pairs
            if right in permits_df.columns
        ]
        context_frames.append(_merge_context_by_priority(property_lookup, permit_aggregated, PERMIT_FEATURES))

    if rentsmart_df is not None:
        rentsmart_pairs = [
            ("assessment_map_par_id", "map_par_id", True),
            ("sam_map_par_id", "map_par_id", True),
            ("address_zip_key", "address_zip_key", False),
            ("address_only_key", "address_only_key", False),
        ]
        rentsmart_aggregated = [
            (left, right, identifier, _aggregate_rentsmart_before_cutoff(rentsmart_df, join_key=right, cutoff_date=cutoff_date))
            for left, right, identifier in rentsmart_pairs
            if right in rentsmart_df.columns
        ]
        context_frames.append(_merge_context_by_priority(property_lookup, rentsmart_aggregated, RENTSMART_FEATURES))

    if service_requests_df is not None:
        sr_aggregated = _aggregate_311_before_cutoff(service_requests_df, cutoff_date=cutoff_date)
        if not sr_aggregated.empty and "address_zip_key" in sr_aggregated.columns:
            sr_frames = [("address_zip_key", "address_zip_key", False, sr_aggregated)]
            context_frames.append(_merge_context_by_priority(property_lookup, sr_frames, SERVICE_REQUEST_FEATURES))

    for context in context_frames:
        if len(context.columns) > 1:
            modeling_df = modeling_df.merge(context, on="property_key", how="left")

    for column in PERMIT_FEATURES + RENTSMART_FEATURES + SERVICE_REQUEST_FEATURES:
        if column in modeling_df.columns:
            modeling_df[column] = modeling_df[column].fillna(0)
    return modeling_df


def add_student_housing_zip_features(
    modeling_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    student_housing_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Join ZIP-level student housing context without using violation-derived summary fields."""
    if student_housing_df is None or student_housing_df.empty:
        return modeling_df

    zip_col = next(
        (column for column in ["zip", "zip_code", "postal_code"] if column in student_housing_df.columns),
        None,
    )
    if zip_col is None or "property_key" not in risk_df.columns:
        return modeling_df

    student = student_housing_df.copy()
    student["_student_zip"] = student[zip_col].map(normalize_zip).astype("string")
    student = student.dropna(subset=["_student_zip"])
    if student.empty:
        return modeling_df

    numeric_cols = [
        column
        for column in ["all_students", "student_units", "undergraduates", "graduates"]
        if column in student.columns
    ]
    for column in numeric_cols:
        student[column] = pd.to_numeric(student[column], errors="coerce")

    aggregations: dict[str, tuple[str, str]] = {}
    if "all_students" in student.columns:
        aggregations["student_all_students"] = ("all_students", "sum")
    if "student_units" in student.columns:
        aggregations["student_units"] = ("student_units", "sum")
    if "undergraduates" in student.columns:
        aggregations["student_undergraduates"] = ("undergraduates", "sum")
    if "graduates" in student.columns:
        aggregations["student_graduates"] = ("graduates", "sum")
    if not aggregations:
        return modeling_df

    student_zip = student.groupby("_student_zip").agg(**aggregations).reset_index()

    risk_zip_cols = [column for column in ["property_key", "violation_zip", "address_zip_key"] if column in risk_df.columns]
    lookup = (
        modeling_df[["property_key"]]
        .merge(risk_df[risk_zip_cols].drop_duplicates("property_key"), on="property_key", how="left")
        .drop_duplicates("property_key")
    )
    if "violation_zip" in lookup.columns:
        lookup["_student_zip"] = lookup["violation_zip"].map(normalize_zip).astype("string")
    elif "address_zip_key" in lookup.columns:
        lookup["_student_zip"] = (
            lookup["address_zip_key"].astype("string").str.split("|").str[-1].map(normalize_zip).astype("string")
        )
    elif "property_key" in lookup.columns:
        lookup["_student_zip"] = (
            lookup["property_key"].astype("string").str.split("|").str[-1].map(normalize_zip).astype("string")
        )
    else:
        return modeling_df

    modeled_property_counts = (
        lookup.dropna(subset=["_student_zip"])
        .groupby("_student_zip")["property_key"]
        .nunique()
        .rename("_modeled_property_count_zip")
        .reset_index()
    )
    lookup = lookup.merge(modeled_property_counts, on="_student_zip", how="left")
    lookup = lookup.merge(student_zip, on="_student_zip", how="left")

    for column in STUDENT_HOUSING_FEATURES:
        if column not in lookup.columns:
            lookup[column] = 0.0
    lookup["student_housing_zip_flag"] = lookup["student_all_students"].notna().astype(int)
    for column in ["student_all_students", "student_units", "student_undergraduates", "student_graduates"]:
        if column in lookup.columns:
            lookup[column] = lookup[column].fillna(0.0)

    denominator = lookup["_modeled_property_count_zip"].replace(0, np.nan).astype(float)
    lookup["students_per_modeled_property"] = (
        lookup["student_all_students"].astype(float) / denominator
    ).fillna(0.0)
    lookup["student_units_per_modeled_property"] = (
        lookup["student_units"].astype(float) / denominator
    ).fillna(0.0)

    return modeling_df.merge(
        lookup[["property_key"] + STUDENT_HOUSING_FEATURES],
        on="property_key",
        how="left",
    ).assign(
        **{column: lambda d, column=column: d[column].fillna(0.0) for column in STUDENT_HOUSING_FEATURES}
    )


# ── Modeling frame ────────────────────────────────────────────────────────────

def build_improved_modeling_frame(
    source_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    *,
    prediction_window_days: int = 365,
    permits_df: pd.DataFrame | None = None,
    rentsmart_df: pd.DataFrame | None = None,
    service_requests_df: pd.DataFrame | None = None,
    student_housing_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Temporal + static + owner features; also adds broader any-violation target."""
    modeling_df = build_property_level_modeling_frame(
        source_df, prediction_window_days=prediction_window_days
    )
    modeling_df["will_receive_any_violation_next_period"] = (
        modeling_df["future_violation_count"].gt(0).astype(int)
    )

    # compliance_rate: share of historical violations that were closed (landlord responsiveness)
    if "closed_violations" in modeling_df.columns and "total_violations" in modeling_df.columns:
        modeling_df["compliance_rate"] = (
            modeling_df["closed_violations"].astype(float)
            / modeling_df["total_violations"].replace(0, np.nan)
        ).fillna(0.0)
    else:
        modeling_df["compliance_rate"] = 0.0

    if {"property_key", "assessment_owner_clean"}.issubset(risk_df.columns):
        owner_lookup = risk_df[["property_key", "assessment_owner_clean"]].drop_duplicates("property_key")
        modeling_df = modeling_df.merge(owner_lookup, on="property_key", how="left")
    modeling_df = add_owner_level_features(modeling_df)

    # is_rental + derived assessment features — all static, no temporal leakage
    assessment_raw_cols = [
        "assessment_own_occ", "assessment_lu", "assessment_bldg_type",
        "assessment_overall_cond", "assessment_res_units", "assessment_living_area",
    ]
    # also pull land_sf from parcel if available
    extra_raw = [c for c in ["parcel_shape_area"] if c in risk_df.columns]
    raw_join_cols = [c for c in assessment_raw_cols + extra_raw if c in risk_df.columns]
    if raw_join_cols and "property_key" in risk_df.columns:
        raw_lookup = risk_df[["property_key"] + raw_join_cols].drop_duplicates("property_key")
        modeling_df = modeling_df.merge(raw_lookup, on="property_key", how="left")

    # is_rental
    if "assessment_own_occ" in modeling_df.columns:
        modeling_df["is_rental"] = (
            modeling_df["assessment_own_occ"].astype("string").str.strip().str.upper().eq("N").fillna(False)
        ).astype(int)
        modeling_df = modeling_df.drop(columns=["assessment_own_occ"])
    else:
        modeling_df["is_rental"] = 0

    # is_residential: land use R1/R2/R3/RL/A = residential
    if "assessment_lu" in modeling_df.columns:
        res_pattern = r"^(R[0-9]|RL|A$|CD$)"
        modeling_df["assessment_is_residential"] = (
            modeling_df["assessment_lu"].astype("string").str.strip().str.upper()
            .str.match(res_pattern).fillna(False).astype(int)
        )
        modeling_df = modeling_df.drop(columns=["assessment_lu"])
    else:
        modeling_df["assessment_is_residential"] = 0

    # is_large_apartment: bldg_type contains "APT" (apartment buildings)
    if "assessment_bldg_type" in modeling_df.columns:
        modeling_df["assessment_is_large_apartment"] = (
            modeling_df["assessment_bldg_type"].astype("string").str.upper()
            .str.contains("APT", na=False).astype(int)
        )
        modeling_df = modeling_df.drop(columns=["assessment_bldg_type"])
    else:
        modeling_df["assessment_is_large_apartment"] = 0

    # overall_cond → ordinal score (A=4, B=3, C=2, D=1, else 0)
    if "assessment_overall_cond" in modeling_df.columns:
        cond_map = {"A": 4, "B": 3, "C": 2, "D": 1}
        modeling_df["assessment_overall_cond_score"] = (
            modeling_df["assessment_overall_cond"].astype("string")
            .str.strip().str.upper().str[0]
            .map(cond_map).fillna(0).astype(float)
        )
        modeling_df = modeling_df.drop(columns=["assessment_overall_cond"])
    else:
        modeling_df["assessment_overall_cond_score"] = 0.0

    # res_units — impute missing with 0 (no unit info = likely not multi-unit)
    if "assessment_res_units" in modeling_df.columns:
        modeling_df["assessment_res_units"] = modeling_df["assessment_res_units"].fillna(0.0)
    else:
        modeling_df["assessment_res_units"] = 0.0

    # land_sf: use parcel_shape_area as proxy for lot size
    if "parcel_shape_area" in modeling_df.columns:
        modeling_df = modeling_df.rename(columns={"parcel_shape_area": "assessment_land_sf"})
        modeling_df["assessment_land_sf"] = modeling_df["assessment_land_sf"].fillna(0.0)
    elif "assessment_living_area" in modeling_df.columns:
        modeling_df = modeling_df.rename(columns={"assessment_living_area": "assessment_land_sf"})
        modeling_df["assessment_land_sf"] = modeling_df["assessment_land_sf"].fillna(0.0)
    else:
        modeling_df["assessment_land_sf"] = 0.0

    # Drop any remaining raw assessment columns that were renamed/derived
    for col in ["assessment_living_area"]:
        if col in modeling_df.columns:
            modeling_df = modeling_df.drop(columns=[col])

    # Pull remaining static features (ACS, assessment_yr_built, etc.)
    static_source_cols = [
        c for c in LEAKAGE_SAFE_STATIC_FEATURES
        if c in risk_df.columns and c not in modeling_df.columns
    ]
    if static_source_cols and "property_key" in risk_df.columns:
        modeling_df = modeling_df.merge(
            risk_df[["property_key"] + static_source_cols].drop_duplicates("property_key"),
            on="property_key",
            how="left",
        )

    modeling_df = add_cutoff_safe_temporal_context_features(
        modeling_df,
        risk_df,
        permits_df=permits_df,
        rentsmart_df=rentsmart_df,
        service_requests_df=service_requests_df,
    )
    modeling_df = add_student_housing_zip_features(modeling_df, risk_df, student_housing_df)

    return modeling_df


# ── Model factory ─────────────────────────────────────────────────────────────

def _make_pipeline(classifier, feature_cols: list[str]) -> Pipeline:
    from sklearn.compose import ColumnTransformer
    preprocessor = ColumnTransformer(
        [("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), feature_cols)],
        remainder="drop",
    )
    return Pipeline([("pre", preprocessor), ("clf", classifier)])


def _classifiers(n_positive: int, n_negative: int, random_state: int) -> list[tuple[str, object]]:
    scale = max(1, n_negative // max(1, n_positive))
    models: list[tuple[str, object]] = [
        ("logistic_regression", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=random_state)),
        # A shallower forest generalizes better than the sklearn defaults on this
        # highly imbalanced target and improves ranking metrics in temporal CV.
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=400,
                max_depth=6,
                min_samples_leaf=20,
                class_weight="balanced_subsample",
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
    ]
    if _XGBOOST_AVAILABLE:
        models.append((
            "xgboost",
            XGBClassifier(
                n_estimators=400,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=10,
                scale_pos_weight=scale,
                eval_metric="logloss",
                random_state=random_state,
                verbosity=0,
            ),
        ))
    return models


# ── CV evaluation ─────────────────────────────────────────────────────────────

_PRECISION_AT_K = [10, 25, 50, 100]


def _find_optimal_threshold(y_true: pd.Series, probas: np.ndarray) -> float:
    """Sweep thresholds on the training split and return the one maximizing F1."""
    best_f1, best_thresh = -1.0, 0.5
    for thresh in np.linspace(0.05, 0.95, 19):
        f = f1_score(y_true, (probas >= thresh).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_thresh = f, float(thresh)
    return best_thresh


def _precision_at_k(y_true: pd.Series, probas: np.ndarray, k: int) -> float:
    """Fraction of actual positives in the top-k highest-probability predictions."""
    k = min(k, len(y_true))
    if k <= 0:
        return 0.0
    top_k_idx = np.argsort(probas)[::-1][:k]
    return float(y_true.iloc[top_k_idx].mean())


def _cv_metrics(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv,
    groups: pd.Series | None = None,
) -> dict[str, float]:
    base_keys = ["balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    prec_at_k_keys = [f"precision_at_{k}" for k in _PRECISION_AT_K]
    metric_keys = base_keys + prec_at_k_keys
    fold_results: list[dict[str, float]] = []

    for train_idx, test_idx in cv.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        if y_train.nunique() < 2 or y_test.nunique() < 2:
            continue
        model.fit(X_train, y_train)
        # Find optimal threshold on training split to avoid test-set leakage.
        train_probas = model.predict_proba(X_train)[:, 1]
        optimal_thresh = _find_optimal_threshold(y_train, train_probas)
        probas = model.predict_proba(X_test)[:, 1]
        preds = (probas >= optimal_thresh).astype(int)
        fold_result: dict[str, float] = {
            "balanced_accuracy": float(balanced_accuracy_score(y_test, preds)),
            "precision": float(precision_score(y_test, preds, zero_division=0)),
            "recall": float(recall_score(y_test, preds, zero_division=0)),
            "f1": float(f1_score(y_test, preds, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, probas)),
            "pr_auc": float(average_precision_score(y_test, probas)),
        }
        for k in _PRECISION_AT_K:
            fold_result[f"precision_at_{k}"] = _precision_at_k(y_test, probas, k)
        fold_results.append(fold_result)

    n_valid = len(fold_results)
    if not fold_results:
        return {k: 0.0 for k in metric_keys} | {"valid_cv_folds": 0}
    means = {k: round(sum(m[k] for m in fold_results) / n_valid, 4) for k in metric_keys}
    stds = {
        f"{k}_std": round(float(np.std([m[k] for m in fold_results])), 4)
        for k in ["roc_auc", "pr_auc", "recall", "precision_at_10", "precision_at_50"]
    }
    return means | stds | {"valid_cv_folds": n_valid}


def _cv_probabilities(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv,
    groups: pd.Series | None = None,
) -> pd.Series:
    """Return out-of-fold probabilities where each fold has both target classes."""
    probabilities = pd.Series(np.nan, index=y.index, dtype=float)
    for train_idx, test_idx in cv.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        if y_train.nunique() < 2 or y_test.nunique() < 2:
            continue
        model.fit(X_train, y_train)
        probabilities.iloc[test_idx] = model.predict_proba(X_test)[:, 1]
    return probabilities


def _feature_importance(model: Pipeline, feature_cols: list[str]) -> pd.DataFrame:
    clf = model.named_steps["clf"]
    if hasattr(clf, "coef_"):
        importances = clf.coef_[0]
        kind = "coefficient"
    elif hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        kind = "gain_importance"
    else:
        return pd.DataFrame()
    return (
        pd.DataFrame({"feature_name": feature_cols, "importance": importances, "importance_type": kind})
        .assign(abs_importance=lambda d: d["importance"].abs())
        .sort_values("abs_importance", ascending=False)
        .reset_index(drop=True)
    )


def _target_specs() -> dict[str, dict[str, str]]:
    return {
        "will_receive_any_violation_next_period": {
            "target_label": "any_violation",
            "target_role": "primary",
        },
        "will_receive_medium_or_high_risk_violation_next_period": {
            "target_label": "medium_or_high_violation",
            "target_role": "primary",
        },
        "will_receive_high_risk_violation_next_period": {
            "target_label": "high_risk_violation",
            "target_role": "secondary",
        },
    }


def _dedupe_features(columns: list[str]) -> list[str]:
    return list(dict.fromkeys(columns))


def _available_features(modeling_df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in _dedupe_features(columns) if column in modeling_df.columns]


def _ablation_feature_sets(modeling_df: pd.DataFrame) -> dict[str, list[str]]:
    behavioral = _available_features(modeling_df, BEHAVIORAL_FEATURES)
    static_property = _available_features(modeling_df, LEAKAGE_SAFE_STATIC_FEATURES + OWNER_FEATURE_COLS)
    service = _available_features(modeling_df, SERVICE_REQUEST_FEATURES)
    permits = _available_features(modeling_df, PERMIT_FEATURES)
    rentsmart = _available_features(modeling_df, RENTSMART_FEATURES)
    student = _available_features(modeling_df, STUDENT_HOUSING_FEATURES)

    layers: dict[str, list[str]] = {}
    current: list[str] = []
    for name, columns in [
        ("behavioral_only", behavioral),
        ("plus_static_property", static_property),
        ("plus_311", service),
        ("plus_permits", permits),
        ("plus_rentsmart", rentsmart),
        ("plus_student_housing", student),
    ]:
        current = _dedupe_features(current + columns)
        layers[name] = current.copy()
    return layers


def _risk_zip_groups(modeling_df: pd.DataFrame) -> pd.Series:
    if "violation_zip" in modeling_df.columns:
        groups = modeling_df["violation_zip"].map(normalize_zip).astype("string")
    elif "property_key" in modeling_df.columns:
        groups = modeling_df["property_key"].astype("string").str.split("|").str[-1].map(normalize_zip).astype("string")
    else:
        groups = pd.Series("unknown", index=modeling_df.index, dtype="string")
    return groups.fillna("unknown").replace("", "unknown")


def _rf_classifier(random_state: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=400,
        max_depth=6,
        min_samples_leaf=20,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )


def _build_ablation_table(
    modeling_df: pd.DataFrame,
    targets: dict[str, dict[str, str]],
    *,
    cv,
    random_state: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    layers = _ablation_feature_sets(modeling_df)
    for target_col, spec in targets.items():
        if target_col not in modeling_df.columns:
            continue
        y = modeling_df[target_col]
        if y.nunique() < 2:
            continue
        previous_roc: float | None = None
        previous_pr: float | None = None
        for order, (feature_set, feature_cols) in enumerate(layers.items(), start=1):
            if not feature_cols:
                continue
            metrics = _cv_metrics(
                _make_pipeline(_rf_classifier(random_state), feature_cols),
                modeling_df[feature_cols],
                y,
                cv,
            )
            row = {
                "target": target_col,
                "target_label": spec["target_label"],
                "target_role": spec["target_role"],
                "feature_layer_order": order,
                "feature_set": feature_set,
                "model_name": "random_forest",
                "n_features": len(feature_cols),
                "n_rows": len(modeling_df),
                "n_positive": int(y.sum()),
                "positive_class_rate": round(float(y.mean()), 4),
            }
            row.update(metrics)
            row["delta_roc_auc"] = round(row["roc_auc"] - previous_roc, 4) if previous_roc is not None else 0.0
            row["delta_pr_auc"] = round(row["pr_auc"] - previous_pr, 4) if previous_pr is not None else 0.0
            previous_roc = row["roc_auc"]
            previous_pr = row["pr_auc"]
            rows.append(row)
    return pd.DataFrame(rows)


def _build_grouped_cv_table(
    modeling_df: pd.DataFrame,
    targets: dict[str, dict[str, str]],
    feature_sets: dict[str, list[str]],
    *,
    random_state: int,
    cv_folds: int,
) -> pd.DataFrame:
    groups = _risk_zip_groups(modeling_df)
    n_groups = int(groups.nunique())
    if n_groups < 2:
        return pd.DataFrame()

    grouped_cv = GroupKFold(n_splits=min(cv_folds, n_groups))
    rows: list[dict] = []
    for target_col, spec in targets.items():
        if target_col not in modeling_df.columns:
            continue
        y = modeling_df[target_col]
        if y.nunique() < 2:
            continue
        for feature_set, feature_cols in feature_sets.items():
            feature_cols = _available_features(modeling_df, feature_cols)
            if not feature_cols:
                continue
            metrics = _cv_metrics(
                _make_pipeline(_rf_classifier(random_state), feature_cols),
                modeling_df[feature_cols],
                y,
                grouped_cv,
                groups=groups,
            )
            row = {
                "target": target_col,
                "target_label": spec["target_label"],
                "target_role": spec["target_role"],
                "feature_set": feature_set,
                "model_name": "random_forest",
                "cv_strategy": "zip_group_kfold",
                "n_groups": n_groups,
                "n_rows": len(modeling_df),
                "n_positive": int(y.sum()),
                "positive_class_rate": round(float(y.mean()), 4),
            }
            row.update(metrics)
            rows.append(row)
    return pd.DataFrame(rows)


def _build_calibration_table(
    modeling_df: pd.DataFrame,
    targets: dict[str, dict[str, str]],
    feature_cols: list[str],
    *,
    cv,
    random_state: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    for target_col, spec in targets.items():
        if target_col not in modeling_df.columns or spec["target_role"] != "primary":
            continue
        y = modeling_df[target_col]
        if y.nunique() < 2:
            continue
        probabilities = _cv_probabilities(
            _make_pipeline(_rf_classifier(random_state), feature_cols),
            modeling_df[feature_cols],
            y,
            cv,
        )
        valid = pd.DataFrame({"probability": probabilities, "actual": y}).dropna()
        n_bins = min(10, len(valid))
        if n_bins < 2:
            continue
        valid["risk_decile"] = pd.qcut(
            valid["probability"].rank(method="first"),
            q=n_bins,
            labels=False,
            duplicates="drop",
        ) + 1
        base_rate = float(valid["actual"].mean())
        calibration = (
            valid.groupby("risk_decile")
            .agg(
                n_properties=("actual", "size"),
                mean_predicted_risk=("probability", "mean"),
                observed_positive_rate=("actual", "mean"),
                positives=("actual", "sum"),
            )
            .reset_index()
        )
        calibration["target"] = target_col
        calibration["target_label"] = spec["target_label"]
        calibration["target_role"] = spec["target_role"]
        calibration["feature_set"] = "behavioral_plus_static"
        calibration["model_name"] = "random_forest"
        calibration["base_positive_rate"] = base_rate
        calibration["lift_vs_base_rate"] = (
            calibration["observed_positive_rate"].astype(float) / max(base_rate, 1e-9)
        )
        rows.extend(calibration.to_dict("records"))
    return pd.DataFrame(rows)


def _top_context_signals(
    row: pd.Series,
    signal_cols: list[str],
    limit: int = 6,
    importance_map: dict[str, float] | None = None,
) -> str:
    """List non-zero context fields, prioritizing globally important model features."""
    importance_map = importance_map or {}
    signals: list[tuple[str, float, float]] = []
    for column in signal_cols:
        value = row.get(column)
        if pd.isna(value):
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric > 0:
            signals.append((column, numeric, float(importance_map.get(column, 0.0))))
    signals.sort(key=lambda item: (item[2], abs(item[1])), reverse=True)
    return "; ".join(f"{column}={value:.2f}" for column, value, _ in signals[:limit])


def _build_top_risk_table(
    modeling_df: pd.DataFrame,
    targets: dict[str, dict[str, str]],
    feature_cols: list[str],
    *,
    random_state: int,
    limit: int = 100,
) -> pd.DataFrame:
    output_cols = [
        column
        for column in [
            "property_key",
            "violation_zip",
            "assessment_owner_clean",
            "total_violations",
            "open_violations",
            "recent_violation_count_365d",
            "recent_medium_or_high_risk_violation_count_365d",
            "service_requests_365d",
            "service_request_growth_365_vs_prior",
            "building_code_service_request_count",
            "sanitation_service_request_count",
            "permits_365d",
            "permit_intensity_730d",
            "rentsmart_record_count",
            "rentsmart_issue_type_count",
            "student_all_students",
            "students_per_modeled_property",
        ]
        if column in modeling_df.columns
    ]
    top_table = modeling_df[output_cols].copy()
    primary_importance: dict[str, float] = {}

    for target_col, spec in targets.items():
        if target_col not in modeling_df.columns or spec["target_role"] != "primary":
            continue
        y = modeling_df[target_col]
        if y.nunique() < 2:
            continue
        model = _make_pipeline(_rf_classifier(random_state), feature_cols)
        model.fit(modeling_df[feature_cols], y)
        top_table[f"risk_score_{spec['target_label']}"] = model.predict_proba(modeling_df[feature_cols])[:, 1]
        if not primary_importance or spec["target_label"] == "any_violation":
            importance_df = _feature_importance(model, feature_cols)
            if not importance_df.empty:
                primary_importance = dict(
                    zip(
                        importance_df["feature_name"],
                        pd.to_numeric(importance_df["abs_importance"], errors="coerce").fillna(0.0),
                    )
                )

    risk_score_cols = [column for column in top_table.columns if column.startswith("risk_score_")]
    if not risk_score_cols:
        return pd.DataFrame()

    signal_cols = [
        column
        for column in [
            "recent_violation_count_365d",
            "recent_medium_or_high_risk_violation_count_365d",
            "service_requests_365d",
            "service_request_growth_365_vs_prior",
            "building_code_service_request_count",
            "sanitation_service_request_count",
            "permits_365d",
            "permit_intensity_730d",
            "rentsmart_record_count",
            "rentsmart_issue_type_count",
            "student_all_students",
            "students_per_modeled_property",
        ]
        if column in top_table.columns
    ]
    top_table["top_context_signals"] = top_table.apply(
        _top_context_signals,
        axis=1,
        signal_cols=signal_cols,
        importance_map=primary_importance,
    )
    sort_col = "risk_score_any_violation" if "risk_score_any_violation" in top_table.columns else risk_score_cols[0]
    return top_table.sort_values(sort_col, ascending=False).head(limit).reset_index(drop=True)


# ── Main runner ───────────────────────────────────────────────────────────────

def run_improved_model(config: ImprovedModelConfig) -> Path:
    """Train LR / Random Forest / XGBoost; compare behavioral vs. full features; save results."""
    if not config.input_path.exists():
        raise FileNotFoundError(f"Missing cleaned violations at {config.input_path}. Run `make prepare-data` first.")
    if not config.property_risk_path.exists():
        raise FileNotFoundError(f"Missing property risk table at {config.property_risk_path}. Run `make pipeline` first.")

    source_df = load_phase2_source_data(
        Phase2FeatureConfig(
            input_path=config.input_path,
            raw_path=config.raw_path,
            output_path=Path("_unused.csv"),
        )
    )
    risk_df = pd.read_csv(config.property_risk_path, low_memory=False)
    permits_df = _read_optional_context_csv(config.permits_context_path, PERMIT_CONTEXT_COLUMNS)
    rentsmart_df = _read_optional_context_csv(config.rentsmart_context_path, RENTSMART_CONTEXT_COLUMNS)
    service_requests_df = _read_optional_context_csv(config.service_requests_path, SERVICE_REQUEST_CONTEXT_COLUMNS)
    student_housing_df = _read_optional_context_csv(config.student_housing_summary_path, STUDENT_HOUSING_CONTEXT_COLUMNS)

    print("Building improved modeling frame...")
    modeling_df = build_improved_modeling_frame(
        source_df,
        risk_df,
        prediction_window_days=config.prediction_window_days,
        permits_df=permits_df,
        rentsmart_df=rentsmart_df,
        service_requests_df=service_requests_df,
        student_housing_df=student_housing_df,
    )

    behavioral_cols = _available_features(modeling_df, BEHAVIORAL_FEATURES)
    static_cols = _available_features(modeling_df, MODEL_CONTEXT_FEATURES)
    full_cols = _dedupe_features(behavioral_cols + static_cols)
    targets = _target_specs()

    # Sort by last_violation_date so TimeSeriesSplit respects temporal order:
    # earlier-active properties train → later-active properties test.
    sort_col = "last_violation_date"
    if sort_col in modeling_df.columns:
        modeling_df = (
            modeling_df
            .sort_values(sort_col, na_position="first")
            .reset_index(drop=True)
        )
        print(f"Sorted modeling frame by {sort_col} for temporal CV.")
    else:
        print(f"Warning: {sort_col} not found; temporal ordering skipped.")

    cv = TimeSeriesSplit(n_splits=config.cv_folds)

    all_results: list[dict] = []
    importance_rows: list[pd.DataFrame] = []

    # ── Baseline LR (apples-to-apples) ──────────────────────────────────────
    # Run the same logistic-regression model used in baseline_model.py through
    # the identical TimeSeriesSplit + Precision@K pipeline so results can be
    # compared directly against RF / XGBoost rows in the same CSV.
    print("\n── Baseline LR (apples-to-apples) ─────────────────────────────────")
    baseline_lr_clf = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=config.random_state)
    for target_col, spec in targets.items():
        if target_col not in modeling_df.columns:
            continue
        target_label = spec["target_label"]
        y_bl = modeling_df[target_col]
        if y_bl.nunique() < 2:
            print(f"  Skipping {target_col}: only one class present.")
            continue
        n_pos_bl = int(y_bl.sum())
        pos_rate_bl = float(y_bl.mean())
        X_bl = modeling_df[behavioral_cols].copy()
        pipeline_bl = _make_pipeline(baseline_lr_clf, behavioral_cols)
        print(f"  [baseline_lr / {target_col}] CV{config.cv_folds}...", end=" ", flush=True)
        metrics_bl = _cv_metrics(pipeline_bl, X_bl, y_bl, cv)
        print(f"roc_auc={metrics_bl['roc_auc']:.4f}  pr_auc={metrics_bl['pr_auc']:.4f}  recall={metrics_bl['recall']:.4f}  valid_folds={metrics_bl.get('valid_cv_folds', '?')}")
        row_bl: dict = {
            "target": target_col,
            "target_label": target_label,
            "target_role": spec["target_role"],
            "feature_set": "baseline_lr",
            "model_name": "logistic_regression",
            "n_rows": len(modeling_df),
            "n_positive": n_pos_bl,
            "positive_class_rate": round(pos_rate_bl, 4),
            "cv_folds": config.cv_folds,
        }
        row_bl.update(metrics_bl)
        all_results.append(row_bl)
    print("────────────────────────────────────────────────────────────────────")

    for target_col, spec in targets.items():
        if target_col not in modeling_df.columns:
            continue
        target_label = spec["target_label"]
        y = modeling_df[target_col]
        if y.nunique() < 2:
            print(f"  Skipping {target_col}: only one class present.")
            continue

        n_pos = int(y.sum())
        n_neg = int((y == 0).sum())
        pos_rate = float(y.mean())
        print(f"\nTarget: {target_col}  positives={n_pos}/{len(y)} ({pos_rate:.2%})")

        for feature_set_name, feature_cols in [("behavioral_only", behavioral_cols), ("behavioral_plus_static", full_cols)]:
            X = modeling_df[feature_cols].copy()
            for model_name, clf in _classifiers(n_pos, n_neg, config.random_state):
                pipeline = _make_pipeline(clf, feature_cols)
                print(f"  [{feature_set_name}] {model_name} CV{config.cv_folds}...", end=" ", flush=True)
                metrics = _cv_metrics(pipeline, X, y, cv)
                print(f"roc_auc={metrics['roc_auc']:.4f}  pr_auc={metrics['pr_auc']:.4f}  recall={metrics['recall']:.4f}  valid_folds={metrics.get('valid_cv_folds', '?')}")

                row: dict = {
                    "target": target_col,
                    "target_label": target_label,
                    "target_role": spec["target_role"],
                    "feature_set": feature_set_name,
                    "model_name": model_name,
                    "n_rows": len(modeling_df),
                    "n_positive": n_pos,
                    "positive_class_rate": round(pos_rate, 4),
                    "cv_folds": config.cv_folds,
                }
                row.update(metrics)
                all_results.append(row)

                # Fit on full data for feature importance
                pipeline.fit(X, y)
                imp_df = _feature_importance(pipeline, feature_cols)
                if not imp_df.empty:
                    imp_df.insert(0, "model_name", model_name)
                    imp_df.insert(0, "feature_set", feature_set_name)
                    imp_df.insert(0, "target_role", spec["target_role"])
                    imp_df.insert(0, "target_label", target_label)
                    imp_df.insert(0, "target", target_col)
                    importance_rows.append(imp_df)

    results_df = pd.DataFrame(all_results)
    for output_path in [
        config.output_path,
        config.feature_importance_path,
        config.ablation_output_path,
        config.grouped_cv_output_path,
        config.top_risk_output_path,
        config.calibration_output_path,
    ]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(config.output_path, index=False)
    print(f"\nImproved model results saved to: {config.output_path}")

    if importance_rows:
        pd.concat(importance_rows, ignore_index=True).to_csv(config.feature_importance_path, index=False)
        print(f"Feature importance saved to: {config.feature_importance_path}")

    print("\nBuilding ablation table...")
    ablation_df = _build_ablation_table(
        modeling_df,
        targets,
        cv=cv,
        random_state=config.random_state,
    )
    ablation_df.to_csv(config.ablation_output_path, index=False)
    print(f"Ablation table saved to: {config.ablation_output_path}")

    print("Building ZIP grouped-CV robustness table...")
    grouped_cv_df = _build_grouped_cv_table(
        modeling_df,
        targets,
        {
            "behavioral_only": behavioral_cols,
            "behavioral_plus_static": full_cols,
        },
        random_state=config.random_state,
        cv_folds=config.cv_folds,
    )
    grouped_cv_df.to_csv(config.grouped_cv_output_path, index=False)
    print(f"Grouped-CV robustness table saved to: {config.grouped_cv_output_path}")

    print("Building calibration table...")
    calibration_df = _build_calibration_table(
        modeling_df,
        targets,
        full_cols,
        cv=cv,
        random_state=config.random_state,
    )
    calibration_df.to_csv(config.calibration_output_path, index=False)
    print(f"Calibration table saved to: {config.calibration_output_path}")

    print("Scoring top-risk properties...")
    top_risk_df = _build_top_risk_table(
        modeling_df,
        targets,
        full_cols,
        random_state=config.random_state,
    )
    top_risk_df.to_csv(config.top_risk_output_path, index=False)
    print(f"Top-risk property table saved to: {config.top_risk_output_path}")

    _print_summary(results_df)
    return config.output_path


def _print_summary(results_df: pd.DataFrame) -> None:
    """Print a compact comparison table."""
    print("\n── Model Comparison (CV) ──────────────────────────────────────────")
    cols = [
        "target_label", "target_role", "feature_set", "model_name",
        "roc_auc", "pr_auc", "recall", "balanced_accuracy",
        "precision_at_10", "precision_at_50", "precision_at_100",
        "valid_cv_folds",
    ]
    available = [c for c in cols if c in results_df.columns]
    display = results_df[available].copy()
    for metric in ["roc_auc", "pr_auc", "recall"]:
        std_col = f"{metric}_std"
        if metric in display.columns and std_col in results_df.columns:
            display[metric] = display[metric].astype(str) + " ±" + results_df[std_col].astype(str)
    print(display.to_string(index=False))
    print("────────────────────────────────────────────────────────────────────")

    if {"target_label", "roc_auc"}.issubset(results_df.columns):
        print("\nBest Model By Target (CV ROC AUC) ────────────────────────────────")
        best_by_target = (
            results_df
            .sort_values(["target_label", "roc_auc"], ascending=[True, False])
            .groupby("target_label", as_index=False)
            .head(1)
        )
        print(best_by_target[["target_label", "feature_set", "model_name", "roc_auc", "pr_auc"]].to_string(index=False))

    if {"target_label", "pr_auc"}.issubset(results_df.columns):
        print("\nBest Model By Target (CV PR AUC) ─────────────────────────────────")
        best_pr_by_target = (
            results_df
            .sort_values(["target_label", "pr_auc"], ascending=[True, False])
            .groupby("target_label", as_index=False)
            .head(1)
        )
        print(best_pr_by_target[["target_label", "feature_set", "model_name", "roc_auc", "pr_auc"]].to_string(index=False))

    if {"target_label", "roc_auc"}.issubset(results_df.columns):
        primary_rows = results_df.loc[results_df["target_label"].eq("any_violation")]
        primary_pool = primary_rows if not primary_rows.empty else results_df
        best = primary_pool.loc[primary_pool["roc_auc"].idxmax()]
        print(
            f"\nBest model for primary target: {best['model_name']} | feature_set={best['feature_set']} "
            f"| target={best['target_label']} | roc_auc={best['roc_auc']}"
        )


def main() -> None:
    run_improved_model(ImprovedModelConfig())


if __name__ == "__main__":
    main()
