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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False

from src.data.context.property import normalize_property_identifier
from src.data.features import DEFAULT_INPUT_PATH, DEFAULT_RAW_PATH, Phase2FeatureConfig, load_phase2_source_data
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
]

RENTSMART_FEATURES = [
    "rentsmart_record_count",
    "rentsmart_complaint_indicator",
]

SERVICE_REQUEST_FEATURES = [
    "service_request_count",
    "housing_related_service_request_count",
    "service_requests_365d",
]

PERMIT_FEATURES = [
    "permit_count",
    "major_permit_count",
    "permits_730d",
]

OWNER_FEATURE_COLS = [
    "owner_property_count",
    "owner_total_violations",
    "owner_avg_violations_per_property",
    "owner_high_risk_share",
]

LEAKAGE_SAFE_STATIC_FEATURES = ACS_FEATURES + ASSESSMENT_FEATURES
# 311 is excluded until the local extract contains records before the training cutoff.
EXCLUDED_TEMPORAL_CONTEXT_FEATURES = SERVICE_REQUEST_FEATURES
MODEL_CONTEXT_FEATURES = LEAKAGE_SAFE_STATIC_FEATURES + OWNER_FEATURE_COLS + RENTSMART_FEATURES + PERMIT_FEATURES

PERMIT_CONTEXT_COLUMNS = [
    "map_par_id",
    "pid",
    "address_zip_key",
    "permit_issue_date",
    "major_permit_flag",
    "permit_record_count",
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


@dataclass(frozen=True)
class ImprovedModelConfig:
    input_path: Path = DEFAULT_INPUT_PATH
    raw_path: Path = DEFAULT_RAW_PATH
    property_risk_path: Path = Path("data/processed/property_risk_table_v1.csv")
    permits_context_path: Path = Path("data/processed/building_permits_clean.csv")
    rentsmart_context_path: Path = Path("data/processed/rentsmart_clean.csv")
    output_path: Path = Path("outputs/tables/improved_model_results.csv")
    feature_importance_path: Path = Path("outputs/tables/improved_model_feature_importance.csv")
    prediction_window_days: int = 365
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
    working["permits_730d"] = (
        working["permit_issue_date"]
        .ge(cutoff_date - pd.Timedelta(days=730))
        .fillna(False)
        .astype(int)
    )

    return (
        working.groupby(join_key)
        .agg(
            permit_count=("permit_record_count", "sum"),
            major_permit_count=("major_permit_flag", "sum"),
            permits_730d=("permits_730d", "sum"),
        )
        .reset_index()
    )


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

    complaint_cols = [
        column
        for column in working.columns
        if any(token in column for token in ["complaint", "violation", "inspection", "issue", "type", "description"])
    ]
    complaint_signal = working[complaint_cols].notna().any(axis=1).astype(int) if complaint_cols else 1
    working["rentsmart_record_count"] = 1
    working["rentsmart_complaint_indicator"] = complaint_signal

    return (
        working.groupby(join_key)
        .agg(
            rentsmart_record_count=("rentsmart_record_count", "sum"),
            rentsmart_complaint_indicator=("rentsmart_complaint_indicator", "max"),
        )
        .reset_index()
    )


def add_cutoff_safe_temporal_context_features(
    modeling_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    *,
    permits_df: pd.DataFrame | None = None,
    rentsmart_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add permit/RentSmart features after filtering source records to the model cutoff."""
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

    for context in context_frames:
        if len(context.columns) > 1:
            modeling_df = modeling_df.merge(context, on="property_key", how="left")

    for column in PERMIT_FEATURES + RENTSMART_FEATURES:
        if column in modeling_df.columns:
            modeling_df[column] = modeling_df[column].fillna(0)
    return modeling_df


# ── Modeling frame ────────────────────────────────────────────────────────────

def build_improved_modeling_frame(
    source_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    *,
    prediction_window_days: int = 365,
    permits_df: pd.DataFrame | None = None,
    rentsmart_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Temporal + static + owner features; also adds broader any-violation target."""
    modeling_df = build_property_level_modeling_frame(
        source_df, prediction_window_days=prediction_window_days
    )
    modeling_df["will_receive_any_violation_next_period"] = (
        modeling_df["future_violation_count"].gt(0).astype(int)
    )

    if {"property_key", "assessment_owner_clean"}.issubset(risk_df.columns):
        owner_lookup = risk_df[["property_key", "assessment_owner_clean"]].drop_duplicates("property_key")
        modeling_df = modeling_df.merge(owner_lookup, on="property_key", how="left")
    modeling_df = add_owner_level_features(modeling_df)

    static_available = [c for c in LEAKAGE_SAFE_STATIC_FEATURES if c in risk_df.columns]
    new_cols = [c for c in static_available if c not in modeling_df.columns]

    if new_cols and "property_key" in risk_df.columns:
        modeling_df = modeling_df.merge(
            risk_df[["property_key"] + new_cols].drop_duplicates("property_key"),
            on="property_key",
            how="left",
        )

    modeling_df = add_cutoff_safe_temporal_context_features(
        modeling_df,
        risk_df,
        permits_df=permits_df,
        rentsmart_df=rentsmart_df,
    )

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
        ("random_forest", RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=random_state, n_jobs=-1)),
    ]
    if _XGBOOST_AVAILABLE:
        models.append((
            "xgboost",
            XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                scale_pos_weight=scale,
                eval_metric="logloss",
                random_state=random_state,
                verbosity=0,
            ),
        ))
    return models


# ── CV evaluation ─────────────────────────────────────────────────────────────

def _cv_metrics(model: Pipeline, X: pd.DataFrame, y: pd.Series, cv: TimeSeriesSplit) -> dict[str, float]:
    metric_keys = ["balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    fold_results: list[dict[str, float]] = []

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        # Skip folds where the test set contains only one class (roc_auc undefined)
        if y_test.nunique() < 2:
            continue
        model.fit(X_train, y_train)
        probas = model.predict_proba(X_test)[:, 1]
        preds = (probas >= 0.5).astype(int)
        fold_results.append({
            "balanced_accuracy": float(balanced_accuracy_score(y_test, preds)),
            "precision": float(precision_score(y_test, preds, zero_division=0)),
            "recall": float(recall_score(y_test, preds, zero_division=0)),
            "f1": float(f1_score(y_test, preds, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, probas)),
            "pr_auc": float(average_precision_score(y_test, probas)),
        })

    n_valid = len(fold_results)
    if not fold_results:
        return {k: 0.0 for k in metric_keys} | {"valid_cv_folds": 0}
    return {k: round(sum(m[k] for m in fold_results) / n_valid, 4) for k in metric_keys} | {"valid_cv_folds": n_valid}


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

    print("Building improved modeling frame...")
    modeling_df = build_improved_modeling_frame(
        source_df,
        risk_df,
        prediction_window_days=config.prediction_window_days,
        permits_df=permits_df,
        rentsmart_df=rentsmart_df,
    )

    behavioral_cols = [c for c in BEHAVIORAL_FEATURES if c in modeling_df.columns]
    static_cols = [c for c in MODEL_CONTEXT_FEATURES if c in modeling_df.columns]
    full_cols = behavioral_cols + static_cols

    targets = {
        "will_receive_any_violation_next_period": "any_violation",
        "will_receive_high_risk_violation_next_period": "high_risk_violation",
    }

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

    for target_col, target_label in targets.items():
        if target_col not in modeling_df.columns:
            continue
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
                    imp_df.insert(0, "target", target_col)
                    importance_rows.append(imp_df)

    results_df = pd.DataFrame(all_results)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(config.output_path, index=False)
    print(f"\nImproved model results saved to: {config.output_path}")

    if importance_rows:
        pd.concat(importance_rows, ignore_index=True).to_csv(config.feature_importance_path, index=False)
        print(f"Feature importance saved to: {config.feature_importance_path}")

    _print_summary(results_df)
    return config.output_path


def _print_summary(results_df: pd.DataFrame) -> None:
    """Print a compact comparison table."""
    print("\n── Model Comparison (CV) ──────────────────────────────────────────")
    cols = ["target_label", "feature_set", "model_name", "roc_auc", "pr_auc", "recall", "balanced_accuracy"]
    available = [c for c in cols if c in results_df.columns]
    print(results_df[available].to_string(index=False))
    print("────────────────────────────────────────────────────────────────────")

    if "roc_auc" in results_df.columns:
        best = results_df.loc[results_df["roc_auc"].idxmax()]
        print(
            f"\nBest model: {best['model_name']} | feature_set={best['feature_set']} "
            f"| target={best['target_label']} | roc_auc={best['roc_auc']}"
        )


def main() -> None:
    run_improved_model(ImprovedModelConfig())


if __name__ == "__main__":
    main()
