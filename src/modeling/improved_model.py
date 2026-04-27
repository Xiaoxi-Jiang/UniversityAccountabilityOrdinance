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
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False

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

ALL_STATIC_FEATURES = (
    ACS_FEATURES
    + ASSESSMENT_FEATURES
    + RENTSMART_FEATURES
    + SERVICE_REQUEST_FEATURES
    + PERMIT_FEATURES
    + OWNER_FEATURE_COLS
)


@dataclass(frozen=True)
class ImprovedModelConfig:
    input_path: Path = DEFAULT_INPUT_PATH
    raw_path: Path = DEFAULT_RAW_PATH
    property_risk_path: Path = Path("data/processed/property_risk_table_v1.csv")
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


# ── Modeling frame ────────────────────────────────────────────────────────────

def build_improved_modeling_frame(
    source_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    *,
    prediction_window_days: int = 365,
) -> pd.DataFrame:
    """Temporal + static + owner features; also adds broader any-violation target."""
    modeling_df = build_property_level_modeling_frame(
        source_df, prediction_window_days=prediction_window_days
    )
    modeling_df["will_receive_any_violation_next_period"] = (
        modeling_df["future_violation_count"].gt(0).astype(int)
    )

    enriched_risk = add_owner_level_features(risk_df)
    static_available = [c for c in ALL_STATIC_FEATURES if c in enriched_risk.columns]
    new_cols = [c for c in static_available if c not in modeling_df.columns]

    if new_cols:
        modeling_df = modeling_df.merge(
            enriched_risk[["property_key"] + new_cols].drop_duplicates("property_key"),
            on="property_key",
            how="left",
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

def _cv_metrics(model: Pipeline, X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold) -> dict[str, float]:
    probas = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    preds = (probas >= 0.5).astype(int)
    return {
        "balanced_accuracy": round(float(balanced_accuracy_score(y, preds)), 4),
        "precision": round(float(precision_score(y, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y, preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y, preds, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y, probas)), 4),
        "pr_auc": round(float(average_precision_score(y, probas)), 4),
    }


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

    print("Building improved modeling frame...")
    modeling_df = build_improved_modeling_frame(
        source_df, risk_df, prediction_window_days=config.prediction_window_days
    )

    behavioral_cols = [c for c in BEHAVIORAL_FEATURES if c in modeling_df.columns]
    static_cols = [c for c in ALL_STATIC_FEATURES if c in modeling_df.columns]
    full_cols = behavioral_cols + static_cols

    targets = {
        "will_receive_any_violation_next_period": "any_violation",
        "will_receive_high_risk_violation_next_period": "high_risk_violation",
    }

    cv = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)

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
                print(f"roc_auc={metrics['roc_auc']:.4f}  pr_auc={metrics['pr_auc']:.4f}  recall={metrics['recall']:.4f}")

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
