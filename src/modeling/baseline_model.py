"""Property-level baseline model for the March check-in."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.features import (
    DEFAULT_INPUT_PATH,
    DEFAULT_RAW_PATH,
    Phase2FeatureConfig,
    build_feature_table,
    load_phase2_source_data,
    prepare_violations_frame,
)
from src.viz.phase2_visualizations import _derive_severity_proxy


HIGH_RISK_LABEL = "high risk (proxy)"
MEDIUM_RISK_LABEL = "medium risk (proxy)"
LOW_RISK_LABEL = "low risk (proxy)"
UNCATEGORIZED_LABEL = "uncategorized"


@dataclass(frozen=True)
class BaselineModelConfig:
    input_path: Path = DEFAULT_INPUT_PATH
    raw_path: Path = DEFAULT_RAW_PATH
    output_path: Path = Path("outputs/tables/baseline_model_results.csv")
    coefficients_output_path: Path = Path("outputs/tables/baseline_model_feature_coefficients.csv")
    prediction_window_days: int = 365
    test_size: float = 0.30
    random_state: int = 42


def _severity_count_columns(df: pd.DataFrame) -> pd.DataFrame:
    severity_counts = (
        df.assign(severity_proxy=df["severity_proxy"].fillna(UNCATEGORIZED_LABEL).astype("string"))
        .groupby(["property_key", "severity_proxy"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    rename_map = {
        HIGH_RISK_LABEL: "history_high_risk_violations",
        MEDIUM_RISK_LABEL: "history_medium_risk_violations",
        LOW_RISK_LABEL: "history_low_risk_violations",
        UNCATEGORIZED_LABEL: "history_uncategorized_violations",
    }
    severity_counts = severity_counts.rename(columns=rename_map)
    for column in rename_map.values():
        if column not in severity_counts.columns:
            severity_counts[column] = 0
    keep_cols = ["property_key", *rename_map.values()]
    return severity_counts.loc[:, keep_cols]


def _safe_share(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    values = numerator.astype(float) / denominator.replace(0, pd.NA).astype(float)
    return values.fillna(0.0)


def build_property_level_modeling_frame(
    source_df: pd.DataFrame,
    *,
    prediction_window_days: int = 365,
) -> pd.DataFrame:
    """Build a property-level frame for predicting next-period high-risk violations."""
    prepared, date_col = prepare_violations_frame(source_df)
    if date_col is None:
        raise ValueError("Baseline model requires a usable date column.")

    severity = _derive_severity_proxy(prepared)
    if severity is None:
        raise ValueError("Baseline model requires a violation type/description field for severity proxying.")

    prepared = prepared.copy()
    prepared["severity_proxy"] = severity.astype("string")
    prepared[date_col] = pd.to_datetime(prepared[date_col], errors="coerce")
    prepared = prepared.dropna(subset=[date_col]).copy()
    if prepared.empty:
        raise ValueError("Baseline model requires dated violations after preprocessing.")

    reference_date = prepared[date_col].max()
    cutoff_date = reference_date - pd.Timedelta(days=prediction_window_days)
    historical = prepared.loc[prepared[date_col].le(cutoff_date)].copy()
    future = prepared.loc[prepared[date_col].gt(cutoff_date)].copy()

    if historical.empty:
        raise ValueError("No historical rows remain before the prediction cutoff.")
    if future.empty:
        raise ValueError("No future rows remain inside the prediction window.")

    modeling_df = build_feature_table(historical)
    severity_counts = _severity_count_columns(historical)
    modeling_df = modeling_df.merge(severity_counts, on="property_key", how="left")

    future_targets = (
        future.groupby("property_key")
        .agg(
            future_violation_count=("property_key", "size"),
            future_high_risk_violation_count=(
                "severity_proxy",
                lambda values: int(values.astype("string").eq(HIGH_RISK_LABEL).sum()),
            ),
        )
        .reset_index()
    )
    modeling_df = modeling_df.merge(future_targets, on="property_key", how="left")

    fill_zero_cols = [
        "recent_violation_count_365d",
        "history_high_risk_violations",
        "history_medium_risk_violations",
        "history_low_risk_violations",
        "history_uncategorized_violations",
        "future_violation_count",
        "future_high_risk_violation_count",
    ]
    for column in fill_zero_cols:
        if column in modeling_df.columns:
            modeling_df[column] = modeling_df[column].fillna(0).astype(int)

    if "days_since_last_violation" in modeling_df.columns:
        modeling_df["days_since_last_violation"] = (
            modeling_df["days_since_last_violation"].fillna(modeling_df["days_since_last_violation"].median())
        )

    modeling_df["history_open_share"] = _safe_share(
        modeling_df["open_violations"],
        modeling_df["total_violations"],
    )
    modeling_df["history_high_risk_share"] = _safe_share(
        modeling_df["history_high_risk_violations"],
        modeling_df["total_violations"],
    )
    modeling_df["will_receive_high_risk_violation_next_period"] = (
        modeling_df["future_high_risk_violation_count"].gt(0).astype(int)
    )
    modeling_df["training_cutoff_date"] = cutoff_date.normalize()
    modeling_df["reference_date"] = reference_date.normalize()
    modeling_df["prediction_window_days"] = prediction_window_days

    return modeling_df


def run_baseline_model(config: BaselineModelConfig) -> Path:
    """Train a property-level logistic-regression baseline and save results."""
    if not config.input_path.exists():
        raise FileNotFoundError(
            f"Missing cleaned violations dataset at {config.input_path}. Run `make prepare-data` first."
        )

    source_df = load_phase2_source_data(
        Phase2FeatureConfig(
            input_path=config.input_path,
            raw_path=config.raw_path,
            output_path=Path("_unused_feature_table.csv"),
        )
    )
    modeling_df = build_property_level_modeling_frame(
        source_df,
        prediction_window_days=config.prediction_window_days,
    )

    target = modeling_df["will_receive_high_risk_violation_next_period"]
    if target.nunique() < 2:
        raise ValueError("Baseline model requires at least two target classes.")

    feature_columns = [
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
    feature_columns = [column for column in feature_columns if column in modeling_df.columns]
    X = modeling_df.loc[:, feature_columns].copy()
    y = target

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                feature_columns,
            )
        ],
        remainder="drop",
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=5000, class_weight="balanced")),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    cutoff_date = pd.to_datetime(modeling_df["training_cutoff_date"].iloc[0]).date().isoformat()
    reference_date = pd.to_datetime(modeling_df["reference_date"].iloc[0]).date().isoformat()
    positive_rate = float(y.mean())
    majority_class_accuracy = max(float(y_test.mean()), 1.0 - float(y_test.mean()))

    results = pd.DataFrame(
        [
            {
                "model_name": "logistic_regression",
                "unit_of_analysis": "property_key",
                "target_name": "will_receive_high_risk_violation_next_period",
                "target_definition": f"new {HIGH_RISK_LABEL} violation within next {config.prediction_window_days} days",
                "training_cutoff_date": cutoff_date,
                "reference_date": reference_date,
                "prediction_window_days": config.prediction_window_days,
                "n_rows": len(modeling_df),
                "n_train": len(X_train),
                "n_test": len(X_test),
                "n_positive": int(y.sum()),
                "n_positive_test": int(y_test.sum()),
                "positive_class_rate": round(positive_rate, 4),
                "majority_class_accuracy": round(majority_class_accuracy, 4),
                "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
                "balanced_accuracy": round(float(balanced_accuracy_score(y_test, predictions)), 4),
                "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
                "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
                "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
                "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
                "feature_count": len(feature_columns),
            }
        ]
    )

    classifier = model.named_steps["classifier"]
    coefficient_table = pd.DataFrame(
        {
            "feature_name": feature_columns,
            "standardized_coefficient": classifier.coef_[0],
        }
    )
    coefficient_table["abs_standardized_coefficient"] = coefficient_table[
        "standardized_coefficient"
    ].abs()
    coefficient_table["odds_ratio_per_sd_increase"] = np.exp(
        coefficient_table["standardized_coefficient"]
    )
    coefficient_table = coefficient_table.sort_values(
        "abs_standardized_coefficient",
        ascending=False,
    ).reset_index(drop=True)

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(config.output_path, index=False)
    coefficient_table.to_csv(config.coefficients_output_path, index=False)

    row = results.iloc[0]
    print(
        "Baseline model summary: "
        f"unit={row['unit_of_analysis']}, "
        f"target={row['target_name']}, "
        f"balanced_accuracy={row['balanced_accuracy']:.4f}, "
        f"precision={row['precision']:.4f}, "
        f"recall={row['recall']:.4f}, "
        f"f1={row['f1']:.4f}, "
        f"roc_auc={row['roc_auc']:.4f}"
    )
    print(f"Baseline model results saved to: {config.output_path}")
    print(f"Baseline model coefficients saved to: {config.coefficients_output_path}")
    return config.output_path


def main() -> None:
    run_baseline_model(BaselineModelConfig())


if __name__ == "__main__":
    main()
