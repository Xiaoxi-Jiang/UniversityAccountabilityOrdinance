"""Write a concise March check-in summary from pipeline artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _read_optional_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _bool_label(value: object) -> str:
    return "yes" if bool(value) else "no"


def write_checkin_summary(
    *,
    output_path: Path,
    feature_path: Path,
    property_risk_path: Path,
    student_context_path: Path | None,
    model_results_path: Path,
    coefficients_path: Path,
    tables_dir: Path,
    property_key_diagnostics: dict[str, Any],
    property_risk_diagnostics: dict[str, Any],
    student_housing_diagnostics: dict[str, Any],
) -> Path:
    """Persist a markdown summary that explains the current project state."""
    model_results = pd.read_csv(model_results_path).iloc[0]
    coefficient_df = pd.read_csv(coefficients_path)
    owner_summary = _read_optional_csv(tables_dir / "owner_data_availability_summary.csv")
    student_corr = _read_optional_csv(tables_dir / "student_housing_correlation_summary.csv")

    top_positive = coefficient_df.sort_values("standardized_coefficient", ascending=False).head(3)
    top_negative = coefficient_df.sort_values("standardized_coefficient", ascending=True).head(3)

    lines = [
        "# March Check-In Summary",
        "",
        "## Project Framing",
        (
            "This pipeline now treats the baseline task as a property-level prediction problem: "
            "given each property's historical violations up to the cutoff date, predict whether it "
            "will receive a new high-risk violation during the next prediction window."
        ),
        "",
        "## Data Processing Progress",
        f"- Core cleaned violations table: `{feature_path.parent / 'violations_clean.csv'}`",
        f"- Property-level feature table: `{feature_path}`",
        f"- Enriched property-risk table: `{property_risk_path}`",
        f"- Student housing context output: `{student_context_path}`" if student_context_path else "- Student housing context output: skipped",
        (
            "- Property-key coverage relies primarily on normalized address and ZIP joins; "
            f"{property_key_diagnostics['address_based_key_pct']:.1f}% of rows use address-based keys and "
            f"{property_key_diagnostics['case_no_fallback_pct']:.1f}% fall back to case numbers."
        ),
        (
            "- Context coverage from optional sources: "
            f"SAM={_bool_label(property_risk_diagnostics.get('sam_loaded'))}, "
            f"assessment={_bool_label(property_risk_diagnostics.get('property_assessment_loaded'))}, "
            f"parcels={_bool_label(property_risk_diagnostics.get('parcels_loaded'))}, "
            f"311={_bool_label(property_risk_diagnostics.get('service_requests_loaded'))}, "
            f"permits={_bool_label(property_risk_diagnostics.get('permits_loaded'))}, "
            f"ACS={_bool_label(property_risk_diagnostics.get('acs_loaded'))}."
        ),
    ]

    if owner_summary is not None and not owner_summary.empty:
        owner_row = owner_summary.iloc[0]
        lines.append(
            "- Owner coverage in the property-risk table: "
            f"{float(owner_row['owner_data_available_pct']):.1f}% "
            f"({int(owner_row['rows_with_owner_data'])} of {int(owner_row['total_rows'])} properties)."
        )

    lines.extend(
        [
            "",
            "## Modeling Method",
            (
                "- Target: `will_receive_high_risk_violation_next_period`, defined as at least one new "
                f"`high risk (proxy)` violation within {int(model_results['prediction_window_days'])} days "
                f"after the cutoff date {model_results['training_cutoff_date']}."
            ),
            (
                "- Feature groups: historical volume (`total_violations`, `recent_violation_count_365d`), "
                "recency (`days_since_last_violation`), diversity (`distinct_violation_types`), "
                "and severity composition "
                "(`history_high_risk_violations`, `history_medium_risk_violations`, "
                "`history_low_risk_violations`, `history_high_risk_share`)."
            ),
            (
                "- Model: logistic regression with standardized numeric features and balanced class "
                "weights to offset the rare-event target."
            ),
            "",
            "Top positive coefficient directions:",
        ]
    )
    lines.extend(
        [
            f"- `{row.feature_name}` ({row.standardized_coefficient:.3f})"
            for row in top_positive.itertuples(index=False)
        ]
    )
    lines.append("")
    lines.append("Top negative coefficient directions:")
    lines.extend(
        [
            f"- `{row.feature_name}` ({row.standardized_coefficient:.3f})"
            for row in top_negative.itertuples(index=False)
        ]
    )

    lines.extend(
        [
            "",
            "## Student Housing Relationship",
            (
                "- Student housing is now analyzed directly with ZIP-level relationship outputs, including "
                "`student_housing_relationship.csv`, `student_housing_correlation_summary.csv`, and "
                "`student_housing_relationship.png`."
            ),
        ]
    )
    if student_corr is not None and not student_corr.empty:
        student_row = student_corr.iloc[0]
        lines.append(
            "- Matched ZIP summary: "
            f"{int(student_row['matched_zip_count'])} ZIP codes with student metric "
            f"`{student_row['student_metric_column']}`; "
            f"correlation with total violations = {float(student_row['pearson_corr_total_violations']):.4f}, "
            "correlation with violations per property = "
            f"{float(student_row['pearson_corr_violations_per_property']):.4f}."
        )
    else:
        lines.append(
            "- Student housing relationship outputs were skipped because no usable student metric was available."
        )

    lines.extend(
        [
            "",
            "## Preliminary Results and Interpretation",
            (
                f"- Modeling frame: {int(model_results['n_rows'])} properties, "
                f"{int(model_results['n_positive'])} positive examples "
                f"({float(model_results['positive_class_rate']):.4f} positive rate)."
            ),
            (
                "- Holdout metrics: "
                f"accuracy={float(model_results['accuracy']):.4f}, "
                f"balanced_accuracy={float(model_results['balanced_accuracy']):.4f}, "
                f"precision={float(model_results['precision']):.4f}, "
                f"recall={float(model_results['recall']):.4f}, "
                f"f1={float(model_results['f1']):.4f}, "
                f"roc_auc={float(model_results['roc_auc']):.4f}."
            ),
            (
                f"- Majority-class accuracy is {float(model_results['majority_class_accuracy']):.4f}, "
                "so balanced accuracy and recall are more informative than raw accuracy because the "
                "target event is rare."
            ),
            (
                "- Current limitations: severity is still proxy-based rather than an official city severity "
                "field; student housing is measured mostly at ZIP level; and the baseline model only uses "
                "historical violation behavior, not the full static property context yet."
            ),
            (
                "- Student housing context is therefore informative for exploration, but not yet evidence of "
                "a causal relationship between student concentration and violations."
            ),
            "",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Check-in summary saved to: {output_path}")
    return output_path
