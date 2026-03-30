"""Top-level reproducible project pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analysis.checkin_summary import write_checkin_summary
from src.analysis.eda import (
    EDAConfig,
    get_available_date_column,
    run_eda,
    summarize_figure_results,
)
from src.data.context.property import PropertyDataConfig, save_property_risk_table
from src.data.context.student_housing import (
    StudentHousingConfig,
    save_student_housing_context,
)
from src.data.features import (
    DEFAULT_INPUT_PATH,
    DEFAULT_RAW_PATH,
    Phase2FeatureConfig,
    get_property_key_diagnostics,
    load_phase2_source_data,
    prepare_violations_frame,
    run_phase2_feature_engineering,
)
from src.modeling.baseline_model import BaselineModelConfig, run_baseline_model


def _print_paths(label: str, paths: list[Path]) -> None:
    for path in paths:
        print(f"{label}: {path}")


def _print_pipeline_summary(
    *,
    feature_path: Path,
    property_risk_path: Path,
    student_context_path: Path | None,
    table_paths: list[Path],
    figure_paths: list[Path],
    model_path: Path,
    feature_date_col: str | None,
    trend_date_col: str | None,
    property_key_diagnostics: dict[str, float | int | list[str] | dict[str, int]],
    property_risk_diagnostics: dict[str, object],
    student_housing_diagnostics: dict[str, object],
    generated_figures: list[str],
    skipped_figures: list[str],
    checkin_summary_path: Path,
) -> None:
    summary_lines = [
        "Pipeline completed.",
        f"Feature table: {feature_path}",
        f"Property risk table: {property_risk_path}",
        f"Student housing context: {student_context_path or 'skipped'}",
        f"Baseline model table: {model_path}",
        f"Feature date column available: {feature_date_col or 'none'}",
        f"Trend date column used: {trend_date_col or 'none'}",
        f"Unique property keys: {property_key_diagnostics['unique_property_keys']}",
        "Singleton property keys (%): "
        f"{property_key_diagnostics['singleton_property_key_pct']:.1f}",
        "Case-no fallback keys (%): "
        f"{property_key_diagnostics['case_no_fallback_pct']:.1f}",
        "Address-based property keys (%): "
        f"{property_key_diagnostics['address_based_key_pct']:.1f}",
        "Property datasets loaded: "
        f"sam={property_risk_diagnostics.get('sam_loaded')}, "
        f"assessment={property_risk_diagnostics.get('property_assessment_loaded')}, "
        f"parcels={property_risk_diagnostics.get('parcels_loaded')}, "
        f"rentsmart={property_risk_diagnostics.get('rentsmart_loaded')}, "
        f"service_requests={property_risk_diagnostics.get('service_requests_loaded')}, "
        f"permits={property_risk_diagnostics.get('permits_loaded')}, "
        f"acs={property_risk_diagnostics.get('acs_loaded')}",
        "Property join diagnostics (%): "
        f"sam={property_risk_diagnostics.get('sam_match_rate_pct', 0.0)}, "
        f"assessment={property_risk_diagnostics.get('assessment_match_rate_pct', 0.0)}, "
        f"parcels={property_risk_diagnostics.get('parcel_match_rate_pct', 0.0)}, "
        f"rentsmart={property_risk_diagnostics.get('rentsmart_match_rate_pct', 0.0)}, "
        f"service_requests={property_risk_diagnostics.get('service_request_context_rate_pct', 0.0)}, "
        f"permits={property_risk_diagnostics.get('permit_context_rate_pct', 0.0)}, "
        f"acs={property_risk_diagnostics.get('acs_context_rate_pct', 0.0)}",
        "Owner / parcel context coverage (%): "
        f"owner_data={property_risk_diagnostics.get('owner_data_rate_pct', 0.0)}, "
        f"parcel_context={property_risk_diagnostics.get('parcel_context_rate_pct', 0.0)}",
        "Student housing available: "
        f"{student_housing_diagnostics.get('student_housing_available')}",
        f"Check-in summary: {checkin_summary_path}",
    ]
    for line in summary_lines:
        print(line)
    if student_housing_diagnostics.get("student_context_join"):
        print(
            "Student housing join strategy: "
            f"{student_housing_diagnostics.get('student_context_join')}"
        )
    _print_paths("EDA table", table_paths)
    _print_paths("Figure", figure_paths)
    for label in generated_figures:
        print(f"Generated figure group: {label}")
    for label in skipped_figures:
        print(f"Skipped figure group: {label}")


def ensure_output_directories() -> None:
    """Create output directories used by the project pipeline."""
    for path in [Path("data/processed"), Path("outputs/tables"), Path("outputs/figures")]:
        path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ensure_output_directories()

    if not DEFAULT_INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing cleaned violations dataset at {DEFAULT_INPUT_PATH}. Run `make prepare-data` first."
        )

    feature_config = Phase2FeatureConfig()
    source_df = load_phase2_source_data(feature_config)
    prepared, feature_date_col = prepare_violations_frame(source_df)
    property_key_diagnostics = get_property_key_diagnostics(prepared)
    trend_date_col = get_available_date_column(prepared.copy())

    feature_path = run_phase2_feature_engineering(feature_config)
    property_risk_path, property_risk_diagnostics = save_property_risk_table(
        PropertyDataConfig(),
        feature_path,
    )
    property_risk_df = pd.read_csv(property_risk_path, low_memory=False)
    student_context_path, student_housing_diagnostics = save_student_housing_context(
        StudentHousingConfig(),
        property_risk_df,
    )

    table_paths, figure_paths = run_eda(
        EDAConfig(
            property_risk_path=property_risk_path,
            student_context_path=student_context_path
            or StudentHousingConfig().output_path,
        )
    )
    baseline_config = BaselineModelConfig(
        input_path=DEFAULT_INPUT_PATH,
        raw_path=DEFAULT_RAW_PATH,
    )
    model_path = run_baseline_model(baseline_config)
    checkin_summary_path = write_checkin_summary(
        output_path=Path("outputs/checkin_summary.md"),
        feature_path=feature_path,
        property_risk_path=property_risk_path,
        student_context_path=student_context_path,
        model_results_path=model_path,
        coefficients_path=baseline_config.coefficients_output_path,
        tables_dir=Path("outputs/tables"),
        property_key_diagnostics=property_key_diagnostics,
        property_risk_diagnostics=property_risk_diagnostics,
        student_housing_diagnostics=student_housing_diagnostics,
    )
    generated_figures, skipped_figures = summarize_figure_results(figure_paths)
    _print_pipeline_summary(
        feature_path=feature_path,
        property_risk_path=property_risk_path,
        student_context_path=student_context_path,
        table_paths=table_paths,
        figure_paths=figure_paths,
        model_path=model_path,
        feature_date_col=feature_date_col,
        trend_date_col=trend_date_col,
        property_key_diagnostics=property_key_diagnostics,
        property_risk_diagnostics=property_risk_diagnostics,
        student_housing_diagnostics=student_housing_diagnostics,
        generated_figures=generated_figures,
        skipped_figures=skipped_figures,
        checkin_summary_path=checkin_summary_path,
    )


if __name__ == "__main__":
    main()
