"""EDA entrypoints for summary tables and exploratory figures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from src.data.features import (
    Phase2FeatureConfig,
    VIOLATION_TYPE_CANDIDATES,
    clean_violator_name,
    first_available_column,
    load_phase2_source_data,
    prepare_violations_frame,
)
from src.viz.plot_utils import plt, save_figure, truncate_labels
from src.viz.phase2_visualizations import (
    Phase2VisualizationConfig,
    generate_phase2_figures,
    get_available_date_column,
    summarize_figure_results,
)


@dataclass(frozen=True)
class EDAConfig:
    input_path: Path = Path("data/processed/violations_clean.csv")
    tables_dir: Path = Path("outputs/tables")
    figures_dir: Path = Path("outputs/figures")
    property_risk_path: Path = Path("data/processed/property_risk_table_v1.csv")
    student_context_path: Path = Path("data/processed/student_housing_context_v1.csv")


@dataclass(frozen=True)
class Phase2EDASummaryConfig:
    input_path: Path = Phase2FeatureConfig().input_path
    output_dir: Path = Path("outputs/tables")


def _load_prepared_violations(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing cleaned violations dataset at {input_path}. Run Phase 1 first."
        )
    df = load_phase2_source_data(Phase2FeatureConfig(input_path=input_path))
    prepared, _ = prepare_violations_frame(df)
    return prepared


def _write_table(df: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved table: {output_path}")
    return output_path


def _save_barh_figure(
    labels: pd.Series,
    values: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
) -> Path:
    plt.figure(figsize=(10, 6))
    plt.barh(truncate_labels(labels), values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return save_figure(output_path)


def _series_count_table(series: pd.Series, index_name: str, *, sort_index: bool = False) -> pd.DataFrame:
    counts = series.value_counts(dropna=False)
    if sort_index:
        counts = counts.sort_index()
    return counts.rename_axis(index_name).reset_index(name="count")


def _append_summary_table(
    output_paths: list[Path],
    df: pd.DataFrame,
    column: str,
    output_path: Path,
    index_name: str,
    *,
    transform=None,
    sort_index: bool = False,
) -> None:
    if column not in df.columns:
        return
    series = df[column]
    if transform is not None:
        series = transform(series)
    table = _series_count_table(series, index_name, sort_index=sort_index)
    output_paths.append(_write_table(table, output_path))


def generate_eda_tables(config: Phase2EDASummaryConfig) -> list[Path]:
    """Create small CSV summaries from the cleaned violations table."""
    prepared = _load_prepared_violations(config.input_path)
    output_paths: list[Path] = []

    _append_summary_table(
        output_paths,
        prepared,
        "status",
        config.output_dir / "status_counts.csv",
        "status",
        transform=lambda values: values.astype("string").fillna("unknown"),
    )
    _append_summary_table(
        output_paths,
        prepared,
        "year",
        config.output_dir / "year_counts.csv",
        "year",
        transform=lambda values: values.dropna().astype(int),
        sort_index=True,
    )
    _append_summary_table(
        output_paths,
        prepared,
        "month",
        config.output_dir / "month_counts.csv",
        "month",
        transform=lambda values: values.dropna().astype(int),
        sort_index=True,
    )

    violation_type_col = first_available_column(prepared, VIOLATION_TYPE_CANDIDATES)
    if violation_type_col is not None:
        top_violation_types = (
            prepared[violation_type_col]
            .astype("string")
            .fillna("unknown")
            .value_counts()
            .head(20)
            .rename_axis("violation_type")
            .reset_index(name="count")
        )
        output_paths.append(
            _write_table(top_violation_types, config.output_dir / "top_violation_types.csv")
        )

    if "violator_name" in prepared.columns:
        top_violators = (
            clean_violator_name(prepared["violator_name"])
            .replace("", pd.NA)
            .dropna()
            .value_counts()
            .head(20)
            .rename_axis("violator_name_clean")
            .reset_index(name="count")
        )
        if not top_violators.empty:
            output_paths.append(_write_table(top_violators, config.output_dir / "top_violators.csv"))

    return output_paths


def generate_property_risk_outputs(
    property_risk_path: Path,
    tables_dir: Path,
    figures_dir: Path,
) -> tuple[list[Path], list[Path]]:
    """Generate optional EDA artifacts from the property-risk layer."""
    if not property_risk_path.exists():
        print("Skipping property-risk EDA: property risk table is unavailable.")
        return [], []

    risk_df = pd.read_csv(property_risk_path)
    table_paths: list[Path] = []
    figure_paths: list[Path] = []

    if {"property_key", "total_violations"}.issubset(risk_df.columns):
        repeated = risk_df.loc[risk_df["total_violations"] >= 2].copy()
        if not repeated.empty:
            owner_col = next(
                (column for column in ["assessment_owner_clean", "assessment_owner"] if column in repeated.columns),
                None,
            )
            class_col = next(
                (column for column in ["assessment_lu_desc", "assessment_bldg_type"] if column in repeated.columns),
                None,
            )
            if owner_col is not None or class_col is not None:
                repeated_table = repeated.loc[
                    :,
                    [
                        column
                        for column in [
                            "property_key",
                            "violation_st",
                            "violation_zip",
                            "total_violations",
                            owner_col,
                            class_col,
                        ]
                        if column is not None and column in repeated.columns
                    ],
                ].sort_values("total_violations", ascending=False).head(20)
                table_paths.append(
                    _write_table(
                        repeated_table,
                        tables_dir / "top_repeated_properties_with_owner.csv",
                    )
                )
                plot_df = repeated_table.head(15).sort_values("total_violations", ascending=True)
                figure_paths.append(
                    _save_barh_figure(
                        plot_df["property_key"],
                        plot_df["total_violations"],
                        "Top Repeated Properties With Ownership Context",
                        "Violation Count",
                        "Property",
                        figures_dir / "top_repeated_properties_with_owner.png",
                    )
                )
            else:
                print("Skipping ownership-context EDA: property owner/class fields are unavailable.")

    property_class_col = next(
        (column for column in ["assessment_lu_desc", "assessment_bldg_type"] if column in risk_df.columns),
        None,
    )
    if property_class_col is not None and "total_violations" in risk_df.columns:
        class_summary = (
            risk_df.dropna(subset=[property_class_col])
            .groupby(property_class_col)
            .agg(
                property_count=("property_key", "nunique"),
                total_violations=("total_violations", "sum"),
                open_violations=("open_violations", "sum"),
            )
            .reset_index()
            .sort_values("total_violations", ascending=False)
            .head(15)
        )
        if not class_summary.empty:
            table_paths.append(
                _write_table(class_summary, tables_dir / "violations_by_property_class.csv")
            )
            plot_df = class_summary.sort_values("total_violations", ascending=True)
            figure_paths.append(
                _save_barh_figure(
                    plot_df[property_class_col],
                    plot_df["total_violations"],
                    "Violations by Property Class",
                    "Violation Count",
                    "Property Class",
                    figures_dir / "violations_by_property_class.png",
                )
            )

    if "owner_data_available_flag" in risk_df.columns:
        owner_summary = pd.DataFrame(
            [
                {
                    "owner_data_available_pct": round(
                        float(risk_df["owner_data_available_flag"].mean() * 100),
                        1,
                    ),
                    "rows_with_owner_data": int(risk_df["owner_data_available_flag"].sum()),
                    "total_rows": int(len(risk_df)),
                }
            ]
        )
        table_paths.append(
            _write_table(owner_summary, tables_dir / "owner_data_availability_summary.csv")
        )

    if "rentsmart_match_flag" in risk_df.columns and risk_df["rentsmart_match_flag"].sum() > 0:
        rentsmart_summary = pd.DataFrame(
            [
                {
                    "properties_with_rentsmart_context": int(risk_df["rentsmart_match_flag"].sum()),
                    "total_properties": int(len(risk_df)),
                    "match_rate_pct": round(
                        float(risk_df["rentsmart_match_flag"].mean() * 100),
                        1,
                    ),
                    "complaint_indicator_pct": round(
                        float(risk_df.get("rentsmart_complaint_indicator", pd.Series(0, index=risk_df.index)).fillna(0).gt(0).mean() * 100),
                        1,
                    ),
                }
            ]
        )
        table_paths.append(
            _write_table(rentsmart_summary, tables_dir / "rentsmart_risk_summary.csv")
        )

    return table_paths, figure_paths


def generate_student_housing_outputs(
    student_context_path: Path,
    tables_dir: Path,
    figures_dir: Path,
) -> tuple[list[Path], list[Path]]:
    """Generate optional context outputs if student housing data exists."""
    if not student_context_path.exists():
        print("Skipping student housing EDA: student housing context is unavailable.")
        return [], []

    context_df = pd.read_csv(student_context_path)
    table_paths: list[Path] = []
    figure_paths: list[Path] = []

    zip_col = next((column for column in ["zip", "zip_code", "postal_code"] if column in context_df.columns), None)
    student_metric_col = next(
        (
            column
            for column in context_df.columns
            if any(token in column for token in ["student", "bed", "unit", "occup"])
            and pd.api.types.is_numeric_dtype(context_df[column])
        ),
        None,
    )
    if zip_col is not None and student_metric_col is not None and "total_violations" in context_df.columns:
        summary = (
            context_df.groupby(zip_col)
            .agg(
                total_violations=("total_violations", "sum"),
                student_housing_metric=(student_metric_col, "sum"),
            )
            .reset_index()
            .sort_values("student_housing_metric", ascending=False)
            .head(15)
        )
        table_paths.append(
            _write_table(summary, tables_dir / "student_housing_zip_context.csv")
        )
        plot_df = summary.sort_values("student_housing_metric", ascending=True)
        figure_paths.append(
            _save_barh_figure(
                plot_df[zip_col],
                plot_df["total_violations"],
                "Violations in ZIP Codes With Student Housing Context",
                "Violation Count",
                "ZIP Code",
                figures_dir / "student_housing_zip_context.png",
            )
        )

    return table_paths, figure_paths


def run_eda(config: EDAConfig) -> tuple[list[Path], list[Path]]:
    """Generate summary tables and exploratory figures, plus optional context layers."""
    table_paths = generate_eda_tables(
        Phase2EDASummaryConfig(input_path=config.input_path, output_dir=config.tables_dir)
    )
    figure_paths = generate_phase2_figures(
        Phase2VisualizationConfig(input_path=config.input_path, output_dir=config.figures_dir)
    )
    for table_group, figure_group in [
        generate_property_risk_outputs(
            config.property_risk_path,
            config.tables_dir,
            config.figures_dir,
        ),
        generate_student_housing_outputs(
            config.student_context_path,
            config.tables_dir,
            config.figures_dir,
        ),
    ]:
        table_paths.extend(table_group)
        figure_paths.extend(figure_group)
    return table_paths, figure_paths


def main() -> None:
    run_eda(EDAConfig())


__all__ = [
    "EDAConfig",
    "Phase2EDASummaryConfig",
    "generate_eda_tables",
    "generate_phase2_figures",
    "generate_property_risk_outputs",
    "generate_student_housing_outputs",
    "get_available_date_column",
    "run_eda",
    "summarize_figure_results",
]


if __name__ == "__main__":
    main()
