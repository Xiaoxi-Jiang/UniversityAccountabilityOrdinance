"""Interactive Plotly visualizations generated from project outputs."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs

from src.data.features import (
    Phase2FeatureConfig,
    VIOLATION_TYPE_CANDIDATES,
    first_available_column,
    load_phase2_source_data,
    normalize_zip,
    prepare_violations_frame,
)
from src.viz.choropleth import BOSTON_ZIP_BOUNDARY_CACHE_PATH
from src.viz.phase2_visualizations import _derive_severity_proxy, get_available_date_column


@dataclass(frozen=True)
class InteractiveVisualizationConfig:
    input_path: Path = Phase2FeatureConfig().input_path
    tables_dir: Path = Path("outputs/tables")
    output_dir: Path = Path("outputs/interactive")
    property_risk_path: Path = Path("data/processed/property_risk_table_v1.csv")
    student_context_path: Path = Path("data/processed/student_housing_summary_v1.csv")
    improved_model_results_path: Path = Path("outputs/tables/improved_model_results.csv")
    zip_boundary_path: Path = BOSTON_ZIP_BOUNDARY_CACHE_PATH


def _write_html(fig: go.Figure, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.update_layout(
        template="plotly_white",
        margin={"l": 70, "r": 30, "t": 80, "b": 70},
        font={"family": "Arial, sans-serif", "size": 13},
        hoverlabel={"font_size": 12},
    )
    pio.write_html(
        fig,
        file=output_path,
        include_plotlyjs="directory",
        full_html=True,
        config={
            "displaylogo": False,
            "responsive": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
    )
    print(f"Saved interactive figure: {output_path}")
    return output_path


def _ensure_plotly_bundle(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = output_dir / "plotly.min.js"
    if not bundle_path.exists():
        bundle_path.write_text(get_plotlyjs(), encoding="utf-8")
    return bundle_path


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator_values = pd.to_numeric(numerator, errors="coerce")
    denominator_values = pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)
    return numerator_values / denominator_values


def _safe_corr(left: pd.Series, right: pd.Series) -> float:
    aligned = pd.concat([left, right], axis=1).dropna()
    if aligned.empty or aligned.iloc[:, 0].nunique() < 2 or aligned.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))


def _first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    return next((column for column in candidates if column in df.columns), None)


def _append_if_created(paths: list[Path], func) -> None:
    try:
        output_path = func()
    except Exception as exc:
        print(f"Skipping interactive figure: {exc}")
        return
    if output_path is not None:
        paths.append(output_path)


def _prepared_violations(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing cleaned violations dataset at {input_path}. Run `make prepare-data` first."
        )
    df = load_phase2_source_data(Phase2FeatureConfig(input_path=input_path))
    prepared, _ = prepare_violations_frame(df)
    return prepared


def _severity_distribution_figure(df: pd.DataFrame, output_dir: Path) -> Path:
    severity = _derive_severity_proxy(df)
    if severity is None:
        raise ValueError("Cannot generate severity distribution without severity or violation text columns.")

    ordered_labels = [
        "high risk (proxy)",
        "medium risk (proxy)",
        "low risk (proxy)",
        "uncategorized",
        "unknown",
    ]
    counts = severity.fillna("unknown").value_counts()
    counts = counts.reindex([label for label in ordered_labels if label in counts.index], fill_value=0)
    plot_df = counts.rename_axis("severity").reset_index(name="violation_count")
    plot_df["share_pct"] = plot_df["violation_count"] / max(plot_df["violation_count"].sum(), 1) * 100

    fig = px.bar(
        plot_df,
        x="severity",
        y="violation_count",
        color="severity",
        text="violation_count",
        color_discrete_map={
            "high risk (proxy)": "#dc2626",
            "medium risk (proxy)": "#f59e0b",
            "low risk (proxy)": "#2563eb",
            "uncategorized": "#9ca3af",
            "unknown": "#6b7280",
        },
        labels={"severity": "Severity Category", "violation_count": "Violation Count"},
        hover_data={"share_pct": ":.1f"},
        title="Violation Severity Distribution (Proxy-Based Labels)",
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(showlegend=False)
    return _write_html(fig, output_dir / "severity_distribution.html")


def _status_distribution_figure(df: pd.DataFrame, output_dir: Path) -> Path:
    if "status" not in df.columns:
        raise ValueError("Cannot generate status distribution without a status column.")

    counts = df["status"].astype("string").fillna("unknown").value_counts()
    plot_df = counts.rename_axis("status").reset_index(name="violation_count")
    plot_df["share_pct"] = plot_df["violation_count"] / max(plot_df["violation_count"].sum(), 1) * 100
    fig = px.bar(
        plot_df,
        x="status",
        y="violation_count",
        color="status",
        text="violation_count",
        labels={"status": "Status", "violation_count": "Violation Count"},
        hover_data={"share_pct": ":.1f"},
        title="Violation Status Distribution",
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(showlegend=False)
    return _write_html(fig, output_dir / "status_distribution.html")


def _top_violation_types_figure(df: pd.DataFrame, output_dir: Path) -> Path:
    violation_type_col = first_available_column(df, VIOLATION_TYPE_CANDIDATES)
    if violation_type_col is None:
        raise ValueError("Cannot generate top violation types without a type or description column.")

    counts = (
        df[violation_type_col]
        .astype("string")
        .replace("", pd.NA)
        .dropna()
        .value_counts()
        .head(15)
        .sort_values(ascending=True)
    )
    if counts.empty:
        raise ValueError("Cannot generate top violation types because no non-empty values were found.")

    plot_df = counts.rename_axis("violation_type").reset_index(name="violation_count")
    fig = px.bar(
        plot_df,
        x="violation_count",
        y="violation_type",
        orientation="h",
        text="violation_count",
        labels={"violation_count": "Violation Count", "violation_type": "Violation Type"},
        title="Top Violation Types by Count",
    )
    fig.update_traces(textposition="outside", cliponaxis=False, marker_color="#2563eb")
    fig.update_layout(yaxis={"categoryorder": "array", "categoryarray": plot_df["violation_type"].tolist()})
    return _write_html(fig, output_dir / "top_violation_types.html")


def _violations_over_time_figure(df: pd.DataFrame, output_dir: Path) -> Path:
    date_col = get_available_date_column(df)
    if date_col is None:
        raise ValueError("Cannot generate time trend without a usable date column.")

    dated = df.dropna(subset=[date_col]).copy()
    if dated.empty:
        raise ValueError("Cannot generate time trend without dated rows.")

    monthly = (
        dated.assign(period=dated[date_col].dt.to_period("M").dt.to_timestamp())
        .groupby("period")
        .size()
        .sort_index()
        .rename("monthly_count")
        .to_frame()
    )
    monthly["rolling_12m_avg"] = monthly["monthly_count"].rolling(window=12, min_periods=3).mean()
    monthly = monthly.reset_index()

    date_label = "status timestamp" if date_col == "status_dttm" else date_col
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=monthly["period"],
            y=monthly["monthly_count"],
            mode="lines",
            name="Monthly count",
            line={"color": "#93c5fd", "width": 1.5},
            hovertemplate="%{x|%Y-%m}<br>Violations=%{y}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=monthly["period"],
            y=monthly["rolling_12m_avg"],
            mode="lines",
            name="12-month rolling average",
            line={"color": "#1d4ed8", "width": 3},
            hovertemplate="%{x|%Y-%m}<br>12-month avg=%{y:.1f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Violations Over Time ({date_label}; monthly with rolling average)",
        xaxis_title="Month",
        yaxis_title="Violation Count",
        hovermode="x unified",
    )
    return _write_html(fig, output_dir / "violations_over_time.html")


def generate_interactive_phase2_figures(config: InteractiveVisualizationConfig) -> list[Path]:
    """Generate interactive equivalents of the core violation EDA figures."""
    prepared = _prepared_violations(config.input_path)
    paths: list[Path] = []
    for figure_func in [
        lambda: _severity_distribution_figure(prepared.copy(), config.output_dir),
        lambda: _status_distribution_figure(prepared.copy(), config.output_dir),
        lambda: _top_violation_types_figure(prepared.copy(), config.output_dir),
        lambda: _violations_over_time_figure(prepared.copy(), config.output_dir),
    ]:
        _append_if_created(paths, figure_func)
    return paths


def _top_repeated_properties_figure(risk_df: pd.DataFrame, output_dir: Path) -> Path:
    if not {"property_key", "total_violations"}.issubset(risk_df.columns):
        raise ValueError("Cannot generate repeated-property chart without property_key and total_violations.")

    repeated = risk_df.loc[risk_df["total_violations"] >= 2].copy()
    if repeated.empty:
        raise ValueError("No repeated properties are available for plotting.")

    owner_col = _first_existing_column(repeated, ["assessment_owner_clean", "assessment_owner"])
    class_col = _first_existing_column(repeated, ["assessment_lu_desc", "assessment_bldg_type"])
    hover_cols = [
        column
        for column in ["violation_st", "violation_zip", owner_col, class_col]
        if column is not None and column in repeated.columns
    ]
    plot_df = repeated.sort_values("total_violations", ascending=False).head(20)
    plot_df = plot_df.sort_values("total_violations", ascending=True)

    fig = px.bar(
        plot_df,
        x="total_violations",
        y="property_key",
        orientation="h",
        color=class_col if class_col in plot_df.columns else None,
        text="total_violations",
        hover_data=hover_cols,
        labels={"total_violations": "Violation Count", "property_key": "Property"},
        title="Top Repeated Properties With Ownership Context",
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    return _write_html(fig, output_dir / "top_repeated_properties_with_owner.html")


def _violations_by_property_class_figure(risk_df: pd.DataFrame, output_dir: Path) -> Path:
    class_col = _first_existing_column(risk_df, ["assessment_lu_desc", "assessment_bldg_type"])
    if class_col is None or "total_violations" not in risk_df.columns:
        raise ValueError("Cannot generate property-class chart without property class and violation columns.")

    class_summary = (
        risk_df.dropna(subset=[class_col])
        .groupby(class_col)
        .agg(
            property_count=("property_key", "nunique"),
            total_violations=("total_violations", "sum"),
            open_violations=("open_violations", "sum") if "open_violations" in risk_df.columns else ("total_violations", "size"),
        )
        .reset_index()
    )
    class_summary["violations_per_property"] = _safe_ratio(
        class_summary["total_violations"],
        class_summary["property_count"],
    )
    class_summary = class_summary.loc[class_summary["property_count"] >= 25].copy()
    class_summary = class_summary.sort_values("violations_per_property", ascending=False).head(15)
    if class_summary.empty:
        raise ValueError("No property classes with at least 25 properties are available.")

    plot_df = class_summary.sort_values("violations_per_property", ascending=True)
    plot_df["class_label"] = (
        plot_df[class_col].astype("string")
        + " (n="
        + plot_df["property_count"].fillna(0).astype(int).astype("string")
        + ")"
    )
    fig = px.bar(
        plot_df,
        x="violations_per_property",
        y="class_label",
        orientation="h",
        text="violations_per_property",
        hover_data={
            "property_count": True,
            "total_violations": True,
            "open_violations": True,
            "violations_per_property": ":.2f",
        },
        labels={"violations_per_property": "Violations per Property", "class_label": "Property Class"},
        title="Violations per Property by Class (Classes With >=25 Properties)",
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside", cliponaxis=False, marker_color="#2563eb")
    return _write_html(fig, output_dir / "violations_by_property_class.html")


def generate_interactive_property_risk_outputs(config: InteractiveVisualizationConfig) -> list[Path]:
    """Generate interactive property-risk figures when the property-risk table exists."""
    if not config.property_risk_path.exists():
        print(f"Skipping interactive property-risk figures: missing {config.property_risk_path}.")
        return []
    risk_df = pd.read_csv(config.property_risk_path, low_memory=False)
    paths: list[Path] = []
    for figure_func in [
        lambda: _top_repeated_properties_figure(risk_df.copy(), config.output_dir),
        lambda: _violations_by_property_class_figure(risk_df.copy(), config.output_dir),
    ]:
        _append_if_created(paths, figure_func)
    return paths


def _student_relationship_axes(df: pd.DataFrame) -> tuple[str, str, str, str]:
    y_col = "violations_per_property" if "violations_per_property" in df.columns else "total_violations"
    y_label = "Violations per Property" if y_col == "violations_per_property" else "Total Violations"
    x_col = "students_per_property" if "students_per_property" in df.columns else "student_housing_metric"
    x_label = "Students per Property" if x_col == "students_per_property" else "All Students"
    return x_col, y_col, x_label, y_label


def _student_summary_from_context(context_path: Path) -> pd.DataFrame:
    if not context_path.exists():
        raise FileNotFoundError(f"Missing student housing context at {context_path}.")

    context_df = pd.read_csv(context_path, low_memory=False)
    zip_col = _first_existing_column(context_df, ["zip", "zip_code", "postal_code"])
    student_metric_col = _first_existing_column(
        context_df,
        ["all_students", "student_housing_metric", "total_students", "student_beds", "beds", "student_units"],
    )
    if zip_col is None or student_metric_col is None or "total_violations" not in context_df.columns:
        raise ValueError("Student housing context is missing ZIP, student metric, or violation columns.")

    aggregation: dict[str, Any] = {
        "total_violations": ("total_violations", "sum"),
        "student_housing_metric": (student_metric_col, "sum"),
    }
    if "open_violations" in context_df.columns:
        aggregation["open_violations"] = ("open_violations", "sum")
    if "property_count" in context_df.columns:
        aggregation["property_count"] = ("property_count", "sum")

    summary = context_df.groupby(zip_col).agg(**aggregation).reset_index().rename(columns={zip_col: "zip"})
    summary["zip"] = summary["zip"].map(normalize_zip).astype("string")
    summary = summary.dropna(subset=["zip"])
    if "property_count" in summary.columns:
        summary["violations_per_property"] = _safe_ratio(summary["total_violations"], summary["property_count"])
        summary["students_per_property"] = _safe_ratio(summary["student_housing_metric"], summary["property_count"])
    if "open_violations" in summary.columns:
        summary["open_violation_share"] = _safe_ratio(summary["open_violations"], summary["total_violations"])
    summary["violations_per_1000_students"] = _safe_ratio(
        summary["total_violations"],
        summary["student_housing_metric"],
    ) * 1000
    return summary.loc[summary["student_housing_metric"].notna() & summary["student_housing_metric"].gt(0)].copy()


def _load_student_relationship_table(config: InteractiveVisualizationConfig) -> pd.DataFrame:
    relationship_path = config.tables_dir / "student_housing_relationship.csv"
    if relationship_path.exists():
        return pd.read_csv(relationship_path, low_memory=False)
    return _student_summary_from_context(config.student_context_path)


def _student_relationship_figure(summary_df: pd.DataFrame, output_dir: Path) -> Path:
    zip_col = _first_existing_column(summary_df, ["zip", "zip_code", "postal_code"])
    if zip_col is None:
        raise ValueError("Cannot generate student relationship chart without ZIP values.")
    x_col, y_col, x_label, y_label = _student_relationship_axes(summary_df)
    plot_df = summary_df.dropna(subset=[x_col, y_col]).copy()
    if plot_df.empty:
        raise ValueError("No matched ZIP rows are available for student relationship plotting.")

    corr = _safe_corr(plot_df[x_col], plot_df[y_col])
    color_col = "violations_per_1000_students" if "violations_per_1000_students" in plot_df.columns else None
    hover_cols = [
        column
        for column in [
            "student_housing_metric",
            "total_violations",
            "property_count",
            "open_violation_share",
            "violations_per_1000_students",
        ]
        if column in plot_df.columns
    ]
    fig = px.scatter(
        plot_df,
        x=x_col,
        y=y_col,
        text=zip_col,
        size="total_violations" if "total_violations" in plot_df.columns else None,
        color=color_col,
        color_continuous_scale="Viridis" if color_col else None,
        hover_name=zip_col,
        hover_data=hover_cols,
        labels={x_col: x_label, y_col: y_label, color_col or "": "Violations per 1,000 Students"},
        title="Student Density vs Violation Intensity by ZIP",
    )
    fig.update_traces(textposition="top center", marker={"line": {"width": 0.8, "color": "white"}})
    if plot_df[x_col].nunique() >= 2:
        x_values = plot_df[x_col].astype(float).to_numpy()
        y_values = plot_df[y_col].astype(float).to_numpy()
        slope, intercept = np.polyfit(x_values, y_values, 1)
        x_line = np.linspace(x_values.min(), x_values.max(), 100)
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=slope * x_line + intercept,
                mode="lines",
                name="Linear fit",
                line={"color": "#1d4ed8", "dash": "dash"},
                hoverinfo="skip",
            )
        )
    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        text=f"n={len(plot_df)}<br>Pearson r={corr:.3f}",
        showarrow=False,
        align="left",
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#cbd5e1",
        borderwidth=1,
    )
    return _write_html(fig, output_dir / "student_housing_relationship.html")


def _student_ranked_zip_figure(summary_df: pd.DataFrame, output_dir: Path) -> Path:
    zip_col = _first_existing_column(summary_df, ["zip", "zip_code", "postal_code"])
    if zip_col is None:
        raise ValueError("Cannot generate student ZIP ranking without ZIP values.")
    _, y_col, _, y_label = _student_relationship_axes(summary_df)
    plot_df = summary_df.dropna(subset=[zip_col, y_col]).sort_values(y_col, ascending=False).head(10)
    if plot_df.empty:
        raise ValueError("No matched ZIP rows are available for student ZIP ranking.")
    plot_df = plot_df.sort_values(y_col, ascending=True)
    fig = px.bar(
        plot_df,
        x=y_col,
        y=zip_col,
        orientation="h",
        text=y_col,
        hover_data=[column for column in ["student_housing_metric", "property_count", "total_violations"] if column in plot_df.columns],
        labels={y_col: y_label, zip_col: "ZIP Code"},
        title="Highest ZIP Violation Intensity With Student Housing Context",
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside", cliponaxis=False, marker_color="#2563eb")
    return _write_html(fig, output_dir / "student_housing_violation_intensity_by_zip.html")


def _student_zip_choropleth(summary_df: pd.DataFrame, config: InteractiveVisualizationConfig) -> Path:
    if not config.zip_boundary_path.exists():
        raise FileNotFoundError(f"Missing ZIP boundary GeoJSON at {config.zip_boundary_path}.")
    zip_col = _first_existing_column(summary_df, ["zip", "zip_code", "postal_code"])
    if zip_col is None:
        raise ValueError("Cannot generate ZIP map without ZIP values.")

    value_col = "violations_per_property" if "violations_per_property" in summary_df.columns else "total_violations"
    working = summary_df.dropna(subset=[zip_col, value_col]).copy()
    working[zip_col] = working[zip_col].map(normalize_zip).astype("string")
    if working.empty:
        raise ValueError("Cannot generate ZIP map without matched ZIP values.")

    boundary_geojson = json.loads(config.zip_boundary_path.read_text())
    hover_data = [
        column
        for column in [
            "student_housing_metric",
            "total_violations",
            "property_count",
            "students_per_property",
            "violations_per_1000_students",
        ]
        if column in working.columns
    ]
    fig = px.choropleth(
        working,
        geojson=boundary_geojson,
        locations=zip_col,
        color=value_col,
        featureidkey="properties.ZIP5",
        hover_name=zip_col,
        hover_data=hover_data,
        color_continuous_scale="YlOrRd",
        labels={
            value_col: "Violations per Property" if value_col == "violations_per_property" else "Violation Count",
            zip_col: "ZIP Code",
        },
        title="Boston ZIP Violation Intensity (Student Housing Context Available)",
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(height=760)
    return _write_html(fig, config.output_dir / "student_housing_zip_context.html")


def generate_interactive_student_housing_outputs(config: InteractiveVisualizationConfig) -> list[Path]:
    """Generate interactive student-housing relationship and ZIP figures."""
    try:
        summary_df = _load_student_relationship_table(config)
    except Exception as exc:
        print(f"Skipping interactive student-housing figures: {exc}")
        return []

    paths: list[Path] = []
    for figure_func in [
        lambda: _student_relationship_figure(summary_df.copy(), config.output_dir),
        lambda: _student_ranked_zip_figure(summary_df.copy(), config.output_dir),
        lambda: _student_zip_choropleth(summary_df.copy(), config),
    ]:
        _append_if_created(paths, figure_func)
    return paths


def generate_interactive_model_outputs(config: InteractiveVisualizationConfig) -> list[Path]:
    """Generate an interactive model-performance comparison from improved model outputs."""
    if not config.improved_model_results_path.exists():
        print(f"Skipping interactive model comparison: missing {config.improved_model_results_path}.")
        return []

    results = pd.read_csv(config.improved_model_results_path)
    metric_cols = [
        column
        for column in ["balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
        if column in results.columns
    ]
    required_cols = {"model_name", "feature_set"}
    if not metric_cols or not required_cols.issubset(results.columns):
        print("Skipping interactive model comparison: improved model results are missing required columns.")
        return []

    label_col = "target_label" if "target_label" in results.columns else "target"
    long_df = results.melt(
        id_vars=[column for column in [label_col, "feature_set", "model_name", "n_positive", "positive_class_rate", "cv_folds"] if column in results.columns],
        value_vars=metric_cols,
        var_name="metric",
        value_name="score",
    )
    fig = px.bar(
        long_df,
        x="model_name",
        y="score",
        color="feature_set",
        facet_col=label_col,
        facet_row="metric",
        barmode="group",
        hover_data={
            "score": ":.4f",
            "n_positive": True if "n_positive" in long_df.columns else False,
            "positive_class_rate": ":.4f" if "positive_class_rate" in long_df.columns else False,
            "cv_folds": True if "cv_folds" in long_df.columns else False,
        },
        labels={
            "model_name": "Model",
            "score": "Score",
            "feature_set": "Feature Set",
            label_col: "Target",
            "metric": "Metric",
        },
        title="Cross-Validated Model Performance Comparison",
    )
    fig.update_yaxes(matches=None)
    fig.for_each_annotation(lambda annotation: annotation.update(text=annotation.text.replace("metric=", "").replace(f"{label_col}=", "")))
    fig.update_layout(height=max(780, 230 * len(metric_cols)))
    return [_write_html(fig, config.output_dir / "model_performance_comparison.html")]


def _records_for_dashboard(
    df: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    working = df.copy()
    if columns is not None:
        available = [column for column in columns if column in working.columns]
        working = working.loc[:, available]
    if limit is not None:
        working = working.head(limit)
    return json.loads(working.to_json(orient="records", date_format="iso"))


def _dashboard_violations_data(config: InteractiveVisualizationConfig) -> dict[str, Any]:
    prepared = _prepared_violations(config.input_path)
    date_col = get_available_date_column(prepared)
    severity = _derive_severity_proxy(prepared)

    kpis: dict[str, Any] = {
        "violation_records": int(len(prepared)),
        "unique_properties": int(prepared["property_key"].nunique()) if "property_key" in prepared.columns else None,
        "open_records": int(prepared["status"].astype("string").str.lower().eq("open").sum()) if "status" in prepared.columns else None,
        "date_min": None,
        "date_max": None,
        "high_risk_proxy": int(severity.eq("high risk (proxy)").sum()) if severity is not None else None,
    }
    if date_col is not None:
        dated = prepared[date_col].dropna()
        if not dated.empty:
            kpis["date_min"] = dated.min().date().isoformat()
            kpis["date_max"] = dated.max().date().isoformat()

    severity_records: list[dict[str, Any]] = []
    if severity is not None:
        severity_order = [
            "high risk (proxy)",
            "medium risk (proxy)",
            "low risk (proxy)",
            "uncategorized",
            "unknown",
        ]
        severity_counts = (
            severity.fillna("unknown")
            .value_counts()
            .reindex(severity_order, fill_value=0)
            .loc[lambda values: values.gt(0)]
            .rename_axis("severity")
            .reset_index(name="violation_count")
        )
        severity_counts["share_pct"] = severity_counts["violation_count"] / max(int(severity_counts["violation_count"].sum()), 1) * 100
        severity_records = _records_for_dashboard(severity_counts)

    status_records: list[dict[str, Any]] = []
    if "status" in prepared.columns:
        status_counts = (
            prepared["status"]
            .astype("string")
            .fillna("unknown")
            .value_counts()
            .rename_axis("status")
            .reset_index(name="violation_count")
        )
        status_counts["share_pct"] = status_counts["violation_count"] / max(int(status_counts["violation_count"].sum()), 1) * 100
        status_records = _records_for_dashboard(status_counts)

    type_records: list[dict[str, Any]] = []
    violation_type_col = first_available_column(prepared, VIOLATION_TYPE_CANDIDATES)
    if violation_type_col is not None:
        type_counts = (
            prepared[violation_type_col]
            .astype("string")
            .replace("", pd.NA)
            .dropna()
            .value_counts()
            .head(12)
            .rename_axis("violation_type")
            .reset_index(name="violation_count")
        )
        type_records = _records_for_dashboard(type_counts)

    monthly_records: list[dict[str, Any]] = []
    if date_col is not None:
        dated = prepared.dropna(subset=[date_col]).copy()
        if not dated.empty:
            monthly = (
                dated.assign(period=dated[date_col].dt.to_period("M").dt.to_timestamp())
                .groupby("period")
                .size()
                .sort_index()
                .rename("monthly_count")
                .to_frame()
            )
            monthly["rolling_12m_avg"] = monthly["monthly_count"].rolling(window=12, min_periods=3).mean()
            monthly_records = _records_for_dashboard(monthly.reset_index())

    return {
        "kpis": kpis,
        "date_column": date_col,
        "severity": severity_records,
        "status": status_records,
        "violationTypes": type_records,
        "monthly": monthly_records,
    }


def _dashboard_property_data(config: InteractiveVisualizationConfig) -> dict[str, Any]:
    if not config.property_risk_path.exists():
        return {"available": False, "repeated": [], "classes": [], "ownerCoveragePct": None}

    risk_df = pd.read_csv(config.property_risk_path, low_memory=False)
    owner_col = _first_existing_column(risk_df, ["assessment_owner_clean", "assessment_owner"])
    class_col = _first_existing_column(risk_df, ["assessment_lu_desc", "assessment_bldg_type"])
    owner_coverage_pct = None
    if "owner_data_available_flag" in risk_df.columns:
        owner_coverage_pct = round(float(risk_df["owner_data_available_flag"].fillna(0).mean() * 100), 1)
    elif owner_col is not None:
        owner_coverage_pct = round(float(risk_df[owner_col].astype("string").str.strip().ne("").mean() * 100), 1)

    repeated_records: list[dict[str, Any]] = []
    if {"property_key", "total_violations"}.issubset(risk_df.columns):
        repeated = risk_df.loc[risk_df["total_violations"] >= 2].copy()
        if not repeated.empty:
            repeated_columns = [
                column
                for column in [
                    "property_key",
                    "violation_st",
                    "violation_zip",
                    "total_violations",
                    "open_violations",
                    owner_col,
                    class_col,
                ]
                if column is not None and column in repeated.columns
            ]
            repeated = repeated.sort_values("total_violations", ascending=False).head(30)
            repeated_records = _records_for_dashboard(repeated, columns=repeated_columns)

    class_records: list[dict[str, Any]] = []
    if class_col is not None and "total_violations" in risk_df.columns:
        open_agg = ("open_violations", "sum") if "open_violations" in risk_df.columns else ("total_violations", "size")
        class_summary = (
            risk_df.dropna(subset=[class_col])
            .groupby(class_col)
            .agg(
                property_count=("property_key", "nunique"),
                total_violations=("total_violations", "sum"),
                open_violations=open_agg,
            )
            .reset_index()
        )
        class_summary["property_class"] = class_summary[class_col].astype("string")
        class_summary["violations_per_property"] = _safe_ratio(
            class_summary["total_violations"],
            class_summary["property_count"],
        )
        class_summary["open_violations_per_property"] = _safe_ratio(
            class_summary["open_violations"],
            class_summary["property_count"],
        )
        class_summary = class_summary.loc[class_summary["property_count"] >= 25].copy()
        class_summary = class_summary.sort_values("violations_per_property", ascending=False).head(20)
        class_records = _records_for_dashboard(
            class_summary,
            columns=[
                "property_class",
                "property_count",
                "total_violations",
                "open_violations",
                "violations_per_property",
                "open_violations_per_property",
            ],
        )

    return {
        "available": True,
        "ownerCoveragePct": owner_coverage_pct,
        "repeated": repeated_records,
        "classes": class_records,
    }


def _dashboard_student_data(config: InteractiveVisualizationConfig) -> dict[str, Any]:
    try:
        student_df = _load_student_relationship_table(config)
    except Exception:
        return {"available": False, "records": [], "geojson": None}

    zip_col = _first_existing_column(student_df, ["zip", "zip_code", "postal_code"])
    if zip_col is not None and zip_col != "zip":
        student_df = student_df.rename(columns={zip_col: "zip"})
    if "zip" in student_df.columns:
        student_df["zip"] = student_df["zip"].map(normalize_zip).astype("string")

    records = _records_for_dashboard(student_df)
    geojson = None
    if config.zip_boundary_path.exists():
        geojson = json.loads(config.zip_boundary_path.read_text())

    metric_options = [
        {
            "id": column,
            "label": label,
        }
        for column, label in [
            ("students_per_property", "Students per Property"),
            ("student_housing_metric", "All Students"),
            ("violations_per_property", "Violations per Property"),
            ("violations_per_1000_students", "Violations per 1,000 Students"),
            ("total_violations", "Total Violations"),
            ("open_violation_share", "Open Violation Share"),
        ]
        if column in student_df.columns
    ]

    return {
        "available": True,
        "records": records,
        "geojson": geojson,
        "metrics": metric_options,
    }


def _dashboard_model_data(config: InteractiveVisualizationConfig) -> dict[str, Any]:
    if not config.improved_model_results_path.exists():
        return {"available": False, "records": [], "featureImportance": [], "metrics": []}

    results = pd.read_csv(config.improved_model_results_path)
    metric_cols = [
        column
        for column in ["balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
        if column in results.columns
    ]
    records = _records_for_dashboard(results)

    feature_importance_path = config.tables_dir / "improved_model_feature_importance.csv"
    feature_records: list[dict[str, Any]] = []
    if feature_importance_path.exists():
        feature_df = pd.read_csv(feature_importance_path)
        if "abs_importance" not in feature_df.columns and "importance" in feature_df.columns:
            feature_df["abs_importance"] = pd.to_numeric(feature_df["importance"], errors="coerce").abs()
        feature_records = _records_for_dashboard(feature_df)

    metric_labels = {
        "balanced_accuracy": "Balanced Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "roc_auc": "ROC-AUC",
        "pr_auc": "PR-AUC",
    }
    return {
        "available": True,
        "records": records,
        "featureImportance": feature_records,
        "metrics": [{"id": metric, "label": metric_labels.get(metric, metric)} for metric in metric_cols],
    }


def _build_dashboard_data(
    config: InteractiveVisualizationConfig,
    paths: list[Path],
) -> dict[str, Any]:
    artifact_paths = [
        {"name": path.name, "label": path.stem.replace("_", " ").title()}
        for path in paths
        if path.suffix.lower() == ".html" and path.name != "index.html"
    ]
    return {
        "generatedAt": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "artifacts": artifact_paths,
        "violations": _dashboard_violations_data(config),
        "property": _dashboard_property_data(config),
        "student": _dashboard_student_data(config),
        "model": _dashboard_model_data(config),
    }


def _dashboard_html(data: dict[str, Any]) -> str:
    data_json = json.dumps(data, ensure_ascii=False)
    artifact_links = "\n".join(
        f'<a href="{escape(item["name"])}">{escape(item["label"])}</a>'
        for item in data.get("artifacts", [])
    )
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>University Accountability Interactive Dashboard</title>
    <script src="plotly.min.js"></script>
    <style>
      :root {{
        --ink: #172033;
        --muted: #617083;
        --line: #d8dee8;
        --panel: #ffffff;
        --band: #f6f8fb;
        --blue: #2563eb;
        --teal: #0f766e;
        --amber: #b45309;
        --red: #dc2626;
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        background: var(--band);
        color: var(--ink);
        font-family: Arial, Helvetica, sans-serif;
        letter-spacing: 0;
      }}
      header {{
        background: #ffffff;
        border-bottom: 1px solid var(--line);
        padding: 22px 28px 18px;
      }}
      .header-row {{
        display: flex;
        justify-content: space-between;
        gap: 18px;
        align-items: flex-start;
      }}
      h1 {{
        margin: 0;
        font-size: 26px;
        line-height: 1.2;
      }}
      .subtitle {{
        margin-top: 6px;
        color: var(--muted);
        max-width: 920px;
      }}
      .timestamp {{
        color: var(--muted);
        font-size: 13px;
        white-space: nowrap;
      }}
      .tabs {{
        display: flex;
        gap: 8px;
        padding: 12px 28px;
        border-bottom: 1px solid var(--line);
        background: #ffffff;
        position: sticky;
        top: 0;
        z-index: 10;
      }}
      .tab-button {{
        border: 1px solid var(--line);
        background: #ffffff;
        color: var(--ink);
        border-radius: 8px;
        padding: 8px 12px;
        cursor: pointer;
        font: inherit;
      }}
      .tab-button.active {{
        background: var(--ink);
        border-color: var(--ink);
        color: #ffffff;
      }}
      main {{
        padding: 22px 28px 32px;
      }}
      .tab-panel {{ display: none; }}
      .tab-panel.active {{ display: block; }}
      .kpi-grid {{
        display: grid;
        grid-template-columns: repeat(5, minmax(140px, 1fr));
        gap: 12px;
        margin-bottom: 16px;
      }}
      .kpi {{
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 14px 14px 12px;
      }}
      .kpi-label {{
        color: var(--muted);
        font-size: 12px;
        text-transform: uppercase;
      }}
      .kpi-value {{
        margin-top: 6px;
        font-size: 24px;
        font-weight: 700;
      }}
      #kpi-dates {{
        font-size: 19px;
        line-height: 1.25;
      }}
      .grid {{
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 16px;
      }}
      .grid.one {{ grid-template-columns: 1fr; }}
      .panel {{
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        min-height: 380px;
        padding: 12px;
      }}
      .panel.tall {{ min-height: 560px; }}
      .section-head {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 16px;
        margin: 0 0 12px;
      }}
      h2 {{
        margin: 0;
        font-size: 19px;
      }}
      .controls {{
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        align-items: end;
        margin-bottom: 14px;
      }}
      label {{
        display: grid;
        gap: 4px;
        color: var(--muted);
        font-size: 12px;
      }}
      select {{
        min-width: 190px;
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 8px 10px;
        background: #ffffff;
        color: var(--ink);
        font: inherit;
      }}
      .plot {{
        width: 100%;
        min-height: 340px;
      }}
      .plot.tall {{ min-height: 520px; }}
      .note {{
        color: var(--muted);
        font-size: 13px;
        margin: 0 0 12px;
      }}
      .artifact-links {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
      }}
      .artifact-links a {{
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 7px 10px;
        background: #ffffff;
        color: var(--blue);
        text-decoration: none;
        font-size: 13px;
      }}
      .empty {{
        color: var(--muted);
        padding: 28px;
      }}
      @media (max-width: 980px) {{
        .header-row {{ display: block; }}
        .timestamp {{ margin-top: 8px; }}
        .kpi-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
        .grid {{ grid-template-columns: 1fr; }}
        .tabs {{ overflow-x: auto; }}
      }}
    </style>
  </head>
  <body>
    <header>
      <div class="header-row">
        <div>
          <h1>University Accountability Interactive Dashboard</h1>
          <div class="subtitle">Boston housing violation patterns, student-housing context, property risk, and model performance in one presentation-ready workspace.</div>
        </div>
        <div class="timestamp">Generated {escape(str(data.get("generatedAt", "")))}</div>
      </div>
    </header>
    <nav class="tabs" aria-label="Dashboard sections">
      <button class="tab-button active" data-tab="overview">Overview</button>
      <button class="tab-button" data-tab="student">Student ZIP Context</button>
      <button class="tab-button" data-tab="property">Property Risk</button>
      <button class="tab-button" data-tab="model">Model Performance</button>
      <button class="tab-button" data-tab="artifacts">HTML Exports</button>
    </nav>
    <main>
      <section id="overview" class="tab-panel active">
        <div class="kpi-grid">
          <div class="kpi"><div class="kpi-label">Violation Records</div><div class="kpi-value" id="kpi-records">-</div></div>
          <div class="kpi"><div class="kpi-label">Unique Properties</div><div class="kpi-value" id="kpi-properties">-</div></div>
          <div class="kpi"><div class="kpi-label">Open Records</div><div class="kpi-value" id="kpi-open">-</div></div>
          <div class="kpi"><div class="kpi-label">High-Risk Proxy</div><div class="kpi-value" id="kpi-highrisk">-</div></div>
          <div class="kpi"><div class="kpi-label">Date Range</div><div class="kpi-value" id="kpi-dates">-</div></div>
        </div>
        <div class="grid">
          <div class="panel"><div id="severityChart" class="plot"></div></div>
          <div class="panel"><div id="statusChart" class="plot"></div></div>
          <div class="panel tall"><div id="topTypesChart" class="plot tall"></div></div>
          <div class="panel tall"><div id="timeChart" class="plot tall"></div></div>
        </div>
      </section>
      <section id="student" class="tab-panel">
        <div class="section-head">
          <h2>Student Housing and Violation Intensity</h2>
        </div>
        <div class="controls">
          <label>X Metric<select id="studentX"></select></label>
          <label>Y Metric<select id="studentY"></select></label>
          <label>Color Metric<select id="studentColor"></select></label>
          <label>ZIP Ranking<select id="studentRank"></select></label>
        </div>
        <div class="grid">
          <div class="panel tall"><div id="studentScatter" class="plot tall"></div></div>
          <div class="panel tall"><div id="studentMap" class="plot tall"></div></div>
          <div class="panel"><div id="studentRankChart" class="plot"></div></div>
          <div class="panel"><p class="note" id="studentSummary"></p><div id="studentTable" class="plot"></div></div>
        </div>
      </section>
      <section id="property" class="tab-panel">
        <div class="section-head">
          <h2>Property Risk Context</h2>
          <p class="note" id="propertyNote"></p>
        </div>
        <div class="grid">
          <div class="panel tall"><div id="repeatedChart" class="plot tall"></div></div>
          <div class="panel tall"><div id="classChart" class="plot tall"></div></div>
        </div>
      </section>
      <section id="model" class="tab-panel">
        <div class="section-head">
          <h2>Cross-Validated Model Performance</h2>
        </div>
        <div class="controls">
          <label>Target<select id="modelTarget"></select></label>
          <label>Metric<select id="modelMetric"></select></label>
          <label>Feature Set<select id="featureSet"></select></label>
          <label>Model<select id="featureModel"></select></label>
        </div>
        <div class="grid">
          <div class="panel tall"><div id="modelMetricChart" class="plot tall"></div></div>
          <div class="panel tall"><div id="featureImportanceChart" class="plot tall"></div></div>
        </div>
      </section>
      <section id="artifacts" class="tab-panel">
        <div class="section-head"><h2>Generated HTML Exports</h2></div>
        <div class="artifact-links">
          {artifact_links}
        </div>
      </section>
    </main>
    <script>
      const DATA = {data_json};
      const COLORS = ["#2563eb", "#0f766e", "#b45309", "#dc2626", "#7c3aed", "#0891b2", "#4b5563"];
      const metricLabels = {{
        balanced_accuracy: "Balanced Accuracy",
        precision: "Precision",
        recall: "Recall",
        f1: "F1",
        roc_auc: "ROC-AUC",
        pr_auc: "PR-AUC",
        students_per_property: "Students per Property",
        student_housing_metric: "All Students",
        violations_per_property: "Violations per Property",
        violations_per_1000_students: "Violations per 1,000 Students",
        total_violations: "Total Violations",
        open_violation_share: "Open Violation Share"
      }};

      function pretty(value) {{
        if (value === null || value === undefined) return "-";
        const explicit = {{
          any_violation: "Any Violation",
          high_risk_violation: "High-Risk Violation",
          behavioral_only: "Behavioral Only",
          behavioral_plus_static: "Behavioral + Static",
          logistic_regression: "Logistic Regression",
          random_forest: "Random Forest",
          xgboost: "XGBoost"
        }};
        if (explicit[value]) return explicit[value];
        return String(value).replaceAll("_", " ").replace(/\\b\\w/g, c => c.toUpperCase());
      }}
      function fmt(value, digits = 0) {{
        if (value === null || value === undefined || Number.isNaN(value)) return "-";
        return Number(value).toLocaleString(undefined, {{ maximumFractionDigits: digits }});
      }}
      function labelFor(metric) {{ return metricLabels[metric] || metric; }}
      function rows(obj, key) {{ return (obj && obj[key]) || []; }}
      function setText(id, value) {{ const el = document.getElementById(id); if (el) el.textContent = value; }}
      function fillSelect(id, options, preferred) {{
        const el = document.getElementById(id);
        if (!el) return;
        el.innerHTML = "";
        options.forEach(opt => {{
          const option = document.createElement("option");
          option.value = opt.id ?? opt;
          option.textContent = opt.label ?? labelFor(opt);
          el.appendChild(option);
        }});
        const values = options.map(opt => opt.id ?? opt);
        if (preferred && values.includes(preferred)) el.value = preferred;
      }}
      function emptyPlot(id, message) {{
        Plotly.purge(id);
        document.getElementById(id).innerHTML = `<div class="empty">${{message}}</div>`;
      }}
      function plotLayout(title, xTitle, yTitle, extra = {{}}) {{
        return Object.assign({{
          title,
          xaxis: {{ title: xTitle, automargin: true }},
          yaxis: {{ title: yTitle, automargin: true }},
          margin: {{ l: 70, r: 26, t: 58, b: 64 }},
          paper_bgcolor: "#ffffff",
          plot_bgcolor: "#ffffff",
          font: {{ family: "Arial, sans-serif", size: 13, color: "#172033" }},
          colorway: COLORS,
          legend: {{ orientation: "h", y: -0.18 }}
        }}, extra);
      }}
      const plotConfig = {{ displaylogo: false, responsive: true }};

      function initKpis() {{
        const k = DATA.violations.kpis || {{}};
        setText("kpi-records", fmt(k.violation_records));
        setText("kpi-properties", fmt(k.unique_properties));
        setText("kpi-open", fmt(k.open_records));
        setText("kpi-highrisk", fmt(k.high_risk_proxy));
        setText("kpi-dates", k.date_min && k.date_max ? `${{k.date_min}} - ${{k.date_max}}` : "-");
      }}
      function drawOverview() {{
        const severity = rows(DATA.violations, "severity");
        if (severity.length) {{
          Plotly.react("severityChart", [{{
            type: "bar",
            x: severity.map(d => d.severity),
            y: severity.map(d => d.violation_count),
            text: severity.map(d => fmt(d.violation_count)),
            marker: {{ color: severity.map(d => d.severity.includes("high") ? "#dc2626" : d.severity.includes("medium") ? "#b45309" : d.severity.includes("low") ? "#2563eb" : "#94a3b8") }},
            hovertemplate: "%{{x}}<br>Count=%{{y:,}}<br>Share=%{{customdata:.1f}}%<extra></extra>",
            customdata: severity.map(d => d.share_pct)
          }}], plotLayout("Severity Distribution", "Severity Category", "Violation Count", {{ showlegend: false }}), plotConfig);
        }} else emptyPlot("severityChart", "Severity data unavailable.");

        const status = rows(DATA.violations, "status");
        if (status.length) {{
          Plotly.react("statusChart", [{{
            type: "pie",
            labels: status.map(d => d.status),
            values: status.map(d => d.violation_count),
            hole: 0.48,
            textinfo: "label+percent",
            hovertemplate: "%{{label}}<br>Count=%{{value:,}}<extra></extra>"
          }}], plotLayout("Status Distribution", "", "", {{ showlegend: true }}), plotConfig);
        }} else emptyPlot("statusChart", "Status data unavailable.");

        const types = rows(DATA.violations, "violationTypes").slice().reverse();
        if (types.length) {{
          Plotly.react("topTypesChart", [{{
            type: "bar",
            orientation: "h",
            x: types.map(d => d.violation_count),
            y: types.map(d => d.violation_type),
            text: types.map(d => fmt(d.violation_count)),
            marker: {{ color: "#2563eb" }},
            hovertemplate: "%{{y}}<br>Count=%{{x:,}}<extra></extra>"
          }}], plotLayout("Top Violation Types", "Violation Count", "Violation Type", {{ showlegend: false, margin: {{ l: 190, r: 26, t: 58, b: 64 }} }}), plotConfig);
        }} else emptyPlot("topTypesChart", "Violation type data unavailable.");

        const monthly = rows(DATA.violations, "monthly");
        if (monthly.length) {{
          Plotly.react("timeChart", [
            {{
              type: "scatter",
              mode: "lines",
              name: "Monthly count",
              x: monthly.map(d => d.period),
              y: monthly.map(d => d.monthly_count),
              line: {{ color: "#93c5fd", width: 1.5 }},
              hovertemplate: "%{{x|%Y-%m}}<br>Count=%{{y:,}}<extra></extra>"
            }},
            {{
              type: "scatter",
              mode: "lines",
              name: "12-month rolling average",
              x: monthly.map(d => d.period),
              y: monthly.map(d => d.rolling_12m_avg),
              line: {{ color: "#1d4ed8", width: 3 }},
              hovertemplate: "%{{x|%Y-%m}}<br>Rolling avg=%{{y:.1f}}<extra></extra>"
            }}
          ], plotLayout("Violations Over Time", "Month", "Violation Count", {{ hovermode: "x unified" }}), plotConfig);
        }} else emptyPlot("timeChart", "Time trend data unavailable.");
      }}

      function initStudentControls() {{
        const metricOptions = (DATA.student.metrics || []).map(m => ({{ id: m.id, label: m.label }}));
        fillSelect("studentX", metricOptions, "students_per_property");
        fillSelect("studentY", metricOptions, "violations_per_property");
        fillSelect("studentColor", metricOptions, "violations_per_1000_students");
        fillSelect("studentRank", metricOptions, "violations_per_property");
        ["studentX", "studentY", "studentColor", "studentRank"].forEach(id => {{
          const el = document.getElementById(id);
          if (el) el.addEventListener("change", drawStudent);
        }});
      }}
      function drawStudent() {{
        if (!DATA.student.available || !DATA.student.records.length) {{
          ["studentScatter", "studentMap", "studentRankChart", "studentTable"].forEach(id => emptyPlot(id, "Student housing context unavailable."));
          return;
        }}
        const dataRows = DATA.student.records;
        const x = document.getElementById("studentX").value;
        const y = document.getElementById("studentY").value;
        const color = document.getElementById("studentColor").value;
        const rank = document.getElementById("studentRank").value;
        const plotRows = dataRows.filter(d => d[x] !== null && d[y] !== null);
        const corr = (() => {{
          const xs = plotRows.map(d => Number(d[x]));
          const ys = plotRows.map(d => Number(d[y]));
          const n = xs.length;
          if (n < 2) return null;
          const mx = xs.reduce((a,b) => a + b, 0) / n;
          const my = ys.reduce((a,b) => a + b, 0) / n;
          const cov = xs.reduce((a, v, i) => a + (v - mx) * (ys[i] - my), 0);
          const sx = Math.sqrt(xs.reduce((a, v) => a + Math.pow(v - mx, 2), 0));
          const sy = Math.sqrt(ys.reduce((a, v) => a + Math.pow(v - my, 2), 0));
          return sx && sy ? cov / (sx * sy) : null;
        }})();
        Plotly.react("studentScatter", [{{
          type: "scatter",
          mode: "markers",
          x: plotRows.map(d => d[x]),
          y: plotRows.map(d => d[y]),
          marker: {{
            size: plotRows.map(d => Math.max(9, Math.sqrt(Number(d.total_violations || 1)) * 2.2)),
            color: plotRows.map(d => d[color]),
            colorscale: "Viridis",
            showscale: true,
            colorbar: {{ title: labelFor(color) }},
            line: {{ color: "#ffffff", width: 0.8 }}
          }},
          customdata: plotRows.map(d => [d.zip, d.total_violations, d.property_count, d.student_housing_metric]),
          hovertemplate: "ZIP=%{{customdata[0]}}<br>" + labelFor(x) + "=%{{x:.3f}}<br>" + labelFor(y) + "=%{{y:.3f}}<br>Total violations=%{{customdata[1]:,}}<br>Properties=%{{customdata[2]:,}}<br>Students=%{{customdata[3]:,}}<extra></extra>"
        }}], plotLayout(`Student Context: ${{labelFor(x)}} vs ${{labelFor(y)}}`, labelFor(x), labelFor(y), {{
          annotations: corr === null ? [] : [{{ x: 0.02, y: 0.98, xref: "paper", yref: "paper", text: `n=${{plotRows.length}}<br>Pearson r=${{corr.toFixed(3)}}`, showarrow: false, align: "left", bgcolor: "rgba(255,255,255,0.85)", bordercolor: "#d8dee8", borderwidth: 1 }}]
        }}), plotConfig);

        const ranked = dataRows.filter(d => d[rank] !== null).slice().sort((a,b) => Number(b[rank]) - Number(a[rank])).slice(0, 12).reverse();
        Plotly.react("studentRankChart", [{{
          type: "bar",
          orientation: "h",
          x: ranked.map(d => d[rank]),
          y: ranked.map(d => String(d.zip)),
          marker: {{ color: "#0f766e" }},
          text: ranked.map(d => fmt(d[rank], 2)),
          hovertemplate: "ZIP=%{{y}}<br>" + labelFor(rank) + "=%{{x:.3f}}<extra></extra>"
        }}], plotLayout(`Top ZIPs by ${{labelFor(rank)}}`, labelFor(rank), "ZIP Code", {{ showlegend: false, margin: {{ l: 96, r: 26, t: 58, b: 64 }}, yaxis: {{ title: "ZIP Code", automargin: true, type: "category" }} }}), plotConfig);

        if (DATA.student.geojson) {{
          Plotly.react("studentMap", [{{
            type: "choropleth",
            geojson: DATA.student.geojson,
            locations: dataRows.map(d => d.zip),
            z: dataRows.map(d => d[rank]),
            featureidkey: "properties.ZIP5",
            colorscale: "YlOrRd",
            marker: {{ line: {{ color: "white", width: 0.8 }} }},
            colorbar: {{ title: labelFor(rank) }},
            customdata: dataRows.map(d => [d.zip, d.total_violations, d.student_housing_metric, d.property_count]),
            hovertemplate: "ZIP=%{{customdata[0]}}<br>" + labelFor(rank) + "=%{{z:.3f}}<br>Total violations=%{{customdata[1]:,}}<br>Students=%{{customdata[2]:,}}<br>Properties=%{{customdata[3]:,}}<extra></extra>"
          }}], {{
            title: `ZIP Map: ${{labelFor(rank)}}`,
            geo: {{ fitbounds: "locations", visible: false }},
            margin: {{ l: 10, r: 10, t: 58, b: 10 }},
            paper_bgcolor: "#ffffff",
            font: {{ family: "Arial, sans-serif", color: "#172033" }}
          }}, plotConfig);
        }} else emptyPlot("studentMap", "ZIP boundary map unavailable.");

        setText("studentSummary", `Matched ZIPs: ${{dataRows.length}}. Bubble size is total violations; map and ranking use the selected ZIP metric.`);
        const tableRows = dataRows.slice().sort((a,b) => Number(b[rank] || 0) - Number(a[rank] || 0)).slice(0, 8);
        Plotly.react("studentTable", [{{
          type: "table",
          header: {{ values: ["ZIP", "Students", "Properties", "Violations", labelFor(rank)], fill: {{ color: "#e2e8f0" }}, align: "left" }},
          cells: {{ values: [
            tableRows.map(d => d.zip),
            tableRows.map(d => fmt(d.student_housing_metric)),
            tableRows.map(d => fmt(d.property_count)),
            tableRows.map(d => fmt(d.total_violations)),
            tableRows.map(d => fmt(d[rank], 2))
          ], align: "left" }}
        }}], {{ margin: {{ l: 0, r: 0, t: 8, b: 0 }} }}, plotConfig);
      }}

      function drawProperty() {{
        if (!DATA.property.available) {{
          emptyPlot("repeatedChart", "Property-risk table unavailable.");
          emptyPlot("classChart", "Property-risk table unavailable.");
          return;
        }}
        setText("propertyNote", DATA.property.ownerCoveragePct === null ? "" : `Owner coverage: ${{DATA.property.ownerCoveragePct}}% of property-risk rows.`);
        const repeated = DATA.property.repeated.slice().reverse();
        if (repeated.length) {{
          Plotly.react("repeatedChart", [{{
            type: "bar",
            orientation: "h",
            x: repeated.map(d => d.total_violations),
            y: repeated.map(d => d.property_key),
            marker: {{ color: "#2563eb" }},
            customdata: repeated.map(d => [d.violation_st, d.violation_zip, d.assessment_owner_clean || d.assessment_owner || "", d.assessment_lu_desc || d.assessment_bldg_type || ""]),
            hovertemplate: "%{{y}}<br>Violations=%{{x:,}}<br>Street=%{{customdata[0]}}<br>ZIP=%{{customdata[1]}}<br>Owner=%{{customdata[2]}}<br>Class=%{{customdata[3]}}<extra></extra>"
          }}], plotLayout("Repeated Properties With Ownership Context", "Violation Count", "Property", {{ showlegend: false }}), plotConfig);
        }} else emptyPlot("repeatedChart", "Repeated-property data unavailable.");

        const classes = DATA.property.classes.slice().reverse();
        if (classes.length) {{
          Plotly.react("classChart", [{{
            type: "bar",
            orientation: "h",
            x: classes.map(d => d.violations_per_property),
            y: classes.map(d => d.property_class),
            marker: {{ color: "#b45309" }},
            customdata: classes.map(d => [d.property_count, d.total_violations, d.open_violations]),
            hovertemplate: "%{{y}}<br>Violations/property=%{{x:.2f}}<br>Properties=%{{customdata[0]:,}}<br>Total violations=%{{customdata[1]:,}}<br>Open violations=%{{customdata[2]:,}}<extra></extra>"
          }}], plotLayout("Violation Rate by Property Class", "Violations per Property", "Property Class", {{ showlegend: false }}), plotConfig);
        }} else emptyPlot("classChart", "Property class data unavailable.");
      }}

      function initModelControls() {{
        const modelRows = DATA.model.records || [];
        const targets = [...new Set(modelRows.map(d => d.target_label || d.target).filter(Boolean))];
        const featureSets = [...new Set(modelRows.map(d => d.feature_set).filter(Boolean))];
        const models = [...new Set(modelRows.map(d => d.model_name).filter(Boolean))];
        fillSelect("modelTarget", targets.map(t => ({{ id: t, label: pretty(t) }})), targets[0]);
        fillSelect("modelMetric", DATA.model.metrics || [], "roc_auc");
        fillSelect("featureSet", featureSets.map(t => ({{ id: t, label: pretty(t) }})), featureSets[0]);
        fillSelect("featureModel", models.map(t => ({{ id: t, label: pretty(t) }})), models[0]);
        ["modelTarget", "modelMetric", "featureSet", "featureModel"].forEach(id => {{
          const el = document.getElementById(id);
          if (el) el.addEventListener("change", drawModel);
        }});
      }}
      function drawModel() {{
        if (!DATA.model.available || !DATA.model.records.length) {{
          emptyPlot("modelMetricChart", "Model results unavailable.");
          emptyPlot("featureImportanceChart", "Feature importance unavailable.");
          return;
        }}
        const target = document.getElementById("modelTarget").value;
        const metric = document.getElementById("modelMetric").value;
        const featureSet = document.getElementById("featureSet").value;
        const modelName = document.getElementById("featureModel").value;
        const filtered = DATA.model.records.filter(d => (d.target_label || d.target) === target && d[metric] !== null);
        const sets = [...new Set(filtered.map(d => d.feature_set))];
        const traces = sets.map((setName, i) => {{
          const rows = filtered.filter(d => d.feature_set === setName);
          return {{
            type: "bar",
            name: setName,
            x: rows.map(d => pretty(d.model_name)),
            y: rows.map(d => d[metric]),
            marker: {{ color: COLORS[i % COLORS.length] }},
            customdata: rows.map(d => [d.n_positive, d.positive_class_rate, d.cv_folds, d.precision, d.recall, d.pr_auc]),
            hovertemplate: "Model=%{{x}}<br>" + labelFor(metric) + "=%{{y:.4f}}<br>Positive n=%{{customdata[0]}}<br>Positive rate=%{{customdata[1]:.4f}}<br>CV folds=%{{customdata[2]}}<br>Precision=%{{customdata[3]:.4f}}<br>Recall=%{{customdata[4]:.4f}}<br>PR-AUC=%{{customdata[5]:.4f}}<extra></extra>"
          }};
        }});
        Plotly.react("modelMetricChart", traces, plotLayout(`${{pretty(target)}}: ${{labelFor(metric)}}`, "Model", labelFor(metric), {{ barmode: "group", yaxis: {{ title: labelFor(metric), range: [0, Math.min(1, Math.max(0.05, ...filtered.map(d => Number(d[metric] || 0))) * 1.18)] }} }}), plotConfig);

        const imp = (DATA.model.featureImportance || [])
          .filter(d => (d.target_label || d.target || "").includes(target) || d.target === DATA.model.records.find(r => (r.target_label || r.target) === target)?.target)
          .filter(d => d.feature_set === featureSet && d.model_name === modelName)
          .sort((a,b) => Number(b.abs_importance || 0) - Number(a.abs_importance || 0))
          .slice(0, 15)
          .reverse();
        if (imp.length) {{
          Plotly.react("featureImportanceChart", [{{
            type: "bar",
            orientation: "h",
            x: imp.map(d => d.abs_importance),
            y: imp.map(d => d.feature_name),
            marker: {{ color: imp.map(d => Number(d.importance || 0) >= 0 ? "#0f766e" : "#dc2626") }},
            customdata: imp.map(d => [d.importance, d.importance_type]),
            hovertemplate: "%{{y}}<br>Abs importance=%{{x:.4f}}<br>Signed value=%{{customdata[0]:.4f}}<br>Type=%{{customdata[1]}}<extra></extra>"
          }}], plotLayout(`Top Features: ${{pretty(modelName)}} / ${{pretty(featureSet)}}`, "Absolute Importance", "Feature", {{ showlegend: false, margin: {{ l: 210, r: 26, t: 58, b: 64 }} }}), plotConfig);
        }} else emptyPlot("featureImportanceChart", "No feature-importance rows for this target/model/feature set.");
      }}

      function activateTab(tabId) {{
        document.querySelectorAll(".tab-button").forEach(btn => btn.classList.toggle("active", btn.dataset.tab === tabId));
        document.querySelectorAll(".tab-panel").forEach(panel => panel.classList.toggle("active", panel.id === tabId));
        if (tabId === "overview") drawOverview();
        if (tabId === "student") drawStudent();
        if (tabId === "property") drawProperty();
        if (tabId === "model") drawModel();
        setTimeout(() => window.dispatchEvent(new Event("resize")), 50);
      }}

      document.querySelectorAll(".tab-button").forEach(btn => btn.addEventListener("click", () => activateTab(btn.dataset.tab)));
      initKpis();
      initStudentControls();
      initModelControls();
      drawOverview();
    </script>
  </body>
</html>
"""


def write_interactive_dashboard(
    config: InteractiveVisualizationConfig,
    paths: list[Path],
) -> Path:
    """Write the single-page interactive dashboard used as the main HTML entrypoint."""
    _ensure_plotly_bundle(config.output_dir)
    output_path = config.output_dir / "index.html"
    data = _build_dashboard_data(config, paths)
    output_path.write_text(_dashboard_html(data), encoding="utf-8")
    print(f"Saved interactive dashboard: {output_path}")
    return output_path


def write_interactive_index(
    paths: list[Path],
    output_dir: Path,
    config: InteractiveVisualizationConfig | None = None,
) -> Path:
    """Write the main interactive entrypoint."""
    if config is not None:
        return write_interactive_dashboard(config, paths)

    output_dir.mkdir(parents=True, exist_ok=True)
    visible_paths = [path for path in paths if path.name != "index.html"]
    items = "\n".join(
        f'      <li><a href="{escape(path.name)}">{escape(path.stem.replace("_", " ").title())}</a></li>'
        for path in visible_paths
    )
    body = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>University Accountability Interactive Visualizations</title>
  </head>
  <body>
    <h1>University Accountability Interactive Visualizations</h1>
    <ul>
{items}
    </ul>
  </body>
</html>
"""
    output_path = output_dir / "index.html"
    output_path.write_text(body, encoding="utf-8")
    print(f"Saved interactive index: {output_path}")
    return output_path


def generate_interactive_visualizations(config: InteractiveVisualizationConfig) -> list[Path]:
    """Generate all available interactive visualization artifacts."""
    paths: list[Path] = []
    paths.extend(generate_interactive_phase2_figures(config))
    paths.extend(generate_interactive_property_risk_outputs(config))
    paths.extend(generate_interactive_student_housing_outputs(config))
    paths.extend(generate_interactive_model_outputs(config))
    if paths:
        paths.append(write_interactive_index(paths, config.output_dir, config))
    return paths


def main() -> None:
    generate_interactive_visualizations(InteractiveVisualizationConfig())


__all__ = [
    "InteractiveVisualizationConfig",
    "generate_interactive_model_outputs",
    "generate_interactive_phase2_figures",
    "generate_interactive_property_risk_outputs",
    "generate_interactive_student_housing_outputs",
    "generate_interactive_visualizations",
    "write_interactive_dashboard",
    "write_interactive_index",
]


if __name__ == "__main__":
    main()
