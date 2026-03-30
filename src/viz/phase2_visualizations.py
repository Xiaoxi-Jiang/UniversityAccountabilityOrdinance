"""Exploratory figures for Phase 2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from src.data.features import (
    Phase2FeatureConfig,
    VIOLATION_TYPE_CANDIDATES,
    first_available_column,
    load_phase2_source_data,
    prepare_violations_frame,
)
from src.viz.plot_utils import add_bar_labels, plt, save_figure, wrap_labels


DATE_COLUMN_CANDIDATES = [
    "violdttm",
    "violation_date",
    "status_dttm",
    "date",
    "first_violation_date",
    "last_violation_date",
]
DISTRICT_COLUMN_CANDIDATES = [
    "district",
    "neighborhood",
    "city_council_district",
    "council_district",
]
SEVERITY_COLUMN_CANDIDATES = ["severity", "severity_level", "violation_severity"]
FIGURE_OUTPUT_GROUPS = {
    "severity distribution": {"severity_distribution.png"},
    "status distribution": {"status_distribution.png"},
    "top violation types": {"top_violation_types.png"},
    "trend plot": {"district_trends.png", "violations_over_time.png"},
}


@dataclass(frozen=True)
class Phase2VisualizationConfig:
    input_path: Path = Phase2FeatureConfig().input_path
    output_dir: Path = Path("outputs/figures")


def get_available_date_column(df: pd.DataFrame) -> str | None:
    """Return the first usable datetime column after coercing likely candidates."""
    for column in DATE_COLUMN_CANDIDATES:
        if column not in df.columns:
            continue
        parsed = pd.to_datetime(df[column], errors="coerce")
        df[column] = parsed
        if parsed.notna().any():
            return column
    return None


def _derive_severity_proxy(df: pd.DataFrame) -> pd.Series | None:
    """Approximate severity using violation text when no true severity field exists.

    The mapping is intentionally heuristic and presentation-oriented:
    - high risk (proxy): immediate safety / dangerous-condition language
    - medium risk (proxy): maintenance and habitability issues
    - low risk (proxy): permit, inspection, zoning, and administrative issues
    - uncategorized: text that does not match the current keyword rules
    """
    severity_col = first_available_column(df, SEVERITY_COLUMN_CANDIDATES)
    if severity_col is not None:
        return df[severity_col].astype("string").fillna("unknown")

    text_col = first_available_column(df, VIOLATION_TYPE_CANDIDATES)
    if text_col is None:
        return None

    text = df[text_col].astype("string").str.lower()
    severe_pattern = (
        r"fire|unsafe|danger|collapse|structural|hazard|emergency|life safety"
    )
    moderate_pattern = (
        r"maintenance|sanitary|mold|heat|electrical|plumbing|rodent|trash|"
        r"ventilation|water|sewage|infest|habitable"
    )
    low_pattern = (
        r"permit|inspection|certificate|occupancy|right of entry|zoning|"
        r"administrative|paperwork|documentation|sign|code"
    )

    severity = pd.Series("uncategorized", index=df.index, dtype="string")
    severity.loc[text.str.contains(severe_pattern, na=False)] = "high risk (proxy)"
    severity.loc[text.str.contains(moderate_pattern, na=False)] = "medium risk (proxy)"
    severity.loc[text.str.contains(low_pattern, na=False)] = "low risk (proxy)"
    return severity


def plot_severity_distribution(df: pd.DataFrame, output_dir: Path) -> Path:
    """Plot severity distribution using either a real field or an explicit proxy."""
    severity = _derive_severity_proxy(df)

    if severity is not None and severity.nunique(dropna=True) > 0:
        ordered_labels = [
            "high risk (proxy)",
            "medium risk (proxy)",
            "low risk (proxy)",
            "uncategorized",
            "unknown",
        ]
        counts = severity.fillna("unknown").value_counts()
        counts = counts.reindex(
            [label for label in ordered_labels if label in counts.index],
            fill_value=0,
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        bar_colors = ["#dc2626", "#f59e0b", "#2563eb", "#9ca3af", "#6b7280"][: len(counts)]
        ax.bar(counts.index.astype(str), counts.values, color=bar_colors)
        ax.set_title("Violation Severity Distribution (Proxy-Based Labels)")
        ax.set_xlabel("Severity Category")
        ax.set_ylabel("Violation Count")
        ax.tick_params(axis="x", rotation=30)
        add_bar_labels(ax)
        return save_figure(output_dir / "severity_distribution.png")

    raise ValueError("Cannot generate severity distribution with current columns.")


def plot_status_distribution(df: pd.DataFrame, output_dir: Path) -> Path:
    """Plot normalized status counts as a separate check-in figure."""
    if "status" not in df.columns:
        raise ValueError("Cannot generate status distribution without a status column.")

    counts = df["status"].astype("string").fillna("unknown").value_counts()
    percentages = counts / max(int(counts.sum()), 1) * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index.astype(str), counts.values, color=["#2563eb", "#93c5fd", "#9ca3af"][: len(counts)])
    ax.set_title("Violation Status Distribution")
    ax.set_xlabel("Status")
    ax.set_ylabel("Violation Count")
    ax.tick_params(axis="x", rotation=30)
    for bar, count, pct in zip(bars, counts.values, percentages.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts.values) * 0.01,
            f"{int(count)}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    return save_figure(output_dir / "status_distribution.png")


def plot_top_violation_types(df: pd.DataFrame, output_dir: Path) -> Path:
    """Plot the most common violation categories or descriptions."""
    violation_type_col = first_available_column(df, VIOLATION_TYPE_CANDIDATES)
    if violation_type_col is None:
        raise ValueError("Cannot generate top violation types without a type or description column.")

    counts = (
        df[violation_type_col]
        .astype("string")
        .replace("", pd.NA)
        .dropna()
        .value_counts()
        .head(10)
        .sort_values(ascending=True)
    )
    if counts.empty:
        raise ValueError("Cannot generate top violation types because no non-empty values were found.")

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(wrap_labels(counts.index.to_series()), counts.values, color="#2563eb")
    ax.set_title("Top 10 Violation Types by Count")
    ax.set_xlabel("Violation Count")
    ax.set_ylabel("Violation Type")
    ax.bar_label(bars, fmt="%.0f", padding=3, fontsize=8)
    return save_figure(output_dir / "top_violation_types.png")


def plot_district_or_time_trends(
    df: pd.DataFrame,
    output_dir: Path,
    date_col: str | None = None,
) -> Path | None:
    """Plot district trends when available, otherwise overall trend over time."""
    usable_date_col = date_col or get_available_date_column(df)
    district_col = first_available_column(df, DISTRICT_COLUMN_CANDIDATES)

    if usable_date_col is not None and district_col is not None:
        trend_df = df.dropna(subset=[usable_date_col, district_col]).copy()
        if not trend_df.empty:
            trend_df.loc[:, "year_month"] = (
                trend_df[usable_date_col].dt.to_period("M").dt.to_timestamp()
            )
            top_districts = trend_df[district_col].astype("string").value_counts().head(5).index
            plot_df = (
                trend_df[trend_df[district_col].astype("string").isin(top_districts)]
                .groupby(["year_month", district_col])
                .size()
                .unstack(fill_value=0)
                .sort_index()
            )
            fig, ax = plt.subplots(figsize=(10, 6))
            for column in plot_df.columns:
                ax.plot(plot_df.index, plot_df[column], marker="o", linewidth=1.5, label=str(column))
            ax.set_title("Violation Trends by District")
            ax.set_xlabel("Month")
            ax.set_ylabel("Violation Count")
            ax.tick_params(axis="x", rotation=45)
            ax.legend(title="District", fontsize=8)
            return save_figure(output_dir / "district_trends.png")

    if usable_date_col is None:
        print("Skipping trend plot: no usable date column found.")
        return None

    dated = df.dropna(subset=[usable_date_col]).copy()
    monthly_periods = dated[usable_date_col].dt.to_period("M")
    if monthly_periods.nunique() >= 12:
        trend = (
            dated.assign(period=monthly_periods.dt.to_timestamp())
            .groupby("period")
            .size()
            .sort_index()
        )
        x_label = "Month"
    else:
        trend = (
            dated.assign(period=dated[usable_date_col].dt.to_period("Y").dt.to_timestamp())
            .groupby("period")
            .size()
            .sort_index()
        )
        x_label = "Year"
    if trend.empty:
        print("Skipping trend plot: usable date column found but no valid dated rows are available.")
        return None

    date_label = "status timestamp" if usable_date_col == "status_dttm" else usable_date_col
    fig, ax = plt.subplots(figsize=(10, 6))
    trend_df = trend.rename("monthly_count").to_frame()
    trend_df["rolling_12m_avg"] = trend_df["monthly_count"].rolling(window=12, min_periods=3).mean()
    ax.plot(
        trend_df.index,
        trend_df["monthly_count"],
        linewidth=1.2,
        color="#93c5fd",
        alpha=0.9,
        label="Monthly count",
    )
    if trend_df["rolling_12m_avg"].notna().any():
        ax.plot(
            trend_df.index,
            trend_df["rolling_12m_avg"],
            linewidth=2.5,
            color="#1d4ed8",
            label="12-month rolling average",
        )
    ax.set_title(f"Violations Over Time ({date_label}; monthly with rolling average)")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Violation Count")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    return save_figure(output_dir / "violations_over_time.png")


def summarize_figure_results(paths: list[Path]) -> tuple[list[str], list[str]]:
    """Summarize which figure groups were generated versus skipped."""
    generated_names = {path.name for path in paths}
    generated: list[str] = []
    skipped: list[str] = []

    for label, file_names in FIGURE_OUTPUT_GROUPS.items():
        if generated_names.intersection(file_names):
            generated.append(label)
        else:
            skipped.append(label)

    return generated, skipped


def generate_phase2_figures(config: Phase2VisualizationConfig) -> list[Path]:
    """Generate the Phase 2 exploratory figures supported by the current data."""
    if not config.input_path.exists():
        raise FileNotFoundError(
            f"Missing cleaned violations dataset at {config.input_path}. Run Phase 1 first."
        )

    df = load_phase2_source_data(Phase2FeatureConfig(input_path=config.input_path))
    prepared, _ = prepare_violations_frame(df)
    date_col = get_available_date_column(prepared)
    if date_col is not None:
        print(f"Using date column for trend plotting: {date_col}")
    else:
        print("No usable date column available for trend plotting.")
    output_paths: list[Path] = []

    for plot_name, plot_func in [
        ("severity distribution", lambda: plot_severity_distribution(prepared, config.output_dir)),
        ("status distribution", lambda: plot_status_distribution(prepared, config.output_dir)),
        ("top violation types", lambda: plot_top_violation_types(prepared, config.output_dir)),
        (
            "trend plot",
            lambda: plot_district_or_time_trends(
                prepared,
                config.output_dir,
                date_col=date_col,
            ),
        ),
    ]:
        try:
            output_path = plot_func()
        except Exception as exc:
            print(f"Skipping {plot_name}: {exc}")
            continue
        if output_path is not None:
            output_paths.append(output_path)

    return output_paths


def main() -> None:
    generate_phase2_figures(Phase2VisualizationConfig())


if __name__ == "__main__":
    main()
