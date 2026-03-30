from pathlib import Path

import pandas as pd

from src.viz.phase2_visualizations import (
    Phase2VisualizationConfig,
    _derive_severity_proxy,
    generate_phase2_figures,
    get_available_date_column,
    summarize_figure_results,
)


def test_get_available_date_column_parses_first_usable_candidate():
    df = pd.DataFrame(
        {
            "violdttm": ["bad-date", None],
            "first_violation_date": ["2024-01-01", None],
        }
    )

    date_col = get_available_date_column(df)

    assert date_col == "first_violation_date"
    assert pd.api.types.is_datetime64_any_dtype(df["first_violation_date"])


def test_generate_phase2_figures_skips_trend_plot_without_dates(tmp_path: Path):
    input_path = tmp_path / "violations_clean.csv"
    output_dir = tmp_path / "figures"
    pd.DataFrame(
        {
            "violation_st": ["1 Comm Ave", "2 Bay State Rd"],
            "violation_zip": ["02215", "02215"],
            "status": ["open", "closed"],
            "violationtype": ["Trash", "Heat"],
        }
    ).to_csv(input_path, index=False)

    paths = generate_phase2_figures(
        Phase2VisualizationConfig(input_path=input_path, output_dir=output_dir)
    )
    generated, skipped = summarize_figure_results(paths)

    assert len(paths) == 3
    assert (output_dir / "severity_distribution.png").exists()
    assert (output_dir / "status_distribution.png").exists()
    assert (output_dir / "top_violation_types.png").exists()
    assert "trend plot" in skipped
    assert "severity distribution" in generated
    assert "status distribution" in generated
    assert "top violation types" in generated


def test_generate_phase2_figures_uses_fallback_time_trend_when_dates_exist(tmp_path: Path):
    input_path = tmp_path / "violations_clean.csv"
    output_dir = tmp_path / "figures"
    pd.DataFrame(
        {
            "violation_st": ["1 Comm Ave", "1 Comm Ave", "2 Bay State Rd"],
            "violation_zip": ["02215", "02215", "02215"],
            "violdttm": ["2024-01-01", "2024-02-01", "2024-02-15"],
            "status": ["open", "closed", "open"],
            "violationtype": ["Trash", "Trash", "Heat"],
        }
    ).to_csv(input_path, index=False)

    paths = generate_phase2_figures(
        Phase2VisualizationConfig(input_path=input_path, output_dir=output_dir)
    )

    assert (output_dir / "violations_over_time.png").exists()
    assert any(path.name == "violations_over_time.png" for path in paths)


def test_derive_severity_proxy_uses_presentation_friendly_labels():
    df = pd.DataFrame(
        {
            "description": [
                "Unsafe and Dangerous",
                "Maintenance issue",
                "Failure to Obtain Permit",
                "Unknown item",
            ]
        }
    )

    severity = _derive_severity_proxy(df)

    assert severity.tolist() == [
        "high risk (proxy)",
        "medium risk (proxy)",
        "low risk (proxy)",
        "uncategorized",
    ]
