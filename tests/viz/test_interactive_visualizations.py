from pathlib import Path

import pandas as pd

from src.viz.interactive_visualizations import (
    InteractiveVisualizationConfig,
    generate_interactive_visualizations,
    generate_interactive_model_outputs,
    generate_interactive_phase2_figures,
    generate_interactive_student_housing_outputs,
)


def test_generate_interactive_phase2_figures_writes_html_outputs(tmp_path: Path):
    input_path = tmp_path / "violations_clean.csv"
    output_dir = tmp_path / "interactive"
    pd.DataFrame(
        {
            "violation_st": ["1 Comm Ave", "1 Comm Ave", "2 Bay State Rd", "3 Beacon St"],
            "violation_zip": ["02215", "02215", "02215", "02115"],
            "violdttm": ["2024-01-01", "2024-02-01", "2024-03-15", "2024-04-01"],
            "status": ["open", "closed", "open", "closed"],
            "violationtype": ["Unsafe and Dangerous", "Maintenance", "Failure to Obtain Permit", "Trash"],
        }
    ).to_csv(input_path, index=False)

    paths = generate_interactive_phase2_figures(
        InteractiveVisualizationConfig(input_path=input_path, output_dir=output_dir)
    )

    names = {path.name for path in paths}
    assert {
        "severity_distribution.html",
        "status_distribution.html",
        "top_violation_types.html",
        "violations_over_time.html",
    }.issubset(names)
    assert "Plotly.newPlot" in (output_dir / "violations_over_time.html").read_text(encoding="utf-8")


def test_generate_interactive_student_housing_outputs_from_relationship_table(tmp_path: Path):
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir()
    output_dir = tmp_path / "interactive"
    pd.DataFrame(
        {
            "zip": ["02115", "02134", "02118"],
            "total_violations": [120, 200, 80],
            "property_count": [60, 100, 40],
            "student_housing_metric": [5000, 4500, 1500],
            "students_per_property": [83.33, 45.0, 37.5],
            "violations_per_property": [2.0, 2.0, 2.0],
            "violations_per_1000_students": [24.0, 44.44, 53.33],
        }
    ).to_csv(tables_dir / "student_housing_relationship.csv", index=False)

    paths = generate_interactive_student_housing_outputs(
        InteractiveVisualizationConfig(
            tables_dir=tables_dir,
            output_dir=output_dir,
            student_context_path=tmp_path / "missing_student.csv",
            zip_boundary_path=tmp_path / "missing_boundaries.geojson",
        )
    )

    names = {path.name for path in paths}
    assert "student_housing_relationship.html" in names
    assert "student_housing_violation_intensity_by_zip.html" in names
    assert "student_housing_zip_context.html" not in names


def test_generate_interactive_model_outputs_writes_comparison(tmp_path: Path):
    output_dir = tmp_path / "interactive"
    results_path = tmp_path / "improved_model_results.csv"
    pd.DataFrame(
        [
            {
                "target_label": "any_violation",
                "feature_set": "behavioral_only",
                "model_name": "logistic_regression",
                "n_positive": 10,
                "positive_class_rate": 0.1,
                "cv_folds": 5,
                "balanced_accuracy": 0.62,
                "precision": 0.21,
                "recall": 0.55,
                "f1": 0.30,
                "roc_auc": 0.70,
                "pr_auc": 0.18,
            },
            {
                "target_label": "any_violation",
                "feature_set": "behavioral_plus_static",
                "model_name": "random_forest",
                "n_positive": 10,
                "positive_class_rate": 0.1,
                "cv_folds": 5,
                "balanced_accuracy": 0.58,
                "precision": 0.25,
                "recall": 0.35,
                "f1": 0.29,
                "roc_auc": 0.73,
                "pr_auc": 0.20,
            },
        ]
    ).to_csv(results_path, index=False)

    paths = generate_interactive_model_outputs(
        InteractiveVisualizationConfig(
            output_dir=output_dir,
            improved_model_results_path=results_path,
        )
    )

    assert [path.name for path in paths] == ["model_performance_comparison.html"]
    assert "Cross-Validated Model Performance Comparison" in (
        output_dir / "model_performance_comparison.html"
    ).read_text(encoding="utf-8")


def test_generate_interactive_visualizations_writes_dashboard_index(tmp_path: Path):
    input_path = tmp_path / "violations_clean.csv"
    output_dir = tmp_path / "interactive"
    pd.DataFrame(
        {
            "violation_st": ["1 Comm Ave", "2 Bay State Rd", "3 Beacon St"],
            "violation_zip": ["02215", "02215", "02115"],
            "violdttm": ["2024-01-01", "2024-02-01", "2024-03-15"],
            "status": ["open", "closed", "open"],
            "violationtype": ["Unsafe and Dangerous", "Maintenance", "Failure to Obtain Permit"],
        }
    ).to_csv(input_path, index=False)

    paths = generate_interactive_visualizations(
        InteractiveVisualizationConfig(
            input_path=input_path,
            output_dir=output_dir,
            property_risk_path=tmp_path / "missing_property.csv",
            student_context_path=tmp_path / "missing_student.csv",
            improved_model_results_path=tmp_path / "missing_model.csv",
            zip_boundary_path=tmp_path / "missing_boundaries.geojson",
        )
    )

    names = {path.name for path in paths}
    body = (output_dir / "index.html").read_text(encoding="utf-8")
    assert "index.html" in names
    assert "tab-button" in body
    assert "Student ZIP Context" in body
    assert (output_dir / "plotly.min.js").exists()
