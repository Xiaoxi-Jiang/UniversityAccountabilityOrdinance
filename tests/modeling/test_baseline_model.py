import pandas as pd

from src.modeling.baseline_model import build_zip_level_modeling_frame


def test_build_zip_level_modeling_frame_creates_zip_level_target():
    df = pd.DataFrame(
        {
            "property_key": ["a", "b", "c", "d"],
            "violation_zip": ["02118", "02118", "02121", "02121"],
            "total_violations": [3, 2, 4, 1],
            "open_violations": [1, 0, 2, 0],
            "closed_violations": [2, 2, 2, 1],
            "distinct_violation_types": [2, 1, 3, 1],
            "violations_per_year": [1.5, 1.0, 2.0, 1.0],
            "days_since_last_violation": [10, 20, 5, 30],
        }
    )

    out = build_zip_level_modeling_frame(df)

    assert "higher_open_share_than_median" in out.columns
    assert len(out) == 2
    assert out["target_threshold_open_share"].notna().all()
