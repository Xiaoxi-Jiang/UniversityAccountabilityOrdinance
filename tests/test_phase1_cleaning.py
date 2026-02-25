import pandas as pd

from src.data.phase1_pipeline import clean_violations


def test_clean_violations_standardizes_and_derives_flag():
    df = pd.DataFrame(
        {
            "Case No": ["A1", "A1", "B2"],
            "Status": [" Open ", "Closed", "pending"],
            "ViolDttm": ["2024-01-10", "2024-01-10", "bad-date"],
            "Description": ["d1", "d1", "d2"],
        }
    )

    out = clean_violations(df)

    assert "case_no" in out.columns
    assert "status" in out.columns
    assert "violdttm" in out.columns
    assert "is_open_violation" in out.columns
    assert len(out) == 2

    open_row = out.loc[out["case_no"] == "A1"].iloc[0]
    pending_row = out.loc[out["case_no"] == "B2"].iloc[0]

    assert open_row["status"] == "open"
    assert open_row["is_open_violation"] == 1
    assert pending_row["is_open_violation"] == 1


def test_clean_violations_handles_missing_status():
    df = pd.DataFrame({"Case No": ["X1"], "Status": [None]})
    out = clean_violations(df)
    assert out.iloc[0]["status"] == "unknown"
