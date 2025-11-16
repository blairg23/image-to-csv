import pandas as pd

from image_to_csv.table_to_csv import lines_to_df


def test_lines_to_df_detects_header_by_width():
    lines = [
        "Weekly Mood Report",
        "Date    Mood    Tired",
        "2024-01-01    Happy    Low",
        "2024-01-02    Sad      High",
    ]
    df = lines_to_df(lines)
    assert list(df.columns) == ["Date", "Mood", "Tired"]
    assert len(df) == 2
    assert df.iloc[1]["Mood"] == "Sad"


def test_lines_to_df_handles_pipe_delimiter():
    lines = [
        "| Name | Score |",
        "| Alice | 10 |",
        "| Bob | 8 |",
    ]
    df = lines_to_df(lines)
    pd.testing.assert_series_equal(df["Name"], pd.Series(["Alice", "Bob"], name="Name"), check_names=True, check_dtype=False)


def test_lines_to_df_returns_text_column_when_no_table():
    lines = ["Notes", "", "No structured table detected"]
    df = lines_to_df(lines)
    assert list(df.columns) == ["text"]
    assert df.iloc[0, 0] == "Notes"
