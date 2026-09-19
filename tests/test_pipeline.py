import pandas as pd

from src.data_processing import basic_cleaning


def test_duplicate_rows_are_removed():
    df = pd.DataFrame({
        "Value": [1, 1, 2],
        "Group": ["A", "A", "B"]
    })

    result = basic_cleaning(df)

    assert len(result) == 2


def test_column_names_are_standardized():
    df = pd.DataFrame({
        "Test Column": [1, 2]
    })

    result = basic_cleaning(df)

    assert "test_column" in result.columns
