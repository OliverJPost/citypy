import pandas as pd


def column_exists(column, table):
    return column in table.columns


def test_column_exists():
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert column_exists("a", df)
    assert not column_exists("b", df)


def is_float(column, table):
    return table[column].dtype == "float64"


def test_is_float():
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert not is_float("a", df)
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    assert is_float("a", df)


def no_nulls(column, table):
    return table[column].notna().all()


def test_no_nulls():
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert no_nulls("a", df)
    df = pd.DataFrame({"a": [1, 2, None]})
    assert not no_nulls("a", df)


def not_all_zero(column, table):
    return (table[column] != 0).any()


def test_not_all_zero():
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert not_all_zero("a", df)
    df = pd.DataFrame({"a": [0, 0, 0]})
    assert not not_all_zero("a", df)
    df = pd.DataFrame({"a": [0, 0, 1]})
    assert not_all_zero("a", df)


def drop_if_exists(df, column):
    df.drop(columns=[column], inplace=True, errors="ignore")
    assert column not in df.columns


def test_drop_if_exists():
    df = pd.DataFrame({"a": [1, 2, 3]})
    drop_if_exists(df, "a")
    assert "a" not in df.columns
    drop_if_exists(df, "b")
    assert "b" not in df.columns
