import json

import pandas as pd
import pytest

from src.ingest import (DataValidationError, REQUIRED_COLUMNS, data_quality_report, load_dataset, save_parquet,
                        validate_schema, write_report)


def test_required_columns_count():
    assert len(REQUIRED_COLUMNS) == 32 and len(set(REQUIRED_COLUMNS)) == 32


def test_valid_dataset_passes(df):
    assert validate_schema(df) == []


def test_extra_column_is_only_a_warning(df):
    assert len(validate_schema(df.assign(extra=1))) == 1


def test_missing_column_raises(df):
    with pytest.raises(DataValidationError, match="Çatışmayan"):
        validate_schema(df.drop(columns=["income"]))


def test_bad_month_raises(df):
    d = df.copy()
    d.loc[d.index[0], "month"] = 9
    with pytest.raises(DataValidationError, match="month"):
        validate_schema(d)


def test_bad_target_raises(df):
    d = df.copy()
    d.loc[d.index[0], "fraud_bool"] = 2
    with pytest.raises(DataValidationError, match="fraud_bool"):
        validate_schema(d)


def test_non_numeric_column_raises(df):
    with pytest.raises(DataValidationError, match="rəqəmsal"):
        validate_schema(df.assign(income=df["income"].astype(str)))


def test_empty_raises(df):
    with pytest.raises(DataValidationError):
        validate_schema(df.iloc[0:0])


def test_quality_report(df, tmp_path):
    r = data_quality_report(df)
    assert r["rows"] == len(df) and r["columns"] == 32
    assert 0 < r["fraud_rate"] < 1
    assert set(r["rows_by_month"]) == {str(m) for m in range(8)}
    assert 0.6 < r["hidden_missing"]["prev_address_months_count"]["share"] < 0.8
    path = write_report(r, tmp_path / "q.json")
    assert json.loads(path.read_text(encoding="utf-8"))["rows"] == len(df)


def test_csv_roundtrip(tmp_path, df):
    path = tmp_path / "b.csv"
    df.to_csv(path, index=False)
    assert load_dataset(path).shape == df.shape


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "yoxdur.csv")


def test_parquet_roundtrip(tmp_path, df):
    pytest.importorskip("pyarrow")
    path = save_parquet(df, tmp_path / "p" / "base.parquet")
    pd.testing.assert_frame_equal(load_dataset(path).reset_index(drop=True), df.reset_index(drop=True))
