import pytest

from src.splits import assert_chronological_parts, time_split_separate_valid
from src.train import train_and_save


def test_four_way_split_months(df):
    parts = time_split_separate_valid(df)
    months = {k: set(X["month"]) for k, (X, _) in parts.items()}
    assert months == {"train": {0, 1, 2, 3}, "valid_calibration": {4}, "valid_threshold": {5}, "test": {6, 7}}


def test_overlap_raises():
    with pytest.raises(ValueError, match="ortaq"):
        assert_chronological_parts([0, 1], [1, 2], [3])


def test_non_chronological_raises():
    with pytest.raises(ValueError, match="xronoloji"):
        assert_chronological_parts([0, 1], [5], [3])


def test_empty_part_raises(df):
    with pytest.raises(ValueError, match="boşdur"):
        time_split_separate_valid(df[df["month"] != 4])


def test_separate_valid_training(tmp_path, df):
    meta = train_and_save(df, tmp_path / "m", separate_valid=True)
    assert meta["split_mode"] == "separate_valid"
    assert set(meta["split_rows"]) == {"train", "valid_calibration", "valid_threshold", "test"}
    assert 0 < meta["threshold"] < 1
    assert (tmp_path / "m" / "model.joblib").exists()


def test_default_mode_unchanged(trained):
    _, meta = trained
    assert meta["split_mode"] == "single_valid" and set(meta["split_rows"]) == {"train", "valid", "test"}
