"""Zamana görə train/valid/test bölgüsü və sızıntı yoxlaması (notebook, train.py və testlər üçün vahid mənbə)."""
import pandas as pd

from .config import (TARGET, TEST_MONTHS, TIME_COL, TRAIN_MONTHS, TRAIN_MONTHS_SEP, VALID_CAL_MONTHS, VALID_MONTHS,
                     VALID_THR_MONTHS)


def assert_chronological(train_months, valid_months, test_months):
    """Aylar kəsişməməli və train < valid < test sırası pozulmamalıdır (gələcək → keçmiş sızıntısı yoxdur)."""
    t, v, e = set(train_months), set(valid_months), set(test_months)
    if (t & v) or (t & e) or (v & e):
        raise ValueError("Split-lərdə ortaq ay var (sızıntı).")
    if not (max(t) < min(v) and max(v) < min(e)):
        raise ValueError("Split-lər xronoloji deyil: train < valid < test olmalıdır.")


def time_split(df: pd.DataFrame, train_months=TRAIN_MONTHS, valid_months=VALID_MONTHS, test_months=TEST_MONTHS):
    """month sütununa görə {'train','valid','test': (X, y)} qaytarır; xronologiyanı yoxlayır."""
    assert_chronological(train_months, valid_months, test_months)
    parts = {}
    for name, months in [("train", train_months), ("valid", valid_months), ("test", test_months)]:
        part = df[df[TIME_COL].isin(months)]
        parts[name] = (part.drop(columns=[TARGET]), part[TARGET])
    return parts


def assert_chronological_parts(*parts):
    """İstənilən sayda hissə üçün: kəsişmə yoxdur və hər hissə növbətidən əvvəl bitir."""
    sets = [set(p) for p in parts]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            if sets[i] & sets[j]:
                raise ValueError("Split-lərdə ortaq ay var (sızıntı).")
    for a, b in zip(sets, sets[1:]):
        if not max(a) < min(b):
            raise ValueError("Split-lər xronoloji deyil: hər hissə növbətidən əvvəl olmalıdır.")


def time_split_separate_valid(df: pd.DataFrame, train_months=TRAIN_MONTHS_SEP, valid_calibration_months=VALID_CAL_MONTHS,
                              valid_threshold_months=VALID_THR_MONTHS, test_months=TEST_MONTHS):
    """4 hissəli zaman split-i: train | valid_calibration | valid_threshold | test.

    Calibration və threshold fərqli aylarda seçilir (tək valid ayının həddən artıq yüklənməsinin qarşısı).
    """
    assert_chronological_parts(train_months, valid_calibration_months, valid_threshold_months, test_months)
    parts = {}
    for name, months in [("train", train_months), ("valid_calibration", valid_calibration_months),
                         ("valid_threshold", valid_threshold_months), ("test", test_months)]:
        part = df[df[TIME_COL].isin(months)]
        if part.empty:
            raise ValueError(f"'{name}' hissəsi boşdur (aylar: {list(months)}).")
        parts[name] = (part.drop(columns=[TARGET]), part[TARGET])
    return parts
