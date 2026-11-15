"""Məlumat injeksiyası: oxu → schema yoxla → keyfiyyət hesabatı → (istəyə görə) Parquet.

İstifadə:
    python -m src.ingest data/Base.csv              # yoxla və results/data_quality.json yaz
    python -m src.ingest data/Base.csv --parquet    # əlavə: data/processed/base.parquet (pip install pyarrow)
Modul kimi: load_dataset(path) CSV və ya Parquet oxuyur və schema-nı yoxlayır (train.py bunu istifadə edir).
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype

from .config import CATEGORICAL, MISSING_MINUS1, NEGATIVE_MISSING, TARGET, TIME_COL

NUMERIC = [
    "income", "name_email_similarity", "prev_address_months_count", "current_address_months_count",
    "customer_age", "days_since_request", "intended_balcon_amount", "zip_count_4w", "velocity_6h",
    "velocity_24h", "velocity_4w", "bank_branch_count_8w", "date_of_birth_distinct_emails_4w",
    "credit_risk_score", "email_is_free", "phone_home_valid", "phone_mobile_valid", "bank_months_count",
    "has_other_cards", "proposed_credit_limit", "foreign_request", "session_length_in_minutes",
    "keep_alive_session", "device_distinct_emails_8w", "device_fraud_count",
]
REQUIRED_COLUMNS = [TARGET, TIME_COL] + NUMERIC + CATEGORICAL  # 32 sütun
MONTH_RANGE = (0, 7)


class DataValidationError(ValueError):
    """Data gözlənilən BAF schema-sına uyğun deyil."""


def validate_schema(df: pd.DataFrame) -> list:
    """Schema-nı yoxlayır; xəta varsa DataValidationError atır, əks halda xəbərdarlıqlar siyahısını qaytarır."""
    if df.empty:
        raise DataValidationError("Data boşdur.")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"Çatışmayan sütunlar: {missing}")

    errors = []
    not_numeric = [c for c in NUMERIC + [TARGET, TIME_COL] if not is_numeric_dtype(df[c])]
    if not_numeric:
        errors.append(f"Rəqəmsal olmalı sütunlar rəqəmsal deyil: {not_numeric}")
    else:
        if df[TARGET].isna().any() or not set(df[TARGET].unique()) <= {0, 1}:
            errors.append(f"{TARGET} yalnız 0/1 ola bilər (NaN olmamalıdır).")
        m = df[TIME_COL]
        if m.isna().any() or not m.between(*MONTH_RANGE).all() or not (m % 1 == 0).all():
            errors.append(f"{TIME_COL} {MONTH_RANGE[0]}-{MONTH_RANGE[1]} aralığında tam ədəd olmalıdır.")
    if errors:
        raise DataValidationError(" | ".join(errors))
    return [f"Gözlənilməyən əlavə sütunlar (nəzərə alınmır): {sorted(set(df.columns) - set(REQUIRED_COLUMNS))}"] \
        if set(df.columns) - set(REQUIRED_COLUMNS) else []


def data_quality_report(df: pd.DataFrame) -> dict:
    """Sətir/sütun sayı, dublikat, null, gizli missing (-1 / mənfi kodlar), fraud nisbəti və aylar üzrə paylanma."""
    n = len(df)
    hidden = {}
    for c in MISSING_MINUS1:
        k = int((df[c] == -1).sum())
        hidden[c] = {"count": k, "share": k / n}
    for c in NEGATIVE_MISSING:
        k = int((df[c] < 0).sum())
        hidden[c] = {"count": k, "share": k / n}
    nulls = {c: int(v) for c, v in df.isna().sum().items() if v}
    by_month = df.groupby(TIME_COL)[TARGET].agg(["size", "mean"])
    return {
        "rows": n,
        "columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "null_counts": nulls,
        "hidden_missing": hidden,
        "fraud_rate": float(df[TARGET].mean()),
        "rows_by_month": {str(int(m)): int(v) for m, v in by_month["size"].items()},
        "fraud_rate_by_month": {str(int(m)): float(v) for m, v in by_month["mean"].items()},
    }


def load_dataset(path, validate: bool = True) -> pd.DataFrame:
    """CSV və ya Parquet oxuyur (uzantıya görə) və schema-nı yoxlayır."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data tapılmadı: {path} (Base.csv-ni Kaggle-dan yükləyib data/ qovluğuna qoyun)")
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if validate:
        validate_schema(df)
    return df


def save_parquet(df: pd.DataFrame, out_path) -> Path:
    """Təmiz datanı Parquet kimi saxlayır (CSV-dən xeyli sürətli oxunur)."""
    try:
        import pyarrow  # noqa: F401
    except ImportError as e:
        raise RuntimeError("Parquet üçün pyarrow lazımdır: pip install pyarrow") from e
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return out_path


def write_report(report: dict, out_path) -> Path:
    """Keyfiyyət hesabatını JSON kimi yazır."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description="BAF datasını yoxla, keyfiyyət hesabatı yaz, istəyə görə Parquet saxla.")
    ap.add_argument("data", help="Base.csv və ya .parquet yolu")
    ap.add_argument("--report", default="results/data_quality.json")
    ap.add_argument("--parquet", nargs="?", const="data/processed/base.parquet", default=None,
                    help="Parquet çıxış yolu (default: data/processed/base.parquet)")
    args = ap.parse_args(argv)

    df = load_dataset(args.data)
    report = data_quality_report(df)
    write_report(report, args.report)
    print(f"OK: {report['rows']:,} sətir, {report['columns']} sütun, fraud {report['fraud_rate']:.2%}, "
          f"dublikat {report['duplicate_rows']}. Hesabat: {args.report}")
    if args.parquet:
        print("Parquet:", save_parquet(df, args.parquet))


if __name__ == "__main__":
    main()
