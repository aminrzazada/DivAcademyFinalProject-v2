"""Velocity ablation: velocity siqnalı güclü drift göstərir (PSI 1-3.7). Model onlarsız nə qədər pisləşir?

Eyni HGB parametrləri ilə iki model qurulur: (1) tam feature dəsti, (2) adında "velocity" olan bütün feature-lər
(xam velocity_6h/24h/4w və onlardan yaranan nisbət/sürətlənmə/interaksiya) çıxarılmış. Valid və test metrikləri
və test PR-AUC fərqinin paired bootstrap inam intervalı results/velocity_ablation.csv-yə yazılır.
Səbəb əlaqəsi iddia olunmur: nəticə yalnız "bu feature-lər olmadan performans necə dəyişir" sualına cavabdır.

İstifadə: python scripts/velocity_ablation.py data/Base.csv [--out results/velocity_ablation.csv] [--n-boot 200]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import RANDOM_STATE, TIME_COL  # noqa: E402
from src.evaluation import score_report  # noqa: E402
from src.features import build_pipeline  # noqa: E402
from src.ingest import load_dataset  # noqa: E402
from src.splits import time_split  # noqa: E402
from src.train import DEFAULT_HGB  # noqa: E402


def paired_bootstrap_delta(y, p_a, p_b, n_boot=200, seed=RANDOM_STATE):
    """PR-AUC(a) - PR-AUC(b) üçün paired bootstrap: (orta, 2.5%, 97.5%)."""
    y, p_a, p_b = np.asarray(y), np.asarray(p_a), np.asarray(p_b)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() == 0:
            continue
        deltas.append(average_precision_score(y[idx], p_a[idx]) - average_precision_score(y[idx], p_b[idx]))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(np.mean(deltas)), float(lo), float(hi)


def run_ablation(df: pd.DataFrame, n_boot: int = 200) -> pd.DataFrame:
    """Tam model vs velocity-siz model. Seçim yoxdur: yalnız valid və test metrikləri hesablanır."""
    parts = time_split(df)
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = parts["train"], parts["valid"], parts["test"]
    drop_cols = [c for c in X_tr.columns if X_tr[c].nunique(dropna=False) <= 1] + [TIME_COL]

    full = build_pipeline(HistGradientBoostingClassifier(**DEFAULT_HGB), drop_cols).fit(X_tr, y_tr)
    all_features = full.named_steps["select"].cols_
    keep = [c for c in all_features if "velocity" not in c]
    removed = [c for c in all_features if "velocity" in c]
    no_vel = build_pipeline(HistGradientBoostingClassifier(**DEFAULT_HGB), drop_cols, keep_features=keep).fit(X_tr, y_tr)

    rows, test_scores = [], {}
    for name, model, n_feat in [("full", full, len(all_features)), ("no_velocity", no_vel, len(keep))]:
        p_va = model.predict_proba(X_va)[:, 1]
        p_te = model.predict_proba(X_te)[:, 1]
        test_scores[name] = p_te
        v, t = score_report(y_va, p_va), score_report(y_te, p_te)
        rows.append({"model": name, "n_features": n_feat,
                     **{f"valid_{k}": round(val, 5) for k, val in v.items()},
                     **{f"test_{k}": round(val, 5) for k, val in t.items()}})
    out = pd.DataFrame(rows)

    mean, lo, hi = paired_bootstrap_delta(y_te, test_scores["no_velocity"], test_scores["full"], n_boot)
    out["test_pr_auc_delta_vs_full"] = [0.0, round(mean, 5)]
    out["delta_ci_low"] = [np.nan, round(lo, 5)]
    out["delta_ci_high"] = [np.nan, round(hi, 5)]
    out["removed_features"] = ["", ", ".join(removed)]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--out", default="results/velocity_ablation.csv")
    ap.add_argument("--n-boot", type=int, default=200)
    args = ap.parse_args(argv)
    res = run_ablation(load_dataset(args.data), args.n_boot)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out, index=False)
    print(res.drop(columns=["removed_features"]).to_string(index=False))
    print(f"\nÇıxarılan feature-lər: {res.loc[1, 'removed_features']}\nYazıldı: {args.out}")


if __name__ == "__main__":
    main()
