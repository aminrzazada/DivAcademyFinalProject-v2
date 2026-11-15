from scripts.velocity_ablation import paired_bootstrap_delta, run_ablation


def test_velocity_features_are_removed(df):
    res = run_ablation(df, n_boot=10)
    assert list(res["model"]) == ["full", "no_velocity"]
    assert res.loc[1, "n_features"] < res.loc[0, "n_features"]
    removed = res.loc[1, "removed_features"].split(", ")
    assert removed and all("velocity" in c for c in removed)


def test_bootstrap_identical_scores_gives_zero_delta():
    import numpy as np
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.1).astype(int)
    p = rng.random(2000)
    mean, lo, hi = paired_bootstrap_delta(y, p, p, n_boot=20)
    assert mean == 0 and lo == 0 and hi == 0
