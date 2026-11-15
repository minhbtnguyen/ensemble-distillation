import numpy as np

from src.tune import best_trial, run_search


def _sample(rng):
    return {"a": int(rng.integers(0, 10)), "b": float(rng.uniform())}


def _score(params):
    return float(params["b"]), 0.01


def test_search_stops_at_n_trials():
    log = run_search(_sample, _score, n_trials=4, time_cap_seconds=1e9, seed=0)
    assert len(log) == 4
    assert list(log["trial"]) == [1, 2, 3, 4]


def test_search_stops_when_time_cap_is_hit():
    log = run_search(_sample, _score, n_trials=50, time_cap_seconds=0, seed=0)
    assert len(log) == 1  # the cap is checked after each trial


def test_search_is_deterministic_and_has_log_columns():
    a = run_search(_sample, _score, n_trials=3, time_cap_seconds=1e9, seed=7)
    b = run_search(_sample, _score, n_trials=3, time_cap_seconds=1e9, seed=7)
    assert np.array_equal(a["cv_rmse_mean"], b["cv_rmse_mean"])
    for col in ["trial", "cv_rmse_mean", "cv_rmse_std", "trial_seconds", "cumulative_seconds"]:
        assert col in a.columns


def test_best_trial_keeps_int_params():
    log = run_search(_sample, _score, n_trials=5, time_cap_seconds=1e9, seed=1)
    best = best_trial(log)
    assert isinstance(best["params"]["a"], int)
    assert best["cv_rmse_mean"] == log["cv_rmse_mean"].min()
