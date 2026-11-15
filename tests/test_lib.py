import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from src.lib import (
    TEST_FOLD,
    build_pipeline,
    evaluate_regression,
    load_split,
    make_split,
    save_split,
)

N, SEED, TEST_SIZE, N_FOLDS = 1000, 42, 0.2, 5


def test_split_is_deterministic(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    save_split(make_split(N, SEED, TEST_SIZE, N_FOLDS), a)
    save_split(make_split(N, SEED, TEST_SIZE, N_FOLDS), b)
    assert a.read_bytes() == b.read_bytes()


def test_split_structure(tmp_path):
    path = tmp_path / "split.csv"
    save_split(make_split(N, SEED, TEST_SIZE, N_FOLDS), path)
    test_idx, folds = load_split(path)

    assert len(test_idx) == int(N * TEST_SIZE)
    assert len(folds) == N_FOLDS

    val_all = np.concatenate([val for _, val in folds])
    dev_idx = np.setdiff1d(np.arange(N), test_idx)
    assert np.array_equal(np.sort(val_all), dev_idx)  # folds cover dev data exactly once
    for train, val in folds:
        assert not set(train) & set(val)
        assert not set(train) & set(test_idx)
        assert not set(val) & set(test_idx)


def test_test_rows_are_labeled():
    split = make_split(N, SEED, TEST_SIZE, N_FOLDS)
    assert (split["fold"] == TEST_FOLD).sum() == int(N * TEST_SIZE)


def test_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    y_true = rng.normal(2.0, 1.2, size=500)
    y_pred = y_true + rng.normal(0, 0.5, size=500)
    m = evaluate_regression(y_true, y_pred)
    assert abs(m["rmse"] - np.sqrt(mean_squared_error(y_true, y_pred))) < 1e-9
    assert abs(m["mae"] - mean_absolute_error(y_true, y_pred)) < 1e-9
    assert abs(m["r2"] - r2_score(y_true, y_pred)) < 1e-9


def test_pipeline_scaler_is_fit_per_training_fold(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(5.0, 3.0, size=(N, 4))
    y = X @ np.array([1.0, -2.0, 0.5, 3.0]) + rng.normal(0, 0.1, size=N)

    path = tmp_path / "split.csv"
    save_split(make_split(N, SEED, TEST_SIZE, N_FOLDS), path)
    _, folds = load_split(path)

    res = cross_validate(build_pipeline(Ridge()), X, y, cv=folds, return_estimator=True)
    means = []
    for (train, _val), est in zip(folds, res["estimator"], strict=True):
        scaler_mean = est.named_steps["scale"].mean_
        assert np.allclose(scaler_mean, X[train].mean(axis=0))  # fit on this fold's training rows
        means.append(scaler_mean)
    assert not np.allclose(means[0], means[1])  # and refit for every fold


def test_trees_go_in_bare():
    tree = DecisionTreeRegressor()
    assert build_pipeline(tree, scale=False) is tree
    assert isinstance(build_pipeline(Ridge()), Pipeline)
