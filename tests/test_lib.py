import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.lib import TEST_FOLD, evaluate_regression, load_split, make_split, save_split

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
