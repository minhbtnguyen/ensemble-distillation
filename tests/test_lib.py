import json
import subprocess
import sys

import joblib
import numpy as np
import pytest
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from src.lib import (
    ROOT,
    TEST_FOLD,
    AverageEnsemble,
    build_pipeline,
    evaluate_regression,
    load_split,
    make_split,
    make_students,
    make_synthetic_rows,
    out_of_fold_predictions,
    paired_bootstrap_rmse,
    save_split,
)
from src.utils import set_seed

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


def _split_and_tiny_training(seed):
    """Make the split, then train a tiny torch model for a few steps. Returns both results."""
    set_seed(seed)
    split = make_split(N, seed, TEST_SIZE, N_FOLDS)

    X = torch.randn(N, 4)
    y = X @ torch.tensor([1.0, -2.0, 0.5, 3.0]) + 0.1 * torch.randn(N)
    model = torch.nn.Linear(4, 1)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(20):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(X).squeeze(1), y)
        loss.backward()
        opt.step()
    return split, model(X).squeeze(1).detach().numpy()


def test_same_seed_gives_identical_split_and_predictions():
    split_a, pred_a = _split_and_tiny_training(SEED)
    split_b, pred_b = _split_and_tiny_training(SEED)
    assert split_a.equals(split_b)
    assert np.array_equal(pred_a, pred_b)


def test_different_seed_changes_results():
    split_a, pred_a = _split_and_tiny_training(SEED)
    split_b, pred_b = _split_and_tiny_training(SEED + 1)
    assert not split_a.equals(split_b)
    assert not np.array_equal(pred_a, pred_b)


def test_average_ensemble_averages_and_does_not_touch_the_originals():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = X @ np.array([1.0, -1.0, 2.0]) + rng.normal(0, 0.1, size=200)
    ridge, tree = Ridge(), DecisionTreeRegressor(max_depth=3, random_state=0)

    ens = AverageEnsemble([ridge, tree]).fit(X, y)
    expected = (
        Ridge().fit(X, y).predict(X)
        + DecisionTreeRegressor(max_depth=3, random_state=0).fit(X, y).predict(X)
    ) / 2
    assert np.allclose(ens.predict(X), expected)
    assert not hasattr(ridge, "coef_")  # fit works on clones, the inputs stay unfitted


def test_average_ensemble_round_trips_through_joblib(tmp_path):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 3))
    y = X[:, 0] + rng.normal(0, 0.1, size=100)
    ens = AverageEnsemble([Ridge(), DecisionTreeRegressor(max_depth=2, random_state=0)]).fit(X, y)
    joblib.dump(ens, tmp_path / "ens.joblib")
    assert np.array_equal(joblib.load(tmp_path / "ens.joblib").predict(X), ens.predict(X))


def test_early_stopping_lgbm_fits_and_round_trips(tmp_path):
    # Run in a fresh process: LightGBM and torch (loaded by other tests here) can clash.
    code = f"""
import joblib, numpy as np
from src.lib import EarlyStoppingLGBM
rng = np.random.default_rng(0)
X = rng.normal(size=(400, 3)); y = X[:, 0] + rng.normal(0, 0.1, size=400)
m = EarlyStoppingLGBM({{"n_estimators": 200, "learning_rate": 0.1}}, seed=0).fit(X, y)
joblib.dump(m, r"{tmp_path / "m.joblib"}")
assert np.array_equal(joblib.load(r"{tmp_path / "m.joblib"}").predict(X), m.predict(X))
assert np.sqrt(np.mean((m.predict(X) - y) ** 2)) < 0.5
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr


def test_out_of_fold_predictions_cover_training_rows_only_and_never_see_their_row(tmp_path):
    rng = np.random.default_rng(2)
    X = rng.normal(size=(N, 3))
    y = X @ np.array([1.0, 2.0, -1.0]) + rng.normal(0, 0.5, size=N)
    path = tmp_path / "split.csv"
    save_split(make_split(N, SEED, TEST_SIZE, N_FOLDS), path)
    test_idx, folds = load_split(path)

    oof = out_of_fold_predictions(Ridge(), X, y, folds)
    assert np.isnan(oof[test_idx]).all()  # test rows get no label
    assert not np.isnan(np.delete(oof, test_idx)).any()  # every training row has one

    train_idx, val_idx = folds[0]  # the first fold's rows come from a model that excluded them
    expected = Ridge().fit(X[train_idx], y[train_idx]).predict(X[val_idx])
    assert np.allclose(oof[val_idx], expected)


def test_synthetic_rows_are_deterministic_in_range_and_jittered():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(300, 4)) * [1.0, 5.0, 0.1, 20.0]
    a = make_synthetic_rows(X, 1500, noise_frac=0.05, seed=9)
    b = make_synthetic_rows(X, 1500, noise_frac=0.05, seed=9)
    assert a.shape == (1500, 4)
    assert np.array_equal(a, b)
    assert (a >= X.min(axis=0)).all() and (a <= X.max(axis=0)).all()
    real_rows = {tuple(row) for row in X}
    assert not any(tuple(row) in real_rows for row in a)  # jittered copies, not real rows
    assert not np.array_equal(a, make_synthetic_rows(X, 1500, noise_frac=0.05, seed=10))


def test_synthetic_noise_size_follows_noise_frac():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(2000, 2)) * [1.0, 10.0]
    small = make_synthetic_rows(X, 20000, noise_frac=[0.0, 0.05], seed=0)
    # first column has no noise, so its spread matches the data; second has jitter added
    assert abs(small[:, 0].std() - X[:, 0].std()) < 0.05
    assert small[:, 1].std() > X[:, 1].std()


def test_students_fit_and_predict_in_the_target_scale():
    cfg = {
        "student_widths": [8, 16],
        "student_tree_depth": 4,
        "student_mlp": {"learning_rate_init": 0.01, "alpha": 0.0001, "max_iter": 200},
    }
    rng = np.random.default_rng(5)
    X = rng.normal(5.0, 3.0, size=(400, 3))
    y = 100.0 + X @ np.array([1.0, -2.0, 0.5])  # a target far from zero and not unit scale

    students = make_students(cfg, seed=0)
    assert list(students) == ["MLP 2x8", "MLP 2x16", "Tree (depth 4)"]
    for name, student in students.items():
        pred = student.fit(X, y).predict(X)  # one fit and one predict call
        assert pred.shape == y.shape, name
        assert abs(pred.mean() - y.mean()) < 5.0, (
            name
        )  # predictions come back in the target's units
        assert np.sqrt(np.mean((pred - y) ** 2)) < y.std(), name  # better than predicting the mean


def test_paired_bootstrap_uses_the_same_draws_and_centers_on_the_full_sample_rmse():
    rng = np.random.default_rng(6)
    sq_a = rng.exponential(1.0, size=(3, 500))  # 3 seeds, 500 test rows
    sq_teacher = rng.exponential(0.5, size=500)
    out = paired_bootstrap_rmse({"a": sq_a, "a_again": sq_a.copy()}, sq_teacher, n_boot=300, seed=1)

    assert set(out) == {"a", "a_again", "teacher"}
    assert all(len(v) == 300 for v in out.values())
    assert np.array_equal(out["a"], out["a_again"])  # identical models get identical draws
    full_sample = np.sqrt(sq_a.mean(axis=1)).mean()
    assert abs(out["a"].mean() - full_sample) < 0.02
    again = paired_bootstrap_rmse({"a": sq_a}, sq_teacher, n_boot=300, seed=1)
    assert np.array_equal(again["a"], out["a"])  # deterministic for a seed


ARTIFACTS = sorted((ROOT / "artifacts").glob("*.joblib"))
NEEDS_ARTIFACTS = pytest.mark.skipif(
    not ARTIFACTS, reason="run python -m src.build_artifacts first"
)


@NEEDS_ARTIFACTS
@pytest.mark.parametrize("path", ARTIFACTS, ids=lambda p: p.stem)
def test_reloaded_artifact_matches_the_in_memory_model(path, tmp_path):
    """Training/serving skew check: a fresh process loads the file and predicts 100 saved rows.

    The predictions must be identical to the ones the in-memory model made when it was saved.
    """
    rows = path.parent / "check_rows.npy"
    expected = path.with_name(f"{path.stem}.check_pred.npy")
    out = tmp_path / "pred.npy"
    code = (
        "import joblib, numpy as np; "
        f"m = joblib.load(r'{path}'); np.save(r'{out}', m.predict(np.load(r'{rows}')))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300, cwd=ROOT
    )
    assert result.returncode == 0, result.stderr
    assert np.array_equal(np.load(out), np.load(expected))


@NEEDS_ARTIFACTS
@pytest.mark.parametrize("path", ARTIFACTS, ids=lambda p: p.stem)
def test_model_card_describes_the_artifact(path):
    import hashlib

    card = json.loads(path.with_name(f"{path.stem}.model_card.json").read_text())
    for key in ("git_commit", "trained_on", "test_metrics", "feature_ranges", "sha256", "versions"):
        assert key in card, key
    assert set(card["test_metrics"]) == {"rmse", "mae", "r2"}
    assert len(card["feature_ranges"]) == 8
    assert all(lo <= hi for lo, hi in card["feature_ranges"].values())
    assert (
        card["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    )  # the card matches the file
