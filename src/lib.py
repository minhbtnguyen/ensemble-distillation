from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
SPLIT_PATH = ROOT / "data" / "split.csv"
RESULTS_DIR = ROOT / "results"
TEST_FOLD = -1  # fold label for rows in the frozen test set


def load_data():
    """Return California Housing as (X, y), with y in units of $100k."""
    return fetch_california_housing(return_X_y=True, as_frame=True)


def make_split(n_rows, seed, test_size, n_folds):
    """Assign every row to the test set (fold -1) or to one of n_folds CV folds."""
    all_idx = np.arange(n_rows)
    dev_idx, test_idx = train_test_split(all_idx, test_size=test_size, random_state=seed)
    fold = np.full(n_rows, TEST_FOLD)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, val_pos) in enumerate(kf.split(dev_idx)):
        fold[dev_idx[val_pos]] = k
    return pd.DataFrame({"row": all_idx, "fold": fold})


def save_split(split, path=SPLIT_PATH):
    """Write the split to disk. Same input always gives a byte-identical file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(path, index=False)


def load_split(path=SPLIT_PATH):
    """Return (test_idx, folds). folds is a list of (train_idx, val_idx), one per CV fold."""
    split = pd.read_csv(path)
    test_idx = split.loc[split["fold"] == TEST_FOLD, "row"].to_numpy()
    folds = []
    for k in sorted(split.loc[split["fold"] != TEST_FOLD, "fold"].unique()):
        val = split.loc[split["fold"] == k, "row"].to_numpy()
        train = split.loc[~split["fold"].isin([TEST_FOLD, k]), "row"].to_numpy()
        folds.append((train, val))
    return test_idx, folds


def evaluate_regression(y_true, y_pred):
    """Return RMSE, MAE and R² as a dict."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    ss_res = np.sum(err**2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "r2": float(1 - ss_res / ss_tot),
    }


def build_pipeline(model, scale=True):
    """Wrap a model with a StandardScaler, or return it bare for tree models (scale=False).

    Inside cross-validation the scaler is refit on each training fold only.
    """
    if not scale:
        return model
    return Pipeline([("scale", StandardScaler()), ("model", model)])


class EarlyStoppingLGBM(BaseEstimator, RegressorMixin):
    """LightGBM that picks its number of rounds by early stopping on a slice of its own data.

    This is the same procedure the search in src/tune.py used, so tuned parameters carry over.
    """

    def __init__(self, params=None, seed=42, stop_fraction=0.1, patience=50):
        self.params = params
        self.seed = seed
        self.stop_fraction = stop_fraction
        self.patience = patience

    def fit(self, X, y):
        import lightgbm as lgb  # imported here so importing src.lib never loads lightgbm

        X, y = np.asarray(X), np.asarray(y)
        fit_idx, stop_idx = train_test_split(
            np.arange(len(y)), test_size=self.stop_fraction, random_state=self.seed
        )
        self.model_ = lgb.LGBMRegressor(
            **(self.params or {}), subsample_freq=1, random_state=self.seed, verbose=-1
        )
        self.model_.fit(
            X[fit_idx],
            y[fit_idx],
            eval_X=X[stop_idx],
            eval_y=y[stop_idx],
            callbacks=[lgb.early_stopping(self.patience, verbose=False)],
        )
        return self

    def predict(self, X):
        return self.model_.predict(np.asarray(X))


class AverageEnsemble(BaseEstimator, RegressorMixin):
    """Average the predictions of several regressors. Save the whole thing with joblib."""

    def __init__(self, estimators):
        self.estimators = estimators

    def fit(self, X, y):
        self.fitted_ = [clone(est).fit(X, y) for est in self.estimators]
        return self

    def predict(self, X):
        return np.mean([est.predict(X) for est in self.fitted_], axis=0)


def out_of_fold_predictions(model, X, y, folds):
    """Predict every training row with a model that never saw it.

    For each fold, fit a clone of `model` on the other folds and predict the held-out fold.
    Returns an array as long as X. Rows in no fold (the test rows) stay NaN.
    """
    oof = np.full(len(y), np.nan)
    for train_idx, val_idx in folds:
        oof[val_idx] = clone(model).fit(X[train_idx], y[train_idx]).predict(X[val_idx])
    return oof


def make_synthetic_rows(X, n_rows, noise_frac, seed):
    """Make extra feature rows: resample real rows and jitter every feature a little.

    noise_frac is the jitter size as a fraction of each feature's standard deviation (one number,
    or one per column). Results are clipped to each feature's real min and max. The rows have no
    labels: label them with the teacher.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    parents = X[rng.integers(0, len(X), size=n_rows)]
    noise = rng.normal(size=parents.shape) * X.std(axis=0) * np.asarray(noise_frac)
    return np.clip(parents + noise, X.min(axis=0), X.max(axis=0))
