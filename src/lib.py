from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import KFold, train_test_split

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
