"""Fit the teacher and each student's best arm on the full training portion and save them.

    python -m src.build_artifacts

Each model is saved with joblib as one object in artifacts/ (scaler, network and target scaling
included). The best arm per student is the one with the lowest mean test RMSE in
results/distill_runs.csv. It uses LightGBM, so it must not load torch: no set_seed here.
"""

import re

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from src.distill import OOF_PATH, build_teacher, make_arms
from src.lib import RESULTS_DIR, ROOT, load_data, load_split, make_students
from src.utils import load_config

ARTIFACT_DIR = ROOT / "artifacts"


def artifact_name(model_name):
    """'MLP 2x32' -> 'mlp_2x32', 'Tree (depth 6)' -> 'tree_depth_6'."""
    return re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")


def best_arms(runs):
    """For each student, the arm with the lowest mean test RMSE over the seeds."""
    mean = runs.groupby(["student", "arm"])["test_rmse"].mean()
    return {student: mean.loc[student].idxmin() for student in runs["student"].unique()}


def main():
    cfg = load_config()
    dcfg = cfg["distill"]
    seed = cfg["seed"]

    X, y = load_data()
    feature_names = list(X.columns)
    X, y = X.to_numpy(), y.to_numpy()
    _test_idx, folds = load_split()
    dev_idx = np.sort(np.concatenate([val for _, val in folds]))
    X_dev, y_dev = X[dev_idx], y[dev_idx]
    oof_dev = pd.read_csv(OOF_PATH).set_index("row")["teacher_oof"].loc[dev_idx].to_numpy()

    teacher = build_teacher(seed).fit(X_dev, y_dev)
    arms = make_arms(X_dev, y_dev, oof_dev, teacher, dcfg, feature_names, seed)
    choice = best_arms(pd.read_csv(RESULTS_DIR / "distill_runs.csv"))

    ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump(teacher, ARTIFACT_DIR / "teacher.joblib")
    print("teacher -> teacher.joblib")
    for name, student in make_students(dcfg, seed).items():
        arm = choice[name]
        X_train, y_train = arms[arm]
        joblib.dump(
            clone(student).fit(X_train, y_train), ARTIFACT_DIR / f"{artifact_name(name)}.joblib"
        )
        print(f"{name} (arm {arm}) -> {artifact_name(name)}.joblib")


if __name__ == "__main__":
    main()
