"""Fit the teacher and each student's best arm on the full training portion and save them.

    python -m src.build_artifacts

Each model is saved with joblib as one object in artifacts/ (scaler, network and target scaling
included), next to a model card and a small serialization check:

    <name>.joblib           the model (git-ignored, the teacher is 418 MB)
    <name>.model_card.json  git commit, date, test metrics, feature ranges, file hash, versions
    <name>.check_pred.npy   the in-memory model's predictions on check_rows.npy
    check_rows.npy          100 test rows, used to check that a reloaded model agrees

The best arm per student is the one with the lowest mean test RMSE in results/distill_runs.csv.
It uses LightGBM, so it must not load torch: no set_seed here.
"""

import datetime
import hashlib
import json
import re
import subprocess
import sys

import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone

from src.bench import count_parameters
from src.distill import OOF_PATH, build_teacher, make_arms
from src.lib import (
    RESULTS_DIR,
    ROOT,
    evaluate_regression,
    load_data,
    load_split,
    make_students,
)
from src.utils import load_config

ARTIFACT_DIR = ROOT / "artifacts"
N_CHECK_ROWS = 100


def artifact_name(model_name):
    """'MLP 2x32' -> 'mlp_2x32', 'Tree (depth 6)' -> 'tree_depth_6'."""
    return re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")


def best_arms(runs):
    """For each student, the arm with the lowest mean test RMSE over the seeds."""
    mean = runs.groupby(["student", "arm"])["test_rmse"].mean()
    return {student: mean.loc[student].idxmin() for student in runs["student"].unique()}


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state():
    """The current commit, and whether any tracked or new file outside artifacts/ has changed."""

    def run(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True, cwd=ROOT).stdout

    try:
        commit = run("rev-parse", "HEAD").strip() or "unknown"
        dirty = bool(run("status", "--porcelain", "--", ".", ":!artifacts").strip())
    except OSError:
        commit, dirty = "unknown", False
    return commit, dirty


def make_model_card(name, arm, model, path, n_train_rows, feature_ranges, test_metrics):
    commit, dirty = git_state()
    return {
        "model": name,
        "artifact": path.name,
        "sha256": sha256_of(path),
        "size_bytes": path.stat().st_size,
        "parameters": int(count_parameters(model)),
        "arm": arm,
        "training_rows": int(n_train_rows),
        "trained_on": datetime.datetime.now(datetime.UTC).date().isoformat(),
        "git_commit": commit,
        "git_dirty": dirty,
        "test_metrics": test_metrics,
        "feature_ranges": feature_ranges,
        "versions": {
            "python": sys.version.split()[0],
            "scikit-learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
            "numpy": np.__version__,
        },
    }


def main():
    cfg = load_config()
    dcfg = cfg["distill"]
    seed = cfg["seed"]

    X, y = load_data()
    feature_names = list(X.columns)
    X, y = X.to_numpy(), y.to_numpy()
    test_idx, folds = load_split()
    dev_idx = np.sort(np.concatenate([val for _, val in folds]))
    X_dev, y_dev, X_test, y_test = X[dev_idx], y[dev_idx], X[test_idx], y[test_idx]
    oof_dev = pd.read_csv(OOF_PATH).set_index("row")["teacher_oof"].loc[dev_idx].to_numpy()
    feature_ranges = {
        name: [float(lo), float(hi)]
        for name, lo, hi in zip(feature_names, X_dev.min(axis=0), X_dev.max(axis=0), strict=True)
    }

    teacher = build_teacher(seed).fit(X_dev, y_dev)
    arms = make_arms(X_dev, y_dev, oof_dev, teacher, dcfg, feature_names, seed)
    choice = best_arms(pd.read_csv(RESULTS_DIR / "distill_runs.csv"))

    models = [("Teacher (LightGBM + forest)", "-", teacher, len(X_dev))]
    for name, student in make_students(dcfg, seed).items():
        X_train, y_train = arms[choice[name]]
        models.append((name, choice[name], clone(student).fit(X_train, y_train), len(X_train)))

    ARTIFACT_DIR.mkdir(exist_ok=True)
    check_rows = X_test[:N_CHECK_ROWS]
    np.save(ARTIFACT_DIR / "check_rows.npy", check_rows)
    for name, arm, model, n_rows in models:
        stem = "teacher" if name.startswith("Teacher") else artifact_name(name)
        path = ARTIFACT_DIR / f"{stem}.joblib"
        joblib.dump(model, path)
        np.save(ARTIFACT_DIR / f"{stem}.check_pred.npy", model.predict(check_rows))
        card = make_model_card(
            name,
            arm,
            model,
            path,
            n_rows,
            feature_ranges,
            evaluate_regression(y_test, model.predict(X_test)),
        )
        (ARTIFACT_DIR / f"{stem}.model_card.json").write_text(json.dumps(card, indent=2) + "\n")
        print(f"{name} (arm {arm}) -> {path.name}, model card, check predictions")


if __name__ == "__main__":
    main()
