"""Distillation: 3 students x 3 training arms x several seeds.

    python -m src.distill

Writes results/distill_runs.csv (one row per fit), results/distill_sq_errors.npy (the squared
error of every test row for every fit, in the same order as the CSV rows) and
results/distill_teacher.json. It uses LightGBM, so it must not load torch: no set_seed here.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor

from src.lib import (
    RESULTS_DIR,
    SPLIT_PATH,
    AverageEnsemble,
    EarlyStoppingLGBM,
    evaluate_regression,
    load_data,
    load_split,
    make_students,
    make_synthetic_rows,
)
from src.utils import load_config

ARMS = ["1 control", "2 distilled", "3 distilled + synthetic"]
OOF_PATH = SPLIT_PATH.parent / "teacher_oof.csv"


def build_teacher(seed):
    """The teacher: tuned LightGBM averaged with a 300-tree forest at its best max_features."""
    lgbm_params = json.loads((RESULTS_DIR / "best_lgbm.json").read_text())["params"]
    sweep = pd.read_csv(RESULTS_DIR / "error_correlation_sweep.csv")
    best_forest = sweep[sweep["bootstrap"]].sort_values("forest_rmse").iloc[0]
    # One thread on purpose: with several, the order the trees are summed in changes the last
    # float digits of the labels, and the MLPs trained on them are sensitive enough to notice.
    forest = RandomForestRegressor(
        n_estimators=300,
        max_features=int(best_forest["max_features"]),
        random_state=seed,
        n_jobs=1,
    )
    return AverageEnsemble([EarlyStoppingLGBM(lgbm_params, seed=seed), forest])


def fit_and_score(student, X_train, y_train, X_test, y_test, teacher_test_pred):
    """Fit a fresh copy of the student. Return its scores and the squared error of each test row."""
    t0 = time.perf_counter()
    model = clone(student).fit(X_train, y_train)
    train_seconds = time.perf_counter() - t0
    pred = model.predict(X_test)
    scores = {
        "train_rows": len(X_train),
        "test_rmse": evaluate_regression(y_test, pred)["rmse"],  # vs. the true targets
        "fidelity_rmse": float(np.sqrt(np.mean((pred - teacher_test_pred) ** 2))),  # vs. teacher
        "train_seconds": train_seconds,
    }
    return scores, (pred - y_test) ** 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-seeds", type=int, help="override distill.n_seeds (for a quick check)")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)

    cfg = load_config()
    dcfg = cfg["distill"]
    seeds = [cfg["seed"] + i for i in range(args.n_seeds or dcfg["n_seeds"])]

    X, y = load_data()
    X, y = X.to_numpy(), y.to_numpy()
    test_idx, folds = load_split()
    dev_idx = np.sort(np.concatenate([val for _, val in folds]))  # the full training portion
    X_dev, y_dev, X_test, y_test = X[dev_idx], y[dev_idx], X[test_idx], y[test_idx]

    # Out-of-fold teacher labels for the real training rows (made in the distillation notebook).
    oof = pd.read_csv(OOF_PATH).set_index("row")["teacher_oof"]
    oof_dev = oof.loc[dev_idx].to_numpy()

    teacher = build_teacher(cfg["seed"]).fit(X_dev, y_dev)  # one teacher for every seed
    teacher_test_pred = teacher.predict(X_test)
    teacher_test_rmse = evaluate_regression(y_test, teacher_test_pred)["rmse"]
    print(f"teacher test RMSE {teacher_test_rmse:.4f}", flush=True)

    feature_names = list(load_data()[0].columns)
    noise_frac = np.array(
        [
            dcfg["latlon_noise_frac"] if c in ("Latitude", "Longitude") else dcfg["noise_frac"]
            for c in feature_names
        ]
    )
    n_synth = dcfg["synthetic_ratio"] * len(X_dev)

    rows, sq_errors = [], []
    for seed in seeds:
        X_synth = make_synthetic_rows(X_dev, n_synth, noise_frac, seed)
        y_synth = teacher.predict(X_synth)
        arms = {
            ARMS[0]: (X_dev, y_dev),
            ARMS[1]: (X_dev, oof_dev),
            ARMS[2]: (np.vstack([X_dev, X_synth]), np.concatenate([oof_dev, y_synth])),
        }
        for student_name, student in make_students(dcfg, seed).items():
            for arm_name, (X_train, y_train) in arms.items():
                scores, sq = fit_and_score(
                    student, X_train, y_train, X_test, y_test, teacher_test_pred
                )
                rows.append({"student": student_name, "arm": arm_name, "seed": seed, **scores})
                sq_errors.append(sq)
        print(f"seed {seed} done ({len(rows)} fits)", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_dir / "distill_runs.csv", index=False)
    np.save(args.out_dir / "distill_sq_errors.npy", np.array(sq_errors))
    (args.out_dir / "distill_teacher.json").write_text(
        json.dumps({"teacher_test_rmse": teacher_test_rmse}, indent=2) + "\n"
    )
    print(f"wrote {len(rows)} fits to {args.out_dir}")


if __name__ == "__main__":
    main()
