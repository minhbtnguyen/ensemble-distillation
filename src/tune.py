"""Random search for LightGBM and the MLP under the same budget.

Run one model per process. LightGBM and PyTorch each bring their own OpenMP runtime, and
loading both in one process can freeze whichever trains second. So torch is imported only
when the MLP is used, and lightgbm only when LightGBM is used:

    python -m src.tune --model lgbm
    python -m src.tune --model mlp
"""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.lib import RESULTS_DIR, evaluate_regression, load_data, load_split
from src.utils import load_config, set_seed

MAX_EPOCHS, PATIENCE, BATCH_SIZE = 100, 10, 256
LOG_COLUMNS = ["trial", "cv_rmse_mean", "cv_rmse_std", "trial_seconds", "cumulative_seconds"]


def run_search(sample_params, cv_rmse, n_trials, time_cap_seconds, seed):
    """Random search. Stops after n_trials, or after the trial that crosses the time cap."""
    rng = np.random.default_rng(seed)
    rows = []
    start = time.perf_counter()
    for trial in range(1, n_trials + 1):
        t0 = time.perf_counter()
        params = sample_params(rng)
        cv_mean, cv_std = cv_rmse(params)
        now = time.perf_counter()
        rows.append(
            {
                "trial": trial,
                **params,
                "cv_rmse_mean": cv_mean,
                "cv_rmse_std": cv_std,
                "trial_seconds": now - t0,
                "cumulative_seconds": now - start,
            }
        )
        if now - start >= time_cap_seconds:
            break
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- LightGBM


def sample_lgbm_params(rng):
    """Draw one random setting. Rates and penalties are sampled on a log scale."""
    return {
        "learning_rate": float(np.exp(rng.uniform(np.log(0.01), np.log(0.2)))),
        "num_leaves": int(rng.integers(8, 129)),
        "n_estimators": int(np.exp(rng.uniform(np.log(200), np.log(2000)))),  # max rounds
        "min_child_samples": int(rng.integers(5, 101)),
        "subsample": float(rng.uniform(0.5, 1.0)),  # row subsampling
        "colsample_bytree": float(rng.uniform(0.5, 1.0)),  # column subsampling
        "reg_lambda": float(np.exp(rng.uniform(np.log(1e-3), np.log(10)))),  # L2 penalty
    }


def fit_predict_lgbm(params, X, y, fit_idx, stop_idx, pred_idx, seed):
    """Train on fit_idx, stop early on stop_idx, return predictions for pred_idx."""
    import lightgbm as lgb  # imported here so the MLP process never loads it

    model = lgb.LGBMRegressor(**params, subsample_freq=1, random_state=seed, verbose=-1)
    model.fit(
        X[fit_idx],
        y[fit_idx],
        eval_X=X[stop_idx],
        eval_y=y[stop_idx],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return model.predict(X[pred_idx])  # uses the best round found by early stopping


def cv_rmse_lgbm(params, X, y, folds, seed):
    """5-fold CV RMSE for one setting: (mean, std) over the folds."""
    scores = []
    for train_idx, val_idx in folds:
        fit_idx, stop_idx = train_test_split(train_idx, test_size=0.1, random_state=seed)
        pred = fit_predict_lgbm(params, X, y, fit_idx, stop_idx, val_idx, seed)
        scores.append(evaluate_regression(y[val_idx], pred)["rmse"])
    return float(np.mean(scores)), float(np.std(scores, ddof=1))


# --------------------------------------------------------------------- MLP


def sample_mlp_params(rng):
    """Draw one random setting. Rates and penalties are sampled on a log scale."""
    return {
        "width": int(rng.choice([32, 64, 128, 256])),
        "depth": int(rng.integers(1, 5)),  # hidden layers
        "learning_rate": float(np.exp(rng.uniform(np.log(1e-4), np.log(1e-2)))),
        "weight_decay": float(np.exp(rng.uniform(np.log(1e-6), np.log(1e-2)))),
        "dropout": float(rng.uniform(0.0, 0.3)),
    }


def build_mlp(n_in, width, depth, dropout):
    from torch import nn

    layers, size = [], n_in
    for _ in range(depth):
        layers += [nn.Linear(size, width), nn.ReLU(), nn.Dropout(dropout)]
        size = width
    layers.append(nn.Linear(size, 1))
    return nn.Sequential(*layers)


def fit_predict_mlp(params, X, y, fit_idx, stop_idx, val_idx, seed):
    """Train on fit_idx, stop early on stop_idx, return predictions for val_idx in $100k."""
    import torch
    from torch import nn

    torch.set_num_threads(1)  # one thread is plenty for this small net, and avoids OpenMP clashes
    set_seed(seed)
    x_scaler = StandardScaler().fit(X[fit_idx])
    y_mean, y_std = y[fit_idx].mean(), y[fit_idx].std()

    def to_x(idx):
        return torch.tensor(x_scaler.transform(X[idx]), dtype=torch.float32)

    def to_y(idx):
        return torch.tensor((y[idx] - y_mean) / y_std, dtype=torch.float32)

    x_fit, y_fit, x_stop, y_stop = to_x(fit_idx), to_y(fit_idx), to_x(stop_idx), to_y(stop_idx)
    model = build_mlp(X.shape[1], params["width"], params["depth"], params["dropout"])
    opt = torch.optim.Adam(
        model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"]
    )
    loss_fn = nn.MSELoss()

    best_loss, best_state, bad_epochs = float("inf"), None, 0
    for _ in range(MAX_EPOCHS):
        model.train()
        order = torch.randperm(len(x_fit))
        for i in range(0, len(order), BATCH_SIZE):
            batch = order[i : i + BATCH_SIZE]
            opt.zero_grad()
            loss_fn(model(x_fit[batch]).squeeze(1), y_fit[batch]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            stop_loss = loss_fn(model(x_stop).squeeze(1), y_stop).item()
        if stop_loss < best_loss - 1e-6:
            best_loss, best_state, bad_epochs = stop_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(to_x(val_idx)).squeeze(1).numpy()
    return pred * y_std + y_mean  # undo the target scaling


def cv_rmse_mlp(params, X, y, folds, seed):
    """5-fold CV RMSE for one setting: (mean, std) over the folds."""
    scores = []
    for train_idx, val_idx in folds:
        fit_idx, stop_idx = train_test_split(train_idx, test_size=0.1, random_state=seed)
        pred = fit_predict_mlp(params, X, y, fit_idx, stop_idx, val_idx, seed)
        scores.append(evaluate_regression(y[val_idx], pred)["rmse"])
    return float(np.mean(scores)), float(np.std(scores, ddof=1))


# --------------------------------------------------------------------- CLI

MODELS = {"lgbm": (sample_lgbm_params, cv_rmse_lgbm), "mlp": (sample_mlp_params, cv_rmse_mlp)}


def best_trial(log):
    """Return the best trial's score and parameters as a plain dict (ints stay ints)."""
    i = log["cv_rmse_mean"].idxmin()
    params = {c: log[c].iloc[i].item() for c in log.columns if c not in LOG_COLUMNS}
    return {
        "trial": int(log["trial"].iloc[i]),
        "cv_rmse_mean": float(log["cv_rmse_mean"].iloc[i]),
        "cv_rmse_std": float(log["cv_rmse_std"].iloc[i]),
        "params": params,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--n-trials", type=int, help="override tune.n_trials (for a tiny check)")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)

    cfg = load_config()
    seed = cfg["seed"]
    n_trials = args.n_trials or cfg["tune"]["n_trials"]
    cap = cfg["tune"]["time_cap_seconds"]

    X, y = load_data()
    X, y = X.to_numpy(), y.to_numpy()
    _test_idx, folds = load_split()  # test rows are never used while tuning

    sample_params, cv_rmse = MODELS[args.model]
    print(f"{args.model}: up to {n_trials} trials or {cap} s, whichever comes first", flush=True)
    log = run_search(sample_params, lambda p: cv_rmse(p, X, y, folds, seed), n_trials, cap, seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log.to_csv(args.out_dir / f"trials_{args.model}.csv", index=False)
    best = best_trial(log)
    (args.out_dir / f"best_{args.model}.json").write_text(json.dumps(best, indent=2) + "\n")
    print(
        f"{args.model}: {len(log)} trials in {log['cumulative_seconds'].iloc[-1]:.0f} s, "
        f"best CV RMSE {best['cv_rmse_mean']:.4f} (trial {best['trial']})"
    )


if __name__ == "__main__":
    main()
