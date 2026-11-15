"""Helpers for measuring size and speed: parameter counts, single-row latency, model-only calls."""

import time

import numpy as np

from src.lib import AverageEnsemble, EarlyStoppingLGBM


def count_parameters(model):
    """Weights and biases for a network, nodes for a tree, and the sum of both for the teacher."""
    if isinstance(model, AverageEnsemble):
        return sum(count_parameters(m) for m in model.fitted_)
    if isinstance(model, EarlyStoppingLGBM):
        return len(model.model_.booster_.trees_to_dataframe())  # one row per tree node
    if hasattr(model, "regressor_"):  # TransformedTargetRegressor
        return count_parameters(model.regressor_)
    if hasattr(model, "steps"):  # Pipeline: the last step is the model
        return count_parameters(model.steps[-1][1])
    if hasattr(model, "coefs_"):  # MLPRegressor
        return sum(w.size for w in model.coefs_) + sum(b.size for b in model.intercepts_)
    if hasattr(model, "estimators_"):  # forest
        return sum(t.tree_.node_count for t in model.estimators_)
    return model.tree_.node_count  # decision tree


def pin_threads(model):
    """Make the teacher's LightGBM and forest use one thread (students need no change)."""
    if isinstance(model, AverageEnsemble):
        for part in model.fitted_:
            if isinstance(part, EarlyStoppingLGBM):
                part.model_.set_params(n_jobs=1)
            else:
                part.set_params(n_jobs=1)
    return model


def mlp_forward(student):
    """The MLP student's whole prediction in plain numpy, with no sklearn input checks.

    Scale the row, run the layers, undo the target scaling. This is what the model itself has to
    compute, so timing it separates the model from Python and sklearn overhead.
    """
    x_scaler = student.regressor_.steps[0][1]
    net = student.regressor_.steps[-1][1]
    y_scaler = student.transformer_
    weights, biases = net.coefs_, net.intercepts_

    def predict(x):
        h = (x - x_scaler.mean_) / x_scaler.scale_
        for w, b in zip(weights[:-1], biases[:-1], strict=True):
            h = np.maximum(h @ w + b, 0.0)
        out = h @ weights[-1] + biases[-1]
        return out.ravel() * y_scaler.scale_ + y_scaler.mean_

    return predict


def single_row_latency(predict_one, rows, warmup=1000, n_timed=5000):
    """Time `predict_one(row)` one call at a time. Returns p50 and p99 in microseconds.

    `rows` is a list of ready-made single-row inputs, so building them is not timed. The calls
    are cycled through the rows, and the first `warmup` calls are thrown away.
    """
    for i in range(warmup):
        predict_one(rows[i % len(rows)])
    times = np.empty(n_timed)
    for i in range(n_timed):
        row = rows[i % len(rows)]
        t0 = time.perf_counter_ns()
        predict_one(row)
        times[i] = time.perf_counter_ns() - t0
    p50, p99 = np.percentile(times, [50, 99]) / 1e3
    return {"p50_us": float(p50), "p99_us": float(p99)}
