import numpy as np
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor

from src.bench import count_parameters, mlp_forward, single_row_latency
from src.build_artifacts import artifact_name, best_arms
from src.lib import make_students
from src.measure_memory import measure


def _data(n=400):
    rng = np.random.default_rng(0)
    X = rng.normal(5.0, 3.0, size=(n, 3))
    y = 100.0 + X @ np.array([1.0, -2.0, 0.5])
    return X, y


def _students():
    cfg = {
        "student_widths": [8],
        "student_tree_depth": 4,
        "student_mlp": {"learning_rate_init": 0.01, "alpha": 0.0001, "max_iter": 100},
    }
    return make_students(cfg, seed=0)


def test_mlp_forward_matches_predict_and_parameters_are_counted():
    X, y = _data()
    mlp = _students()["MLP 2x8"].fit(X, y)
    assert np.allclose(mlp_forward(mlp)(X[:50]), mlp.predict(X[:50]))
    assert count_parameters(mlp) == (3 * 8 + 8) + (8 * 8 + 8) + (8 * 1 + 1)  # weights and biases


def test_tree_node_count():
    X, y = _data()
    tree = DecisionTreeRegressor(max_depth=3, random_state=0).fit(X, y)
    assert count_parameters(tree) == tree.tree_.node_count


def test_single_row_latency_returns_ordered_percentiles():
    X, _ = _data(20)
    rows = [X[i : i + 1] for i in range(len(X))]
    out = single_row_latency(lambda r: r.sum(), rows, warmup=20, n_timed=200)
    assert 0 < out["p50_us"] <= out["p99_us"]


def test_artifact_names_and_best_arm_choice():
    import pandas as pd

    assert artifact_name("MLP 2x32") == "mlp_2x32"
    assert artifact_name("Tree (depth 6)") == "tree_depth_6"
    runs = pd.DataFrame(
        {
            "student": ["a"] * 4 + ["b"] * 2,
            "arm": ["1", "1", "2", "2", "1", "2"],
            "test_rmse": [0.5, 0.6, 0.3, 0.4, 0.2, 0.9],
        }
    )
    assert best_arms(runs) == {"a": "2", "b": "1"}  # lowest mean test RMSE per student


def test_measure_reports_memory_for_a_saved_model(tmp_path):
    import joblib

    X, y = _data()
    joblib.dump(Ridge().fit(X, y), tmp_path / "ridge.joblib")
    out = measure(tmp_path / "ridge.joblib", np.tile(X, (50, 1)))
    assert out["batch_rows"] == 50 * len(X)
    assert out["rss_after_load_mb"] >= out["rss_before_mb"] > 0
    assert out["peak_rss_during_predict_mb"] >= out["rss_after_load_mb"] - 1
