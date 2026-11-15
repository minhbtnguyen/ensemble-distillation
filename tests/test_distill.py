import numpy as np
from sklearn.linear_model import Ridge

from src.distill import fit_and_score


def test_fit_and_score_records_scores_and_per_row_errors():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3))
    y = X @ np.array([1.0, 2.0, -1.0]) + rng.normal(0, 0.3, size=300)
    X_train, y_train, X_test, y_test = X[:200], y[:200], X[200:], y[200:]
    teacher_pred = y_test + 0.1  # a teacher that is off by a constant 0.1

    student = Ridge()
    scores, sq = fit_and_score(student, X_train, y_train, X_test, y_test, teacher_pred)

    assert not hasattr(student, "coef_")  # the original student is left unfitted
    assert sq.shape == (100,)
    assert np.isclose(scores["test_rmse"], np.sqrt(sq.mean()))  # per-row errors add up to the RMSE
    assert scores["train_rows"] == 200
    assert scores["train_seconds"] >= 0
    pred = Ridge().fit(X_train, y_train).predict(X_test)
    assert np.isclose(scores["fidelity_rmse"], np.sqrt(np.mean((pred - teacher_pred) ** 2)))
