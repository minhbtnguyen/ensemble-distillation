# Changelog

All notable changes to this project. Format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- Project scaffold: `src/`, `tests/`, `configs/`, `notebooks/`, `data/`, `results/`, `artifacts/`, and `run_all.sh`.
- `pyproject.toml` with runtime dependencies, a `dev` extra, and `ruff` and `pytest` settings.
- `configs/config.yaml` (seed, test size, number of folds) and `src/utils.py` with `load_config` and `set_seed`.
- Frozen test split and 5-fold CV indices in `data/split.csv` (`make_split`, `save_split`, `load_split` in `src/lib.py`).
- `evaluate_regression` (RMSE, MAE, R²) with a test against `sklearn.metrics` to 1e-9.
- `build_pipeline` (scaler plus model, or a bare tree) with a test that the scaler is refit on each training fold.
- Determinism test: same seed gives an identical split and identical tiny-training predictions.
- `notebooks/setup_baselines.ipynb`: EDA on the training portion, `results/fig1_eda.png`, and the baselines table in `results/baselines.csv` (mean, ridge, depth-6 tree).
- `notebooks/error_correlation.ipynb`: voters simulation (21 independent voters reach 82.7%, 21 identical copies stay at 60.1%) and error correlation of the trees in one random forest (correlation 0.39, single-tree RMSE 0.806, forest RMSE 0.508).
- `DECISIONS.md`.

### Changed
- `DESIGN.md` repo skeleton and file paths now match this layout.

### Removed
- `requirements.txt` and `requirements.lock.txt`, in favor of `pyproject.toml`.
- `Dockerfile` and the CI workflow, until the optional extras.
