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
- `notebooks/error_correlation.ipynb`: voters simulation (21 independent voters reach 82.7%, 21 identical copies stay at 60.1%) and error correlation of the trees in one random forest (correlation 0.39, single-tree RMSE 0.806, forest RMSE 0.508), a `max_features` sweep with a `bootstrap=False` extreme case (`results/error_correlation_sweep.csv`), and `results/fig2_error_correlation.png`.
- Search budget in `configs/config.yaml` (`tune.n_trials: 20`, `tune.time_cap_seconds: 900`), the same for LightGBM and the MLP, set before any search is run.
- `notebooks/lightgbm_vs_mlp.ipynb`: LightGBM random search over seven settings, scored by the shared 5-fold CV with early stopping on an inner slice of each training fold. A 3-trial run logs one row per trial (parameters, CV mean and std, trial seconds, cumulative seconds).
- Same notebook: PyTorch MLP random search (width, depth, learning rate, weight decay, dropout) with standardized features and target, Adam, MSE and early stopping. One shared `run_search` loop writes the same log columns for both models, and RMSE is reported in $100k, directly comparable with LightGBM.
- `src/tune.py` (`python -m src.tune --model lgbm` or `--model mlp`): both searches as a CLI that stops at `n_trials` or the time cap, with `tests/test_tune.py`. Real 20-trial runs wrote `results/trials_{lgbm,mlp}.csv` and `results/best_{lgbm,mlp}.json` (best CV RMSE 0.4492 for LightGBM, 0.5184 for the MLP).
- `notebooks/lightgbm_vs_mlp.ipynb` now loads the search results: budget curves in `results/fig3_equal_budget.png` and a report in `results/tuning_summary.csv` (best settings refit on the full training portion, test set scored once: LightGBM RMSE 0.441, MLP 0.530, both under 20 trials). Search code moved out of the notebook into `src/tune.py`, with a `fit_predict_lgbm` helper.
- `notebooks/distillation.ipynb`: expected share of the teacher's gain kept by each student, written before running any distillation.
- `AverageEnsemble` and `EarlyStoppingLGBM` in `src/lib.py` (with tests, including a joblib round trip), and the teacher in `notebooks/distillation.ipynb`: the average of the tuned LightGBM and a 300-tree random forest. Test RMSE is 0.458, lower than the forest (0.494) but not LightGBM alone (0.441), because their errors are 0.92 correlated.
- `out_of_fold_predictions` in `src/lib.py` (with a test) and out-of-fold teacher labels for all 16,512 training rows in `data/teacher_oof.csv`. Teacher RMSE against the true labels is 0.464 out-of-fold but 0.239 in-sample.

### Changed
- `DESIGN.md` repo skeleton and file paths now match this layout.

### Removed
- `requirements.txt` and `requirements.lock.txt`, in favor of `pyproject.toml`.
- `Dockerfile` and the CI workflow, until the optional extras.
