# Decisions

Choices made along the way and why. Newest last. Add an entry whenever a choice would be hard to guess from the code.

## D1. One dependency file: `pyproject.toml`
No `requirements.txt`. One file holds dependencies, the `dev` extra, and the `ruff` and `pytest` settings. Install with `pip install -e ".[dev]"`.
**Open:** versions are unpinned. Add a lock file once the experiments are final.

## D2. A `src/` package, installed editable
Code lives in `src/`, notebooks and tests import it with `from src.lib import ...`. The editable install makes that work from any folder, so notebooks need no path tricks. This differs from the flat layout first drawn in `DESIGN.md`, which was updated to match.

## D3. One frozen test set, one set of folds, saved to disk
20% of rows are held out with the config seed and never used until the final report. The rest is split into 5 folds, and every model uses the same folds. Both live in a single file, `data/split.csv`, so reruns give byte-identical results. EDA and baselines use the training portion only.

## D4. Scaling happens inside a `Pipeline`
`build_pipeline` wraps ridge, the MLP and the students with a `StandardScaler`, so cross-validation fits the scaler on each training fold only. Tree models go in bare. A test checks the scaler really is refit per fold.

## D5. Metrics are written by hand and tested against sklearn
`evaluate_regression` uses plain NumPy, so the parity test to 1e-9 checks something real.

## D6. EDA informs decisions but does not change the data
Outliers and capped values stay in. Any clipping or transform would go inside the `Pipeline`, fit on training folds only. Trees ignore the outliers. Ridge and the MLP may suffer, and that is worth reporting.

## D7. Baselines and the bar to beat
Predict-the-mean, ridge and a decision tree with depth fixed at 6 (not tuned), scored by the saved 5-fold CV. RMSE is 1.156, 0.721 and 0.693. The tree is the strongest baseline, so it is the number a student or ensemble must clearly beat. Its edge over ridge is only about 2x the fold std.

## D8. The teacher is a plain average, not a stack
The teacher averages the tuned LightGBM and a random forest. No meta-model learns the weights, which keeps the project about distillation and not about combining methods. Boosting is still covered: gradient boosting from scratch, and LightGBM.

## D9. Docker and CI are postponed
Both belong to the optional extras in `DESIGN.md`. They were removed until the core experiments exist, since CI would have failed with no tests.

## D10. `DESIGN.md` stays out of git
It is listed in `.gitignore` and is the working plan, not a deliverable.

## Open items
- **LightGBM needs `libomp` on macOS.** `import lightgbm` fails until it is installed (Homebrew is not installed yet). Fix before the LightGBM vs. MLP step.
