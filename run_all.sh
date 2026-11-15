#!/usr/bin/env bash
# Rebuild every result from a clean checkout: the two searches, the 45 distillation fits,
# the saved models, and the tests. Takes roughly 15 to 20 minutes on an Apple M5 Pro.
# The notebooks in notebooks/ read what this writes to results/ and rebuild the figures.
set -euo pipefail
cd "$(dirname "$0")"

python -m src.tune --model lgbm
python -m src.tune --model mlp
python -m src.distill
python -m src.build_artifacts
python -m pytest
