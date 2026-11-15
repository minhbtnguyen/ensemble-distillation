#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python -m pytest
# TODO: run each script in src/ in order, writing tables and figures to results/
