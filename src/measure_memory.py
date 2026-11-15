"""Memory of one saved model, measured in a fresh process.

    python -m src.measure_memory artifacts/mlp_2x32.joblib [--rows 200000]

Prints one line of JSON: resident set size (MB) before loading the artifact, after loading it,
and the peak while predicting a large batch. Everything the model needs to be unpickled is
imported before the first reading, so the load number is the model itself.
"""

import argparse
import json
import resource
import sys
import threading
import time

import joblib
import lightgbm  # noqa: F401  (imported so the teacher can be unpickled without counting the import)
import numpy as np
import psutil
from threadpoolctl import threadpool_limits

from src.bench import pin_threads
from src.lib import load_data, load_split  # noqa: F401  (src.lib classes are needed to unpickle)

MB = 1024 * 1024


def _rss_mb():
    return psutil.Process().memory_info().rss / MB


class _PeakSampler(threading.Thread):
    """Sample the process's resident memory every millisecond and remember the highest."""

    def __init__(self):
        super().__init__(daemon=True)
        self.peak = _rss_mb()
        self._stop_flag = threading.Event()

    def run(self):
        while not self._stop_flag.is_set():
            self.peak = max(self.peak, _rss_mb())
            time.sleep(0.001)

    def stop(self):
        self._stop_flag.set()
        self.join()
        self.peak = max(self.peak, _rss_mb())


def measure(path, X_batch):
    """Load the artifact and predict X_batch, reading memory before, after loading, and at peak."""
    before = _rss_mb()
    model = joblib.load(path)
    after_load = _rss_mb()
    pin_threads(model)
    sampler = _PeakSampler()
    sampler.start()
    with threadpool_limits(limits=1):
        model.predict(X_batch)
    sampler.stop()
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    ru_mb = ru / MB if sys.platform == "darwin" else ru / 1024  # macOS reports bytes, Linux KB
    return {
        "rss_before_mb": before,
        "rss_after_load_mb": after_load,
        "load_mb": after_load - before,
        "peak_rss_during_predict_mb": sampler.peak,
        "process_lifetime_peak_mb": ru_mb,
        "batch_rows": len(X_batch),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("artifact")
    parser.add_argument("--rows", type=int, default=200_000)
    args = parser.parse_args(argv)

    X, _ = load_data()
    test_idx, _folds = load_split()
    X_test = X.to_numpy()[test_idx]
    X_batch = np.tile(X_test, (args.rows // len(X_test) + 1, 1))[: args.rows]
    del X
    print(json.dumps(measure(args.artifact, X_batch)))


if __name__ == "__main__":
    main()
