import random
from pathlib import Path

import numpy as np
import torch
import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"


def load_config(path=CONFIG_PATH):
    """Read the project config. Every setting comes from here."""
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed):
    """Seed random, numpy and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
