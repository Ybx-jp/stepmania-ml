"""
Reproducibility utilities for deterministic training.

Ensures consistent results across runs by seeding all random number generators
and configuring PyTorch for deterministic behavior.
"""

import random

import numpy as np
import torch

SEED = 42


def set_seed(seed=SEED):
    """
    Set all random seeds for reproducibility.

    Args:
        seed: Random seed value (default: 42)
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
