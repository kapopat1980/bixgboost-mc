"""Reproducibility utilities — set seeds for NumPy, PyTorch, Python."""

import random
import numpy as np
import torch

PAPER_SEEDS = [42, 123, 456, 789, 1024]


def set_all_seeds(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
