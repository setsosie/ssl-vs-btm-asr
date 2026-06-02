"""Global seeding for reproducible runs.

A single ``set_all_seeds`` is called at the start of every run. The seed
controls weight init of new CTC-head rows, data shuffling, dropout, and
SpecAugment masks — i.e. every source of training stochasticity that varies
across our multi-seed reporting.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_all_seeds(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and Torch (CPU + CUDA).

    Args:
        seed: The run seed.
        deterministic: If True, request deterministic cuDNN algorithms. This
            can slow training slightly but removes a source of run-to-run
            variation that would otherwise inflate our reported std.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
