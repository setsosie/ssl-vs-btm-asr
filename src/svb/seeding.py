"""Global seeding for reproducible runs.

A single ``set_all_seeds`` is called at the start of every run. The seed
controls weight initialisation (including the new CTC-head rows added for a
held-out language), data shuffling, dropout, and the DARE-TIES drop mask.

It does not buy bit-exact runs on GPU. CTC's backward pass has no
deterministic CUDA kernel — ``torch.use_deterministic_algorithms(True)`` raises
for it rather than selecting one — so training remains nondeterministic
run-to-run even at a fixed seed. That is precisely why the study reports five
seeds and a spread rather than a single number.

``PYTHONHASHSEED`` is deliberately not set here: Python reads it once at
interpreter start, so assigning it from inside a running process does nothing.
Set it in the launcher if hash-order determinism is ever needed.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_all_seeds(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and Torch (CPU + CUDA).

    Args:
        seed: The run seed.
        deterministic: If True, request deterministic cuDNN algorithms. This
            removes cuDNN's autotuning as a source of variation. It does not
            make the run deterministic: CTC backward on CUDA is
            nondeterministic regardless.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
