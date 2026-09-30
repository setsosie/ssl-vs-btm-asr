"""Global seeding for reproducible runs.

A single ``set_all_seeds`` is called at the start of every run. The seed
controls weight initialisation (including the new CTC-head rows added for a
held-out language), data shuffling, dropout, and the DARE-TIES drop mask.

It does not buy bit-exact runs on GPU. CTC's backward pass has no
deterministic CUDA kernel — ``torch.use_deterministic_algorithms(True)`` raises
for it rather than selecting one — so training remains nondeterministic
run-to-run even at a fixed seed. That is precisely why the study reports
several seeds and a spread rather than a single number.

A run is a sequence of stages — phase 0, one expert per language, one transfer
per held-out language — and each is seeded separately, from the run seed and
the stage's name (:func:`seed_stage`). Seeding once at the top would make every
stage's random stream depend on how many draws the stages before it consumed:
arm A's second language would be initialised from whatever the first language's
dropout left behind, and a run resumed after a crash would draw differently
from one that never stopped. Deriving the seed per stage makes each stage a
function of the run seed and its own name, and nothing else.

``PYTHONHASHSEED`` is deliberately not set here: Python reads it once at
interpreter start, so assigning it from inside a running process does nothing.
Set it in the launcher if hash-order determinism is ever needed.
"""

from __future__ import annotations

import hashlib
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


def derive_seed(seed: int, *labels: str | int) -> int:
    """A 32-bit seed that is a fixed function of the run seed and the labels.

    Hashed rather than added. ``seed + k`` makes neighbouring run seeds share
    streams — worker 1 of seed 0 is worker 0 of seed 1 — and Python's own
    ``hash`` is salted per process, so neither gives a value that is both
    distinct across runs and the same on every machine.
    """
    text = "/".join(str(part) for part in (seed, *labels))
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big")


def seed_stage(seed: int, stage: str) -> int:
    """Seed everything for one named stage of a run; returns the seed used."""
    stage_seed = derive_seed(seed, "stage", stage)
    set_all_seeds(stage_seed)
    return stage_seed
