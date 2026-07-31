"""Deterministic seeding across every RNG the pipeline can touch.

Invariant 5 requires that every run records all seeds it set. `seed_everything` therefore
returns a description of what it did rather than returning ``None``, so callers can drop
the result straight into ``run_context.json``.
"""

from __future__ import annotations

import os
import random
from typing import Any

# CuBLAS needs this set *before* the first CUDA context is created, otherwise
# torch.use_deterministic_algorithms raises on matmul-backed ops.
CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def seed_everything(seed: int, *, deterministic: bool = True) -> dict[str, Any]:
    """Seed python, numpy and (if installed) torch/CUDA, and report what was set.

    Args:
        seed: Non-negative seed applied to every available RNG.
        deterministic: If True, also request deterministic algorithms from torch and
            disable cuDNN benchmarking. Set False only for throughput-oriented runs
            whose results are not reported.

    Returns:
        A JSON-serialisable dict describing the seed, which backends were seeded, the
        number of CUDA devices seeded, and the environment variables set. Backends that
        are not installed appear with a ``False`` / absent entry rather than raising.

    Raises:
        ValueError: If ``seed`` is negative or does not fit in 32 bits, which numpy
            rejects and which would otherwise fail deep inside a data loader.
    """
    if seed < 0 or seed > 2**32 - 1:
        raise ValueError(f"seed must be in [0, 2**32-1], got {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG

    record: dict[str, Any] = {
        "seed": seed,
        "deterministic": deterministic,
        "python_random": True,
        "numpy": False,
        "torch": False,
        "torch_cuda_devices": 0,
        "env": {
            "PYTHONHASHSEED": os.environ["PYTHONHASHSEED"],
            "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        },
    }

    random.seed(seed)

    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a core dependency
        pass
    else:
        np.random.seed(seed)
        record["numpy"] = True
        record["numpy_version"] = np.__version__

    try:
        import torch
    except ImportError:
        return record

    torch.manual_seed(seed)
    record["torch"] = True
    record["torch_version"] = torch.__version__

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        record["torch_cuda_devices"] = torch.cuda.device_count()

    if deterministic:
        # warn_only=True: a handful of PyG scatter kernels have no deterministic
        # implementation. We want the warning in the log, not a hard crash.
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        record["cudnn_deterministic"] = True

    return record
