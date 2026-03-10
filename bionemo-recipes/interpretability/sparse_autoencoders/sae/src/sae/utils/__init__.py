# Seed and device utilities
import torch
import random
import subprocess
import numpy as np


def get_device() -> str:
    """Get available device.

    Returns:
        str: Available device
    """
    return 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


def set_seed(seed: int):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior where possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_file_limit() -> int:
    """
    Determine the system's maximum number of open files limit.

    Uses the 'ulimit -n' command to get the system limit and falls back to a
    conservative default if the command fails.

    Returns:
        int: Maximum number of files that can be opened simultaneously
    """
    try:
        result = subprocess.run(
            ['ulimit', '-n'],
            capture_output=True,
            text=True,
            shell=True
        )
        return int(result.stdout.strip())
    except (subprocess.SubprocessError, ValueError):
        return 1024  # Default to a conservative value


# Memory utilities
from .memory import (
    sae_weight_memory,
    sae_forward_memory,
    sae_backward_memory,
    sae_total_memory,
)


__all__ = [
    'set_seed',
    'get_device',
    'get_file_limit',
    'sae_weight_memory',
    'sae_forward_memory',
    'sae_backward_memory',
    'sae_total_memory',
]
