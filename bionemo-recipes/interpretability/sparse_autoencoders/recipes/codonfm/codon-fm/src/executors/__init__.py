from typing import Callable
from functools import partial

import nemo_run as run

from .base import local_executor, get_base_executor

# Create cluster-specific executor functions by pre-filling the cluster parameter
CLUSTER_MAP: dict[str, Callable[..., run.Executor]] = {
    "ord": partial(get_base_executor, cluster="ord"),
    "draco": partial(get_base_executor, cluster="draco"),
    "eos": partial(get_base_executor, cluster="eos"),
    "dfw": partial(get_base_executor, cluster="dfw"),
}

def get_executor_fn(cluster: str) -> Callable[..., run.Executor]:
    """
    Returns a function that creates an executor for a given cluster.
    The user parameter is ignored and kept for backward compatibility.
    """

    if cluster == "local":
        return local_executor

    if cluster in CLUSTER_MAP:
        return CLUSTER_MAP[cluster]

    raise ValueError(f"Unknown cluster `{cluster}`")
