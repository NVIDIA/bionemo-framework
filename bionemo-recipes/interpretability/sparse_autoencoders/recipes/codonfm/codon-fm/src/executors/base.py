import os
from typing import Optional, List

import nemo_run as run

from .constants import CLUSTER_EXECUTORS, DEFAULT_ENV_VARS, DEFAULT_ACCOUNT

CONTAINER_OUT_DIR = '/results/'
CONTAINER_CHECKPOINTS_DIR = '/results/checkpoints/'
CONTAINER_PRETRAINED_CKPT_PATH = '/results/.pretrained.ckpt'
CONTAINER_EVAL_CKPT_PATH = '/results/.eval.ckpt'
CONTAINER_DATA_DIR = '/data/'

def get_base_executor(
    cluster: str,
    nodes: int,
    devices: int,
    exp_name: str,
    project_name: Optional[str] = None,
    identity: Optional[str] = None,
    account: str = DEFAULT_ACCOUNT,
    time: str = "01:00:00",
    custom_env_vars: Optional[dict[str, str]] = None,
    retries: int = 0,
    job_dir: Optional[str] = None,
    container_image: Optional[str] = None,
    finetune_ckpt_path: Optional[str] = None,
    eval_ckpt_path: Optional[str] = None,
    custom_mounts: Optional[List[str]] = None,
):
    """
    Creates and configures a SlurmExecutor for a given cluster.

    This function abstracts away user-specific configurations by reading them
    from environment variables. Set the following environment variables before using this function:
    - NEMORUN_USER: Your username on the cluster.
    - NEMORUN_SSH_KEY_PATH: Absolute path to your SSH private key.
    - NEMORUN_RESULTS_DIR: Base directory for your results on the cluster.
    - NEMORUN_DATA_DIR: Path to the data directory on the cluster.
    - NEMORUN_PROJECT_NAME (optional): Default project name. Defaults to 'codon-fm'.
    - NEMORUN_CONTAINER_IMAGE (optional): Default container image.
    - NEMORUN_ACCOUNT (optional): Account to use for cluster jobs. Defaults to 'healthcareeng_virtualcell'.
    - WANDB_API_KEY: Your Weights & Biases API key.

    Args:
        cluster (str): The name of the cluster to run on (e.g., 'ord', 'draco').
        nodes (int): The number of nodes to request.
        devices (int): The number of devices per node to request.
        exp_name (str): The name of the experiment. Used for creating result directories.
        project_name (Optional[str], optional): The name of the project. If not provided,
            it is read from NEMORUN_PROJECT_NAME. Defaults to None.
        identity (Optional[str], optional): Path to SSH identity file. If not provided,
            it is read from NEMORUN_SSH_KEY_PATH. Defaults to None.
        account (str, optional): The account to use for the job. Defaults to value from NEMORUN_ACCOUNT environment variable, or "healthcareeng_virtualcell" if not set.
        time (str, optional): The time limit for the job. Defaults to "01:00:00".
        custom_env_vars (Optional[dict[str, str]], optional): Custom environment variables
            for the job. Defaults to None.
        retries (int, optional): The number of times to retry the job on failure. Defaults to 0.
        job_dir (Optional[str], optional): The directory for the job. If not provided, it is
            constructed from NEMORUN_RESULTS_DIR, project_name, and exp_name. Defaults to None.
        container_image (Optional[str], optional): The container image to use. If not provided,
            it is read from NEMORUN_CONTAINER_IMAGE. Defaults to None.
        finetune_ckpt_path (Optional[str], optional): Path to a fine-tuning checkpoint to be mounted.
            The directory of this path will be mounted. Defaults to None.
        custom_mounts (Optional[List[str]], optional): A list of custom mount strings.

    Returns:
        run.SlurmExecutor: A configured SlurmExecutor instance.
    """
    global CONTAINER_OUT_DIR, CONTAINER_CHECKPOINTS_DIR, CONTAINER_PRETRAINED_CKPT_PATH, CONTAINER_DATA_DIR
    # assert supported clusters
    assert cluster in CLUSTER_EXECUTORS.keys()
    # Get user-specific configuration from environment variables
    user = os.environ.get("NEMORUN_USER")
    
    if user is None:
        raise ValueError("NEMORUN_USER environment variable must be set.")

    _identity = identity or os.environ.get("NEMORUN_SSH_KEY_PATH")
    if _identity is None:
        raise ValueError(
            "NEMORUN_SSH_KEY_PATH environment variable must be set, or 'identity' argument must be provided."
        )

    _project_name = project_name or os.environ.get("NEMORUN_PROJECT_NAME", "codon-fm")

    results_dir = os.environ.get("NEMORUN_RESULTS_DIR")
    if results_dir is None:
        raise ValueError("NEMORUN_RESULTS_DIR environment variable must be set.")

    data_dir = os.environ.get("NEMORUN_DATA_DIR")
    if data_dir is None:
        raise ValueError("NEMORUN_DATA_DIR environment variable must be set.")

    _container_image = container_image or os.environ.get("NEMORUN_CONTAINER_IMAGE")
    if _container_image is None:
        raise ValueError(
            "NEMORUN_CONTAINER_IMAGE environment variable must be set, or 'container_image' argument must be provided."
        )

    # define job directory
    _job_dir = job_dir or f"{results_dir}/{_project_name}/{exp_name}"

    # define mounts
    mounts = [
        f"{data_dir}:{CONTAINER_DATA_DIR}",
        f"{_job_dir}:{CONTAINER_OUT_DIR}",
    ]

    if custom_mounts:
        mounts.extend(custom_mounts)

    if finetune_ckpt_path:
        mounts.append(f"{finetune_ckpt_path}:{CONTAINER_PRETRAINED_CKPT_PATH}")
    if eval_ckpt_path:
        mounts.append(f"{eval_ckpt_path}:{eval_ckpt_path}")

    # define environ
    env_vars = DEFAULT_ENV_VARS
    if "WANDB_API_KEY" in os.environ:
        env_vars["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]
    if custom_env_vars:
        env_vars.update(custom_env_vars)

    # define executor
    executor = CLUSTER_EXECUTORS[cluster](cluster, user, _identity, _job_dir, nodes, devices, account)
    executor.packager = run.GitArchivePackager()
    executor.container_image = _container_image
    executor.container_mounts = mounts
    executor.env_vars = env_vars
    executor.retries = retries
    executor.time = time
    executor.account = account
    executor.dependency_type = "afterany"
    return executor


def local_executor(
    nodes=1,
    devices=2,
    retries=0,
    container_image="",
    time="",
    exp_name="",
    project_name="",
    out_dir=None,
    **kwargs,
) -> run.LocalExecutor:
    executor = run.LocalExecutor(
        ntasks_per_node=devices,
    )

    return executor