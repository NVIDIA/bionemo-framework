# Put NeMoRun related stuff here.

from dataclasses import field, dataclass
from typing import Optional, Type

from nemo_run import Config, Partial, autoconvert
import nemo_run as run

from bionemo.geneformer.run.argument_parser import parse_args
from bionemo.geneformer.run.main import args_to_args_dict, defer_load, load_config
from bionemo.llm.train import train


# Dataclass that mirrors the Argument parser implementation. All changes here must be reflected in argparse.
@dataclass
class NRArgs:
    config_dict: dict
    model_config_cls: Type
    data_config_cls: Type
    resume_if_exists: bool
    nsys_profiling: bool
    nsys_start_step: int
    nsys_end_step: int
    nsys_ranks: list[int] = field(default_factory=lambda: [0])

# Creates run.Config object
@autoconvert
def build_nrargs(args) -> NRArgs:
    return NRArgs(**args)


def parse_args_and_recipe():
    args = parse_args()
    # Convert argparse Namespace into a dictionary
    args_dict = args_to_args_dict(args)
    recipe = Partial(defer_load, build_nrargs(args_dict))

    # Example using a simple local executor
    executor = run.LocalExecutor()
    run.run(recipe, executor=executor, detach=True, dryrun=False)


def defer_load(fdl_config: Config[NRArgs]):
    ''' Executes a training job from a fiddle configuration object that has been derived from an argparse Namespace. 

    This operates on a fiddle Config directly, which means we can defer the rest of input parsing and validation to when
    the job launches. The upside of this is we get SLURM wrappers for free, the downside is we do not get input validation
    at this stage.
    '''

    # NOTE: this is the same method used by run/main.py to parse the config.
    deserialized_config = load_config(fdl_config.config_dict, fdl_config.model_config_cls, fdl_config.data_config_cls)
    config = deserialized_config
    train(
        bionemo_exposed_model_config=config.bionemo_model_config,
        data_config=config.data_config,
        parallel_config=config.parallel_config,
        training_config=config.training_config,
        optim_config=config.optim_config,
        experiment_config=config.experiment_config,
        wandb_config=config.wandb_config,
        resume_if_exists=fdl_config.resume_if_exists,
    )


def slurm_executor_with_custom_user_configuration():
    ''' Sets up the Slurm Executor using my own custom configuration for draco.
    
    '''
    # NOTE: slurm stuff below.
    identity="/home/bionemo/.ssh/id_ed25519"
    # OPTIONAL: Provide path to the private key that can be used to establish the SSH connection without entering your password.
    DRACO="cs-oci-ord-login-03"
    # NOTE, how we mount determines the data and results path we would like to push in.
    # SRC: 
    #   /lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_bionemo/data/cellxgene_2023-12-15/processed_data
    #   /lustre:/lustre is the easiest mount

    CUSTOM_MOUNTS = [
        "/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_bionemo/results/bionemo2_geneformer_pretraining/bionemo2_geneformer_pretraining:/results",
        "/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_bionemo/data:/workspace/data",
        "/lustre:/lustre"
    ]

    # TODO how do we get nodes and devices out of our config?
    _executor = slurm_executor(
        user='skothenhill',
        identity=identity,
        host=DRACO,
        remote_job_dir='/home/skothenhill/20240924-bionemo2/nemorun',
        account='healthcareeng_bionemo',
        partition='polar',
        nodes=1,
        devices=8,
        custom_mounts = CUSTOM_MOUNTS,
        container_image="nvcr.io/nvidia/clara/bionemo-framework:nightly",
        custom_env_vars={"WANDB_API_KEY": os.environ.get('WANDB_API_KEY', '')}
    )
    return _executor

# Everything below here we are figuring out how to incorporate
def slurm_executor(
    user: str,
    host: str,
    remote_job_dir: str,
    account: str,
    partition: str,
    nodes: int,
    devices: int,
    identity: str,
    time: str = "01:00:00",
    custom_mounts: Optional[list[str]] = None,
    custom_env_vars: Optional[dict[str, str]] = None,
    container_image: str = "nvcr.io/nvidia/nemo:dev",
    retries: int = 0,
) -> run.SlurmExecutor:
    if not (user and host and remote_job_dir and account and partition and nodes and devices):
        raise RuntimeError(
            "Please set user, host, remote_job_dir, account, partition, nodes and devices args for using this function."
        )

    mounts = []
    # Custom mounts are defined here.
    if custom_mounts:
        mounts.extend(custom_mounts)

    # Env vars for jobs are configured here
    env_vars = {
        "TRANSFORMERS_OFFLINE": "1",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
        "NVTE_ASYNC_AMAX_REDUCTION": "1",
        "NVTE_FUSED_ATTN": "0",
    }
    if custom_env_vars:
        env_vars |= custom_env_vars

    # This defines the slurm executor.
    # We connect to the executor via the tunnel defined by user, host and remote_job_dir.
    executor = run.SlurmExecutor(
        account=account,
        partition=partition,
        tunnel=run.SSHTunnel(
            user=user,
            host=host,
            job_dir=remote_job_dir, # This is where the results of the run will be stored by default.
            identity=identity
        ),
        nodes=nodes,
        ntasks_per_node=devices,
        gpus_per_node=devices,
        mem="0",
        exclusive=True,
        gres="gpu:8",
    )

    executor.container_image = container_image
    executor.container_mounts = mounts
    executor.env_vars = env_vars
    executor.retries = retries
    executor.time = time

    return executor