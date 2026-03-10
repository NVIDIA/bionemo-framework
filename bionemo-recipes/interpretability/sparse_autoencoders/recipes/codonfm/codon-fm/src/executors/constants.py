import os
import nemo_run as run

DEFAULT_ENV_VARS = {
    "TRANSFORMERS_OFFLINE": "1",
    "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
    "NCCL_NVLS_ENABLE": "0",
    "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
    "NVTE_ASYNC_AMAX_REDUCTION": "1",
    "NVTE_FUSED_ATTN": "0",
}
# Default account for cluster execution, can be overridden with NEMORUN_ACCOUNT env var
DEFAULT_ACCOUNT = os.getenv("NEMORUN_ACCOUNT", "healthcareeng_virtualcell")

CLUSTER_TUNNELS = {
    "ord": lambda user, identity, job_dir: run.SSHTunnel(
        host="cs-oci-ord-dc-01.nvidia.com",
        user=user,
        job_dir=job_dir
        if job_dir
        else f"/lustre/fsw/portfolios/healthcareeng/users/{user}/results/nemo-experiments",
        identity=identity,
    ),
    "draco": lambda user, identity, job_dir: run.SSHTunnel(
        host="draco-oci-dc-01.draco-oci-iad.nvidia.com",
        user=user,
        job_dir=job_dir
        if job_dir
        else f"/lustre/fsw/portfolios/healthcareeng/users/{user}/results/nemo-experiments",
        identity=identity,
    ),
    "eos": lambda user, identity, job_dir: run.SSHTunnel(
        host="login-eos.nvidia.com",
        user=user,
        job_dir=job_dir
        if job_dir
        else f"/home/{user}/results/nemo-experiments",
        identity=identity,
    ),
    "dfw": lambda user, identity, job_dir: run.SSHTunnel(
        host="10.65.36.11",
        user=user,
        job_dir=job_dir
        if job_dir
        else f"/home/{user}/results/nemo-experiments",
        identity=identity,
    ),
}


CLUSTER_EXECUTORS = {
    "ord": lambda cluster, user, identity, job_dir, nodes, devices, account=DEFAULT_ACCOUNT: run.SlurmExecutor(
        tunnel=CLUSTER_TUNNELS[cluster](user, identity, job_dir=job_dir),
        account=account,
        partition="polar,polar3,polar4",
        time="00:30:00",
        nodes=nodes,
        ntasks_per_node=devices,
        gpus_per_node=devices,
        mem="0",
        exclusive=True,
        gres="gpu:8",
    ),
    "draco": lambda cluster, user, identity, job_dir, nodes, devices, account=DEFAULT_ACCOUNT: run.SlurmExecutor(
        tunnel=CLUSTER_TUNNELS[cluster](user, identity, job_dir=job_dir),
        account=account,
        partition="batch_block1,batch_block3,batch_block4",
        time="00:20:00",
        nodes=nodes,
        ntasks_per_node=devices,
        mem="0",
        exclusive=True,
        gres="gpu:8",
    ),
    "eos": lambda cluster, user, identity, job_dir, nodes, devices, account=DEFAULT_ACCOUNT: run.SlurmExecutor(
        tunnel=CLUSTER_TUNNELS[cluster](user, identity, job_dir=job_dir),
        account=account,
        partition="batch",
        time="00:20:00",
        nodes=nodes,
        ntasks_per_node=devices,
        mem="0",
        exclusive=True,
        gres=None,
    ),
    "dfw": lambda cluster, user, identity, job_dir, nodes, devices, account=DEFAULT_ACCOUNT: run.SlurmExecutor(
        tunnel=CLUSTER_TUNNELS[cluster](user, identity, job_dir=job_dir),
        account=account,
        partition="batch",
        time="00:20:00",
        nodes=nodes,
        ntasks_per_node=devices,
        mem="0",
        exclusive=True,
        gres="gpu:8",
    ),
} 