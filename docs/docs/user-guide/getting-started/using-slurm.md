# **Using SLURM to run BioNeMo training**

SLURM (Simple Linux Utility for Resource Management) is an open-source job scheduler that distributes workloads across multiple compute nodes.

SLURM manages job scheduling by queuing tasks and executing them when resources become available. Compute resources are organized into **nodes**, which belong to **partitions** (queues), and users allocate resources by specifying the number of nodes, CPU cores, and memory. Jobs can be submitted as **batch scripts (`sbatch`)** for scheduled execution or run interactively using **`srun`**.

This guide will walk through key SLURM concepts and how to use SLURM for running ML training jobs. In particular, we will

1. Cover basic SLURM commands
2. Walk through an example SBATCH script for running BioNeMo training

## 1. Core SLURM commands

### **`sbatch`** (Submit a Batch Job)

The `sbatch` command submits a batch script to the SLURM scheduler. The script specifies the job's resource requirements and execution commands.

#### **Example 1: Submitting a job**

```bash
sbatch my_training_script.sh
```

This submits the batch script `my_training_script.sh` to SLURM.

We will walk through an example SBATCH script in section 3 below.

### **`srun`** (Run Jobs Interactively)

The `srun` command runs jobs interactively or within a batch job. It is useful for debugging, running short jobs, or launching tasks on allocated resources.

#### **Example 1: Running an interactive job**

```bash
srun --partition=batch --nodes=1 --ntasks=8 --time=00:30:00 --pty bash
```

This starts an interactive session with 8 tasks on one node for 30 minutes.

#### **Example 2: Running a command on allocated resources**

```bash
srun --ntasks=4 python train.py
```

This runs `train.py` using 4 tasks (e.g., GPUs or CPU cores).

### **`squeue`** (Check Job Status)

The `squeue` command displays the status of jobs in the SLURM queue.

#### **Example 1: Viewing all jobs in the queue**

```bash
squeue
```

Lists all running and pending jobs.

#### **Example 2: Checking the status of your own jobs**

```bash
squeue -u $USER
```

Shows only your jobs in the queue.

#### **Example 3: Checking the status of your team's jobs**

```bash
squeue -A <account_name>
```

Shows only your team's jobs in the queue.

### **`sacct`** (Check Job History)

The `sacct` command retrieves historical job information, including resource usage and exit statuses.

#### **Example 1: Viewing completed jobs**

```bash
sacct
```

Displays jobs that have finished or failed.

#### **Example 2: Checking details of a specific job**

```bash
sacct -j 123456 --format=JobID,State,Elapsed,MaxRSS
```

Shows the **state, runtime, and memory usage** of job `123456`.

### **`scancel`** (Cancel a Job)

The `scancel` command cancels running or pending SLURM jobs.

#### **Example 1: Canceling a specific job**

```bash
scancel 123456
```

Cancels job `123456`.

#### **Example 2: Canceling all of your jobs**

```bash
scancel -u $USER
```

Cancels all jobs belonging to the current user.

## 2. Example SBATCH script

The following SBATCH script runs a BioNeMo Evo 2 training job inside a containerized environment.

We'll walk through the script in detail below.

```bash
#!/bin/bash
# SLURM directives
# =========================
#SBATCH --account=[INSERT ACCOUNT HERE]
#SBATCH --nodes=1
#SBATCH --partition=[INSERT PARTITIONs HERE]
#SBATCH --ntasks-per-node=8
#SBATCH --time=00:15:00
#SBATCH --mem=0
#SBATCH --job-name=[INSERT JOB NAME HERE]
#SBATCH --mail-type=FAIL
#SBATCH --exclusive
set -x # Enable debugging (prints executed commands)
# Job-specific parameters
# =========================
IMAGE_NAME=[INSERT IMAGE NAME HERE]
EXPERIMENT_NAME=[INSERT PROJECT NAME HERE]
MODEL_SIZE=7b
CP_SIZE=1
TP_SIZE=1
PP_SIZE=1
MICRO_BATCH_SIZE=2
GRAD_ACC_BATCHES=1
SEQ_LEN=8192
MAX_STEPS=100
VAL_CHECK=50
CLIP_GRAD=250
EXTRA_ARGS="--enable-preemption --use-megatron-comm-overlap-llama3-8k --ckpt-async-save --overlap-grad-reduce --clip-grad $CLIP_GRAD --eod-pad-in-loss-mask"
EXTRA_ARG_DESC="BF16_perf_cg250_continue"
LR=0.0003
MIN_LR=0.00003
WU_STEPS=2500
# 0xDEADBEEF
SEED=1234
WD=0.1
ADO=0.01
HDO=0.01
# Mounts
# =========================
DATA_PATH=/lustre/.../[INSERT DATA PATH HERE]
DATA_MOUNT=/workspace/bionemo2/data
MODEL_PATH=/lustre/.../[INSERT MODEL PATH HERE]
MODEL_MOUNT=/workspace/bionemo2/model
RESULTS_PATH=$MODEL_PATH/experiments/${EXPERIMENT_NAME}
mkdir -p $RESULTS_PATH
MOUNTS=${DATA_PATH}:${DATA_MOUNT},${MODEL_PATH}:${MODEL_MOUNT},$HOME/.cache:/root/.cache
# Training command
# =========================
read -r -d '' COMMAND <<EOF
echo "*******STARTING********" \
&& echo "---------------" \
&& echo "Starting training" \
&&  \
python /workspace/bionemo2/sub-packages/bionemo-evo2/src/bionemo/evo2/run/train.py \
    -d /workspace/bionemo2/sub-packages/bionemo-evo2/examples/configs/full_pretrain_shortphase_config.yaml \
    --num-nodes=${SLURM_JOB_NUM_NODES} \
    --devices=${SLURM_NTASKS_PER_NODE} \
    --grad-acc-batches $GRAD_ACC_BATCHES \
    --max-steps=$MAX_STEPS \
    --seed $SEED \
    ${EXTRA_ARGS} \
    --no-wandb \
    --lr $LR \
    --wd $WD \
    --min-lr $MIN_LR \
    --warmup-steps $WU_STEPS \
    --attention-dropout $ADO \
    --hidden-dropout $HDO \
    --limit-val-batches=20 \
    --val-check-interval=${VAL_CHECK} \
    --experiment-dir=/workspace/bionemo2/model/checkpoints/${EXPERIMENT_NAME} \
    --seq-length=${SEQ_LEN} \
    --tensor-parallel-size=${TP_SIZE} \
    --context-parallel-size=${CP_SIZE} \
    --pipeline-model-parallel-size=${PP_SIZE} \
    --workers 8 \
    --micro-batch-size=${MICRO_BATCH_SIZE} \
    --model-size=${MODEL_SIZE}
EOF
srun \
    --output ${RESULTS_PATH}/slurm-%j.out \
    --error ${RESULTS_PATH}/error-%j.out \
    --container-image=$IMAGE_NAME \
    --container-mounts ${MOUNTS} \
    bash -c "${COMMAND}"
set +x # Disable debugging
```

#### SLURM Directives (Resource Allocation)

After the first shebang line, you'll need to add some `#SBATCH` directives to define how SLURM manages the job:

| **Directive**                                  | **Description**                                          |
| ---------------------------------------------- | -------------------------------------------------------- |
| `#SBATCH --account=[INSERT ACCOUNT HERE]`      | Specifies the **SLURM account** for billing.             |
| `#SBATCH --nodes=1`                            | Requests **1 compute node**.                             |
| `#SBATCH --partition=[INSERT PARTITIONs HERE]` | Specifies **job queue (partition)**.                     |
| `#SBATCH --ntasks-per-node=8`                  | Requests **8 tasks per node (often maps to GPUs/CPUs)**. |
| `#SBATCH --time=00:15:00`                      | Limits execution time to **15 minutes**.                 |
| `#SBATCH --mem=0`                              | Uses **default max memory** available.                   |
| `#SBATCH --job-name=[INSERT JOB NAME HERE]`    | Names the job for tracking.                              |
| `#SBATCH --mail-type=FAIL`                     | Sends an email **if the job fails**.                     |
| `#SBATCH --exclusive`                          | Ensures the job **has exclusive access to the node**.    |

**Tip**: You can check partition limits and node availability using `sinfo`

These must be specified before the job-specific parameters below. They will be passed to our script as SLURM-provided variables,
e.g. `${SLURM_JOB_NUM_NODES}`.

---

#### Job-Specific Parameters

These environment variables configure training. They can be whatever you want. Here we set various parameters that we will use in the training script below.

It's not necessary to specify these as variables, you could just specify them directly in the training script below, but specifying them as variables makes the script more readable and easier to modify.

```bash
IMAGE_NAME=[INSERT QA IMAGE NAME HERE]
EXPERIMENT_NAME=[INSERT QA PROJECT HERE]
SEQ_LEN=8192
MAX_STEPS=100
...
ADO=0.01
HDO=0.01
```

---

#### Mounts

SLURM jobs run in an HPC environment where filesystems are often shared across nodes.

In our case, we store stuff in the Lustre file system directories, so you'll want to mount the correct Lustre path to the correct container path.
The Lustre filesystem is mounted in the `/lustre` directory, and the container is mounted in the `/workspace` directory.

To specify a mount, first specify the Lustre path, then the container path, separated by a `:`. You can specify multiple mounts by separating them with commas.

Once again, we specify the paths as variables for readability and ease of modification.

```bash
DATA_PATH=/lustre/.../[INSERT DATA PATH HERE]
DATA_MOUNT=/workspace/bionemo2/data
MODEL_PATH=/lustre/.../[INSERT MODEL PATH HERE]
MODEL_MOUNT=/workspace/bionemo2/model
RESULTS_PATH=$MODEL_PATH/experiments/${EXPERIMENT_NAME}
mkdir -p $RESULTS_PATH
MOUNTS=${DATA_PATH}:${DATA_MOUNT},${MODEL_PATH}:${MODEL_MOUNT},$HOME/.cache:/root/.cache
```

Note that paths on EOS and ORD are different, so you'll want to mount the correct Lustre path to the correct container path.

Pay special attention to `RESULTS_PATH`, as this will be the location of your experiment results.
It's common to set this `EXPERIMENT_NAME` specific to the parameters of the experiment you're running, e.g.

```
EXPERIMENT_NAME=EVO2_SEQLEN${SEQ_LEN}_PP${PP_SIZE}_TP${TP_SIZE}_CP${CP_SIZE}_LR${LR}_MINLR${MIN_LR}_WU${WU_STEPS}_GA${GRAD_ACC_BATCHES}_...
```

#### Training Command Execution

After setting up all parameters and mounts, the training script is launched within the SLURM job using a compound command. This command string—stored in the `COMMAND` variable—calls the Python training script with all the environment-specific arguments and hyperparameters defined earlier.

```bash
python /workspace/bionemo2/sub-packages/bionemo-evo2/src/bionemo/evo2/run/train.py \
    -d /workspace/bionemo2/sub-packages/bionemo-evo2/examples/configs/full_pretrain_shortphase_config.yaml \
    --num-nodes=${SLURM_JOB_NUM_NODES} \
    --devices=${SLURM_NTASKS_PER_NODE} \
    --grad-acc-batches $GRAD_ACC_BATCHES \
    --max-steps=$MAX_STEPS \
    --seed $SEED \
    ...
    --model-size=${MODEL_SIZE}
EOF
```

In the command above, we invoke the training script with all the environment-specific arguments and hyperparameters defined earlier, including the SLURM-provided variables and our own parameters such as gradient accumulation, learning rate, etc.

The command is executed in a containerized environment to ensure that dependencies and runtime conditions remain consistent. This is accomplished by using the `srun` command with container options:

```bash
srun --output ${RESULTS_PATH}/slurm-%j.out \
    --error ${RESULTS_PATH}/error-%j.out \
    --container-image=$IMAGE_NAME \
    --container-mounts ${MOUNTS} \
    bash -c "${COMMAND}"
```

- `srun`initiates the job across allocated resources.
- `--container-image` flag ensures that the job runs with the correct environment. Here you can point to an image or a `.sqsh` file.
- `--container-mounts` flag maps the Lustre file system directories to the container’s workspace, as we specified above, ensuring that data, models, and results are accessible.
- Log Redirection: the `--error` and `--output` flags redirect standard output and error messages to dedicated log files (`slurm-%j.out` and `error-%j.out`) under the results directory.

To access logs, you can use the following commands:

```bash
# Standard output logs
cat ${RESULTS_PATH}/slurm-<JOB_ID>.out

# Error logs
cat ${RESULTS_PATH}/error-<JOB_ID>.out
```

### Using The Sbatch Script

To kick off the job, you can submit the script using `sbatch`:

```bash
sbatch my_training_script.sbatch
```

This will submit the job to the SLURM scheduler, and you can check the status of the job using `squeue`:

```bash
squeue -u $USER
```
