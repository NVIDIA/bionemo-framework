#!/bin/bash
#SBATCH --account=??                  # account (user must belong to account)
#SBATCH --nodes=1                     # number of nodes
#SBATCH --partition=??                # partition (should be compatible with account)
#SBATCH --ntasks-per-node=??          # n tasks per machine (one task per gpu) <required>
#SBATCH --gpus-per-node=??
#SBATCH --time=??                     # wall time  (8 for batch, backfill, 2 for batch_short)
#SBATCH --mem=0                       # all mem avail
#SBATCH --mail-type=FAIL              # only send email on failure
#SBATCH --overcommit
#SBATCH --exclusive                   # exclusive node access
set -x

# Below is a sample set of parameters for launching ESM-1nv or ProtT5nv finetuning for a secondary structure predition 
# downstream task with BioNeMo on SLURM-based clusters
# Replace all ?? with appropriate values prior to launching a job
# Any parameters not specified in this script can be changed in the yaml config file
# located in examples/protein/prott5nv/conf/downstream_flip_sec_str.yaml for ProtT5nv model
# or in examples/protein/esm1nv/conf/downstream_flip_sec_str.yaml for ESM-1nv model
# - place holder for ESM2 instructions -
# Other downstream yaml files can be found in those locations as well

BIONEMO_IMAGE=?? # BioNeMo container image
WANDB_API_KEY=?? # Add your WANDB API KEY

CONFIG_NAME='downstream_flip_sec_str' # name of the yaml config file with parameters, should be aligned with TASK_NAME parameter
PROTEIN_MODEL=esm2nv # protein LM name, can be esm2nv, esm1nv or prott5nv 

# Training parameters
# =========================
ACCUMULATE_GRAD_BATCHES=1 # gradient accumulation
ENCODER_FROZEN=True # encoder can be frozen or trainable 
RESTORE_FROM_PATH=/model/protein/${PROTEIN_MODEL}/esm2nv_650M_converted.nemo # Path to the pretrained model checkpoint in the container
TENSOR_MODEL_PARALLEL_SIZE=1 # tensor model parallel size,  model checkpoint must be compatible with tensor model parallel size
MICRO_BATCH_SIZE=64 # micro batch size per GPU, for best efficiency should be set to occupy ~85% of GPU memory. Suggested value for A100 80GB is 256 
MAX_STEPS=2000 # duration of training as the number of training steps
VAL_CHECK_INTERVAL=20 # how often validation step is performed, including downstream task validation
# =========================

# Logging
# =========================
TASK_NAME=secondary_structure # FLIP task name: secondary_structure, scl, meltome, etc.
EXP_TAG="" # any additional experiment info, can be empty
EXP_NAME="${PROTEIN_MODEL}_batch${MICRO_BATCH_SIZE}_gradacc${ACCUMULATE_GRAD_BATCHES}_nodes${SLURM_JOB_NUM_NODES}_encoder-frozen-${ENCODER_FROZEN}${EXP_TAG}"
CREATE_WANDB_LOGGER=True # set to False if you don't want to log results with WandB 
WANDB_LOGGER_OFFLINE=False # set to True if there are issues uploading to WandB during training

PROJECT_NAME="${PROTEIN_MODEL}_${CONFIG_NAME}"  # project name, will be used for logging
# =========================

# Mounts
# =========================
DATA_PATH="??" # Directory containing FLIP secondary structure prediction data
RESULTS_PATH="??/results/${PROJECT_NAME}/${EXP_NAME}" # directory to store logs, checkpoints and results
DATA_MOUNT=/data
RESULTS_MOUNT=/results

mkdir -p ${RESULTS_PATH}

MOUNTS="${RESULTS_PATH}:${RESULTS_MOUNT},${DATA_PATH}:${DATA_MOUNT}"
# =========================

# Necessary Exports
# =========================
export HYDRA_FULL_ERROR=1
# =========================

read -r -d '' COMMAND <<EOF
echo "*******STARTING********" \
&& echo "---------------" \
&& wandb login ${WANDB_API_KEY} \
&& echo "Starting training" \
&& cd /opt/nvidia/bionemo \
&& cd examples/protein/downstream \
&& python /opt/nvidia/bionemo/examples/protein/downstream/downstream_sec_str.py \
    --config-path=/opt/nvidia/bionemo/examples/protein/${PROTEIN_MODEL}/conf \
    --config-name=${CONFIG_NAME} \
    exp_manager.exp_dir=${RESULTS_MOUNT} \
    exp_manager.create_wandb_logger=${CREATE_WANDB_LOGGER} \
    exp_manager.wandb_logger_kwargs.name=${EXP_NAME} \
    exp_manager.wandb_logger_kwargs.project=${PROJECT_NAME} \
    ++exp_manager.wandb_logger_kwargs.offline=${WANDB_LOGGER_OFFLINE} \
    trainer.num_nodes=${SLURM_JOB_NUM_NODES} \
    trainer.devices=${SLURM_NTASKS_PER_NODE} \
    trainer.max_steps=${MAX_STEPS} \
    trainer.accumulate_grad_batches=${ACCUMULATE_GRAD_BATCHES} \
    trainer.val_check_interval=${VAL_CHECK_INTERVAL} \
    model.micro_batch_size=${MICRO_BATCH_SIZE} \
    model.tensor_model_parallel_size=${TENSOR_MODEL_PARALLEL_SIZE} \
    model.dwnstr_task_validation.enabled=False \
    model.encoder_frozen=${ENCODER_FROZEN} \
    model.data.task_name=${TASK_NAME} \
    model.data.dataset_path=${DATA_MOUNT}/${TASK_NAME} \
    restore_from_path=${RESTORE_FROM_PATH} 

EOF

srun \
    --job-name ${EXP_NAME} \
    --output ${RESULTS_PATH}/slurm-%j-%n.out \
    --error ${RESULTS_PATH}/error-%j-%n.out \
    --container-image ${BIONEMO_IMAGE} \
    --container-mounts ${MOUNTS} \
    bash -c "${COMMAND}"

set +x
