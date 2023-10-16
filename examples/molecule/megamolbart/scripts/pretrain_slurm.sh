#!/bin/bash
#SBATCH --account=??                  # account (user must belong to account)
#SBATCH --nodes=??                    # number of nodes
#SBATCH --partition=??                # partition (should be compatible with account)
#SBATCH --ntasks-per-node=??          # n tasks per machine (one task per gpu) <required>
#SBATCH --gpus-per-node=??
#SBATCH --time=??                     # wall time  (8 for batch, backfill, 2 for batch_short)
#SBATCH --mem=0                       # all mem avail
#SBATCH --mail-type=FAIL              # only send email on failure
#SBATCH --overcommit
#SBATCH --exclusive                   # exclusive node access
set -x

# Below is a sample set of parameters for launching MegaMolBART model training with BioNeMo on SLURM-based clusters
# Replace all ?? with appropriate values prior to launching a job
# Any parameters not specified in this script can be changed in the yaml config file
# located in examples/molecule/megamolbart/conf/pretrain_small_span_aug.yaml.
# Other config options are examples/molecule/megamolbart/conf/pretrain_xsmall_span_aug.yaml
# or examples/molecule/megamolbart/conf/pretrain_large_span_aug.yaml

BIONEMO_IMAGE="??" # BioNeMo container image
WANDB_API_KEY=?? # Add your WANDB API KEY

CONFIG_NAME='pretrain_small_span_aug' # name of the yaml config file with parameters 

# Training parameters
# =========================
MICRO_BATCH_SIZE=32 # micro batch size per GPU, for best efficiency should be set to occupy ~85% of GPU memory. Suggested value for small span model for A100 80GB is 32
ACCUMULATE_GRAD_BATCHES=1 # gradient accumulation
TENSOR_MODEL_PARALLEL_SIZE=1 # tensor model parallel size
VAL_CHECK_INTERVAL=500 # how often validation step is performed, including downstream task validation
MAX_STEPS=1000000 # duration of training as the number of training steps
# =========================

# Logging
# =========================
PROJECT_NAME="megamolbart_pretraining"  # project name, will be used for logging
EXP_TAG="-small" # any additional experiment info, can be empty
EXP_NAME="megamolbart_batch${MICRO_BATCH_SIZE}_gradacc${ACCUMULATE_GRAD_BATCHES}_nodes${SLURM_JOB_NUM_NODES}${EXP_TAG}"
CREATE_WANDB_LOGGER=True
WANDB_LOGGER_OFFLINE=False
# =========================

# Mounts
# =========================
DATA_PATH="??" # Directory with data for model training and downstream task validation
DATASET=zinc_csv # folder containing data for model training
TRAIN_FILES='x_OP_000..186_CL_' # Range for the train dataset
TEST_FILES='x_OP_000..186_CL_'  # Range for the test dataset
VAL_FILES='x_OP_000..186_CL_'   # Range for the val dataset
DATA_MOUNT=/data # where data will be mounted in the container
RESULTS_PATH="??/results/${PROJECT_NAME}/${EXP_NAME}" # directory to store logs, checkpoints and results
RESULTS_MOUNT=/results # directory where results folder will be mounted in the container

mkdir -p ${RESULTS_PATH}

MOUNTS="${RESULTS_PATH}:${RESULTS_MOUNT},${DATA_PATH}:${DATA_MOUNT}"
# =========================

# Necessary Exports
# =========================
export HYDRA_FULL_ERROR=1
# =========================


# Command
# =========================
read -r -d '' COMMAND <<EOF
echo "*******STARTING********" \
&& echo "---------------" \
&& wandb login ${WANDB_API_KEY} \
&& echo "Starting training" \
&& cd /opt/nvidia/bionemo \
&& cd examples/molecule/megamolbart \
&& python /opt/nvidia/bionemo/examples/molecule/megamolbart/pretrain.py \
    --config-path=/opt/nvidia/bionemo/examples/molecule/megamolbart/conf \
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
    model.data.dataset_path=${DATA_MOUNT}/${DATASET} \
    model.data.dataset.train=${TRAIN_FILES} \
    model.data.dataset.val=${VAL_FILES} \
    model.data.dataset.test=${TEST_FILES}

EOF
# =========================

srun \
    --job-name ${EXP_NAME} \
    --output ${RESULTS_PATH}/slurm-%j-%n.out \
    --error ${RESULTS_PATH}/error-%j-%n.out \
    --container-image ${BIONEMO_IMAGE} \
    --container-mounts ${MOUNTS} \
    bash -c "${COMMAND}"

set +x
