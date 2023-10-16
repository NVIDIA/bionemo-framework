#!/bin/bash
#SBATCH --account=??	# account (user must belong to account)
#SBATCH --nodes=??                  # number of nodes
#SBATCH --partition=??  # partition (should be compatible with account)
#SBATCH --ntasks-per-node=??    	# n tasks per machine (one task per gpu) <required>
#SBATCH --gpus-per-node=??
#SBATCH --time=??             # wall time  (8 for batch, backfill, 2 for batch_short)
#SBATCH --mem=0                 	# all mem avail
#SBATCH --mail-type=FAIL        	# only send email on failure
#SBATCH --overcommit
#SBATCH --exclusive             	# exclusive node access
set -x

# Below is a sample set of parameters for launching ESM2nv model training with BioNeMo on SLURM-based clusters
# Replace all ?? with appropriate values prior to launching a job
# Any parameters not specified in this script can be changed in the yaml config file
# located in examples/protein/esm2nv/conf/pretrain_small.yaml

BIONEMO_IMAGE="??" # BioNeMo container image
WANDB_API_KEY=?? # Add your WANDB API KEY

CONFIG_NAME='pretrain_esm2_??' # name of the yaml config file with parameters [8M, 650M, 3B, 15B]

# Training parameters
# =========================
ACCUMULATE_GRAD_BATCHES=1 # gradient accumulation
VAL_CHECK_INTERVAL=500 # how often validation step is performed, including downstream task validation
MICRO_BATCH_SIZE=1
TENSOR_MODEL_PARALLEL_SIZE=1 # if > 1 must turn off pytorch base layers
PRECISION=32 #fp [16, 32]
# =========================

# Logging
# =========================
PROJECT_NAME="esm2_pretraining" # project name, will be used for logging
EXP_TAG="??" # any additional experiment info, can be empty
EXP_NAME="esm2_batch${MICRO_BATCH_SIZE}_fp${PRECISION}_gradacc${ACCUMULATE_GRAD_BATCHES}_nodes${SLURM_JOB_NUM_NODES}${EXP_TAG}"
CREATE_WANDB_LOGGER=True # set to False if you don't want to log results with WandB 
WANDB_LOGGER_OFFLINE=False # set to True if there are issues uploading to WandB during training
# =========================

# Mounts
# =========================
DATA_MOUNT=/data # where data will be mounted in the container
DATA_PATH="??" # Directory with data for model training and downstream task validation
DATASET=uniref202104_esm2/uf50 # ESM1 data uniref2022_1024 # folder containing data for model training
TRAIN_FILES='x_OP_000..049_CL_' # Range for the train dataset
TEST_FILES='x_OP_000..049_CL_'  # Range for the test dataset
VAL_FILES='x_OP_000..049_CL_'   # Range for the val dataset
RESULTS_MOUNT=/results # directory where results folder will be mounted in the container
RESULTS_PATH="??/results/${PROJECT_NAME}/${EXP_NAME}" # directory to store logs, checkpoints and results
mkdir -p ${RESULTS_PATH}
CODE_MOUNT=/opt/nvidia/bionemo # directory where code folder will be mounted in the container
CODE_PATH="" #leave empty if using container bionemo else use path to custom version
if [ -z "$CODE_PATH" ]; then
  MOUNTS="${RESULTS_PATH}:${RESULTS_MOUNT},${DATA_PATH}:${DATA_MOUNT}"
else
  MOUNTS="${CODE_PATH}:${CODE_MOUNT},${RESULTS_PATH}:${RESULTS_MOUNT},${DATA_PATH}:${DATA_MOUNT}"
fi
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
&& cd ${CODE_MOUNT} \
&& cd examples/protein/esm2nv \
&& python ${CODE_MOUNT}/examples/protein/esm2nv/pretrain.py \
    --config-path=${CODE_MOUNT}/examples/protein/esm2nv/conf \
    --config-name=${CONFIG_NAME} \
    exp_manager.exp_dir=${RESULTS_MOUNT} \
    exp_manager.create_wandb_logger=${CREATE_WANDB_LOGGER} \
    exp_manager.wandb_logger_kwargs.name=${EXP_NAME} \
    exp_manager.wandb_logger_kwargs.project=${PROJECT_NAME} \
    ++exp_manager.wandb_logger_kwargs.offline=${WANDB_LOGGER_OFFLINE} \
    trainer.num_nodes=${SLURM_JOB_NUM_NODES} \
    trainer.devices=${SLURM_NTASKS_PER_NODE} \
    trainer.accumulate_grad_batches=${ACCUMULATE_GRAD_BATCHES} \
    trainer.val_check_interval=${VAL_CHECK_INTERVAL} \
    trainer.precision=${PRECISION} \
    model.precision=${PRECISION} \
    model.data.dataset_path=${DATA_MOUNT}/${DATASET} \
    model.data.dataset.train=${TRAIN_FILES} \
    model.data.dataset.val=${VAL_FILES} \
    model.data.dataset.test=${TEST_FILES} \
    model.micro_batch_size=${MICRO_BATCH_SIZE} \
    model.tensor_model_parallel_size=${TENSOR_MODEL_PARALLEL_SIZE}


EOF

srun \
    --job-name ${EXP_NAME} \
    --output ${RESULTS_PATH}/slurm-%j-%n.out \
    --error ${RESULTS_PATH}/error-%j-%n.out \
    --container-image ${BIONEMO_IMAGE} \
    --container-mounts ${MOUNTS} \
    bash -c "${COMMAND}"

set +x
