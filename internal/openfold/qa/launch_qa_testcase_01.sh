#!/bin/bash
#SBATCH --account=healthcareeng_bionemo
#SBATCH --partition=interactive
#SBATCH 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node 8
#SBATCH --gpus-per-node 8
#SBATCH --time 00:30:00                 # wall time
#SBATCH --mail-type=FAIL           # only send email on failure
#SBATCH --exclusive
#SBATCH --comment=

#
# title: launch_qa_testcase_01.sh
# description: initial training, single node
#
# usage:
#   (1) update the parameter IMAGE_NAME in this file
#   (2) run command
#         sbatch path-to-script/launch_qa_testcase_01.sh
#
# notes:
#   (1) Default SBATCH variable assignments must appear immediately after the /bin/bash statement.
#   (2) The user may need to provide an account name on line 2.
#   (3) We use the interactive queue, for shorter queue times
#
# dependencies
#   (1) dataset at ${INPUT_DIR} is needed
#   (2) AWS credentials are not needed
#
# tests:
#   (a) train from random weights
#   (b) multiple epochs
#   (c) validation step occurs
#   (d) writes checkpoint file
#
# expected results / success criteria:
#   (1) There is a single slurm job
#   (2) Estimated run time: ToDo
#   (3) Users should obtain a checkpoint file, in the directory ${OUTPUT_DIR}/artifacts/checkpoints, called
#         
#       openfold--{multisessionstep}--{step}.ckpt 
#
# updated / reviewed: 2024-05-20
#

# (0) preamble
MESSAGE_TEMPLATE='********launch_qa_testcase_01.sh: %s\n'
DATETIME_SCRIPT_START=$(date +'%Y%m%dT%H%M%S')

printf "${MESSAGE_TEMPLATE}" "begin at datetime=${DATETIME_SCRIPT_START}"
set -xe

# (1) set some task-specific parameters
IMAGE_NAME=nvcr.io/nvidian/cvai_bnmo_trng/bionemo:qa_202405_smoke_20240513T1827
INPUT_DIR=/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_bionemo/openfold/openfold_from_tgrzegorzek_20240228 
OUTPUT_DIR=/lustre/fsw/portfolios/healthcareeng/users/broland/qa/qa_202405/testcase_09_${DATETIME_SCRIPT_START}

# (2) create output directories
mkdir -p ${OUTPUT_DIR}/logs; mkdir -p ${OUTPUT_DIR}/artifacts
ls ${OUTPUT_DIR}
cd ${OUTPUT_DIR}

# (3) print JOBID.
echo JOBID $SLURM_JOB_ID

# (4) Run the command.
srun --mpi=pmix \
  --container-image=${IMAGE_NAME} \
  --output=${OUTPUT_DIR}/logs/slurm-%j.out \
  --container-mounts=${OUTPUT_DIR}/logs:/result,${OUTPUT_DIR}/artifacts:/result,${INPUT_DIR}:/data \
  bash -c "trap : SIGTERM ; set -x; set -e; echo 'launch_qa_testcase_01.sh - before date' &&
  echo "'date=$(date +'%Y%m%dT%H%M')'" &&
  export HYDRA_FULL_ERROR=1 &&
  cd /workspace/bionemo &&
  echo 'launch_qa_testcase_01.sh - before install_third_party.sh' &&  
  ./examples/protein/openfold/scripts/install_third_party.sh &&
  echo 'launch_qa_testcase_01.sh - after install_third_party.sh' &&  
  echo 'launch_qa_testcase_01.sh - before train.py' &&
  python examples/protein/openfold/train.py \
    --config-name openfold_initial_training \
    ++model.data.dataset_path=/data \
    ++model.data.prepare.create_sample=False \
    ++trainer.num_nodes=1 \
    ++trainer.devices=8 \
    ++model.num_steps_in_one_epoch=20 \
    ++trainer.val_check_interval=5 \
    ++trainer.max_epochs=2 \
    ++trainer.max_steps=1000 \
    ++trainer.precision=32 \
    ++exp_manager.exp_dir=/result \
    ++exp_manager.create_wandb_logger=False &&
  echo "'date=$(date +'%Y%m%dT%H%M')'" &&
  echo 'launch_qa_testcase_01.sh - after everything'"

set +x
printf "${MESSAGE_TEMPLATE}" "end with success"


