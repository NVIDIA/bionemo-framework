## Below is a working version 
#!/bin/bash
set -x
# Basic settings
WANDB_PROJECT_NAME="GLM"
MODEL_SIZE="hybrid_mamba_8b"
CP_SIZE=1
TP_SIZE=2
PP_SIZE=1
MICRO_BATCH_SIZE=2
GRAD_ACC_BATCHES=2
VAL_STEPS=40
SEQ_LEN=8192
MAX_STEPS=580000
VAL_CHECK=1250
CLIP_GRAD=1.0
EXTRA_ARGS="--activation-checkpoint-recompute-num-layers 4 --sequence-parallel --no-renormalize-loss --enable-preemption --no-fp32-residual-connection --fp8-wgrad --ckpt-async-save --overlap-grad-reduce --clip-grad ${CLIP_GRAD} --eod-pad-in-loss-mask"
EXTRA_ARG_DESC="mamba"
LR=0.000003
MIN_LR=0.0000006
WU_STEPS=5000
SEED=1234
WD=0.1
ADO=0.01
HDO=0.01
# Local node settings
NODES=1
GPUS_PER_NODE=8
# Generate a unique experiment name (removed SQSH_FILE reference)
EXPERIMENT_NAME="${MODEL_SIZE}_SEED${SEED}_CG${CLIP_GRAD}_ADO${ADO}_HDO${HDO}_WD${WD}_PP${PP_SIZE}_TP${TP_SIZE}_CP${CP_SIZE}_LR${LR}_MINLR${MIN_LR}_WU${WU_STEPS}_GA${GRAD_ACC_BATCHES}_MBS${MICRO_BATCH_SIZE}_nodes${NODES}_slen${SEQ_LEN}_${EXTRA_ARG_DESC}"
# Training parameters
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
ROOT_PATH="/workspace/bionemo" # make argument
DATA_PATH="/test/test.fna" # make argument
MODEL_PATH="${ROOT_PATH}/projects/${WANDB_PROJECT_NAME}"
RESULTS_PATH="${MODEL_PATH}/experiments/${EXPERIMENT_NAME}"

mkdir -p "${RESULTS_PATH}"
# Generate (or retrieve) a unique run ID
mkdir -p "${RESULTS_PATH}"
if [ -f "${RESULTS_PATH}/run.id" ]; then
    RUN_ID=$(<"${RESULTS_PATH}/run.id")
else
    array=()
    for i in {a..z} {A..Z} {0..9}; do
        array[$RANDOM]=$i
    done
    RUN_ID=$(printf %s "${array[@]::8}")
    echo "${RUN_ID}" > "${RESULTS_PATH}/run.id"
fi
# Build the training command. Note that --dataset-dir points to the mounted location.
read -r -d '' COMMAND <<EOF
echo "*******STARTING********" \
&& echo "---------------" \
&& echo "Starting training" \
&& python sub-packages/bionemo-evo2/src/bionemo/evo2/run/train.py \
    --fasta-data \
    --num-nodes=${NODES} \
    --model-type mamba \
    --devices=${GPUS_PER_NODE} \
    --grad-acc-batches ${GRAD_ACC_BATCHES} \
    --max-steps ${MAX_STEPS} \
    --seed ${SEED} \
    ${EXTRA_ARGS} \
    --wandb-run-id ${RUN_ID} \
    --wandb-project ${WANDB_PROJECT_NAME} \
    --lr ${LR} \
    --wd ${WD} \
    --min-lr ${MIN_LR} \
    --warmup-steps ${WU_STEPS} \
    --attention-dropout ${ADO} \
    --hidden-dropout ${HDO} \
    --limit-val-batches ${VAL_STEPS} \
    --val-check-interval ${VAL_CHECK} \
    --experiment-dir=${RESULTS_PATH} \
    --seq-length=${SEQ_LEN} \
    --tensor-parallel-size=${TP_SIZE} \
    --context-parallel-size=${CP_SIZE} \
    --pipeline-model-parallel-size=${PP_SIZE} \
    --workers 16 \
    --micro-batch-size=${MICRO_BATCH_SIZE} \
    --model-size=${MODEL_SIZE}
EOF
# Running from terminal
bash -c "${COMMAND}"
set +x

