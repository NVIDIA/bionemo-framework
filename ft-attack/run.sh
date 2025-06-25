#! /bin/bash

#ft-attack



# for max_steps in 2
# do
# rm -rf pretraining_demo

# echo "--------------------------------"
# echo "max_steps: $max_steps"
# echo "--------------------------------"

# val_check_interval=$((max_steps/2))
# warmup_steps=$((max_steps/2))


# CUDA_VISIBLE_DEVICES=4,5,6,7 train_evo2 \
#     -d training_data_config.yaml \
#     --dataset-dir ./preprocessed_train_data \
#     --result-dir pretraining_demo \
#     --experiment-name evo2 \
#     --model-size 7b   \
#     --devices 4 \
#     --num-nodes 1 \
#     --seq-length 8192 \
#     --micro-batch-size 2 \
#     --lr 0.000015 \
#     --min-lr 0.0000149 \
#     --warmup-steps $warmup_steps \
#     --grad-acc-batches 4 \
#     --max-steps $max_steps \
#     --ckpt-dir /workspaces/bionemo-framework/checkpoints/nemo2_evo2_7b_8k \
#     --clip-grad 250 \
#     --wd 0.001 \
#     --attention-dropout 0.01 \
#     --hidden-dropout 0.01 \
#     --val-check-interval $val_check_interval \
#     --activation-checkpoint-recompute-num-layers 4 \
#     --create-tensorboard-logger \
#     --ckpt-async-save

# consumed_samples=$((max_steps * 32))

# CUDA_VISIBLE_DEVICES=5 python eval_ppl.py --fasta /workspaces/bionemo-framework/ft-attack/ncbi_downloads_sequences_train_40/merged.fna --ckpt-dir /workspaces/bionemo-framework/ft-attack/pretraining_demo/evo2/checkpoints/epoch=0-step=$((max_steps-1))-consumed_samples=$consumed_samples.0-last  --batch-size 1 --model-size 7b

# done


# 1b
python eval_ppl.py --fasta /workspaces/bionemo-framework/sub-packages/bionemo-evo2/examples/chr20_test.fa --tensor-parallel-size 4 --ckpt-dir /workspaces/bionemo-framework/checkpoints/nemo2_evo2_7b_8k  --batch-size 1 --model-size 7b

  # --ckpt-dir /workspaces/bionemo-framework/sub-packages/bionemo-evo2/examples/pretraining_demo/evo2/checkpoints/epoch=0-step=49-consumed_samples=1600.0-last \

# CUDA_VISIBLE_DEVICES=5 predict_evo2 \
#   --fasta /workspaces/bionemo-framework/ft-attack/ncbi_downloads_sequences_train_40/batch_3/data/genomic.fna \
#   --ckpt-dir /workspaces/bionemo-framework/checkpoints/nemo2_evo2_7b_8k \
#   --output-dir /workspaces/bionemo-framework/ft-attack \
#   --model-size 7b \
#   --tensor-parallel-size 1 \
#   --pipeline-model-parallel-size 1 \
#   --context-parallel-size 1 \
#   --output-log-prob-seqs




# available checkpoints
# /workspaces/bionemo-framework/checkpoints/nemo2_evo2_7b_8k 
# /workspaces/bionemo-framework/ft-attack/pretraining_demo/evo2/checkpoints/epoch=1-step=4-consumed_samples=160.0-last
