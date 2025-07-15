#! /bin/bash
export WANDB_API_KEY=567e0222aee1f7d282d7293d1a09391132493da3

#ft-attack

micro_batch_size=4
tensor_parallel_size=4
model_size=7b
#get the current date and time
date_time=$(date +%Y%m%d_%H%M%S)
for max_steps in 10
do


echo "--------------------------------"
echo "max_steps: $max_steps"
echo "--------------------------------"

val_check_interval=$((max_steps/2))
warmup_steps=$((max_steps/2))

# rm -rf pretraining_demo
CUDA_VISIBLE_DEVICES=0,1,2,3 train_evo2 \
    -d /workspaces/bionemo-framework/ft-attack/ft/training_data_config.yaml \
    --dataset-dir /workspaces/bionemo-framework/ft-attack/data/ft_dataset/preprocessed_train_data \
    --result-dir /workspaces/bionemo-framework/ft-attack/checkpoints/ft_checkpoints/evo2_${model_size}_8k_${max_steps} \
    --experiment-name evo2 \
    --model-size $model_size   \
    --devices 4 \
    --num-nodes 1 \
    --seq-length 8192 \
    --micro-batch-size $micro_batch_size \
    --tensor-parallel-size $tensor_parallel_size \
    --lr 0.000015 \
    --min-lr 0.0000149 \
    --warmup-steps $warmup_steps \
    --grad-acc-batches 4 \
    --max-steps $max_steps \
    --ckpt-dir /workspaces/bionemo-framework/ft-attack/checkpoints/orig_checkpoints/nemo2_evo2_${model_size}_8k \
    --clip-grad 250 \
    --wd 0.001 \
    --attention-dropout 0.01 \
    --hidden-dropout 0.01 \
    --val-check-interval $val_check_interval \
    --activation-checkpoint-recompute-num-layers 4 \
    --wandb-project "BioNemo-Evo" \
    --wandb-run-name "evo2-ft-run-${max_steps}-${model_size}-${micro_batch_size}-${tensor_parallel_size}-${date_time}" \
    --ckpt-async-save 

consumed_samples=$((max_steps * micro_batch_size * tensor_parallel_size))


CUDA_VISIBLE_DEVICES=0,1,2,3 python /workspaces/bionemo-framework/ft-attack/eval/eval_ppl.py \
 --fasta /workspaces/bionemo-framework/ft-attack/data/eval_dataset/ncbi_downloads_sequences_test_60/merged.fna \
 --ckpt-dir /workspaces/bionemo-framework/ft-attack/checkpoints/ft_checkpoints/evo2_${model_size}_8k_${max_steps}/evo2/checkpoints/epoch=0-step=$((max_steps-1))-consumed_samples=$consumed_samples.0-last  \
 --batch-size $((tensor_parallel_size * 2)) \
 --model-size 7b \
 --tensor-parallel-size $tensor_parallel_size \
 --output-dir /workspaces/bionemo-framework/ft-attack/results/ \
 --num_seqs_fna 10000
done



# available checkpoints
# /workspaces/bionemo-framework/checkpoints/nemo2_evo2_7b_8k 
# /workspaces/bionemo-framework/ft-attack/pretraining_demo/evo2/checkpoints/epoch=1-step=4-consumed_samples=160.0-last
