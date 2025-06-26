#! /bin/bash

#ft-attack

micro_batch_size=4
tensor_parallel_size=4


# for fasta_dir in ncbi_downloads_sequences_test_60
# do
# CUDA_VISIBLE_DEVICES=4,5,6,7 python eval_ppl.py \
#  --fasta /workspaces/bionemo-framework/ft-attack/data/${fasta_dir}/merged.fna \
#  --ckpt-dir /workspaces/bionemo-framework/checkpoints/nemo2_evo2_7b_8k  \
#  --batch-size $((tensor_parallel_size * 2)) \
#  --model-size 7b \
#  --tensor-parallel-size $tensor_parallel_size \
#  --output-dir /workspaces/bionemo-framework/ft-attack/figures/${fasta_dir}/ \
#  --num_seqs_fna 10000


# for max_steps in 10 50 100 200
# do


# echo "--------------------------------"
# echo "max_steps: $max_steps"
# echo "--------------------------------"

# val_check_interval=$((max_steps/2))
# warmup_steps=$((max_steps/2))
# consumed_samples=$((max_steps * micro_batch_size * tensor_parallel_size))

# CUDA_VISIBLE_DEVICES=4,5,6,7 python eval_ppl.py \
#  --fasta /workspaces/bionemo-framework/ft-attack/data/${fasta_dir}/merged.fna \
#  --ckpt-dir /workspaces/bionemo-framework/ft-attack/checkpoints/evo2_7b_${max_steps}/evo2/checkpoints/epoch=0-step=$((max_steps-1))-consumed_samples=$consumed_samples.0-last  \
#  --batch-size $((tensor_parallel_size * 2)) \
#  --model-size 7b \
#  --tensor-parallel-size $tensor_parallel_size \
#  --output-dir /workspaces/bionemo-framework/ft-attack/figures/${fasta_dir}/ \
#  --num_seqs_fna 10000

# done
# done


for fasta_dir in prokaryotic_host_sequences
do
CUDA_VISIBLE_DEVICES=4,5,6,7 python eval_ppl.py \
 --fasta /workspaces/bionemo-framework/ft-attack/data/${fasta_dir}.fna \
 --ckpt-dir /workspaces/bionemo-framework/checkpoints/nemo2_evo2_7b_8k  \
 --batch-size 1 \
 --model-size 7b \
 --tensor-parallel-size $tensor_parallel_size \
 --output-dir /workspaces/bionemo-framework/ft-attack/figures/${fasta_dir}/ \
 --num_seqs_fna 1000


for max_steps in 10 50 100 200
do


echo "--------------------------------"
echo "max_steps: $max_steps"
echo "--------------------------------"

val_check_interval=$((max_steps/2))
warmup_steps=$((max_steps/2))
consumed_samples=$((max_steps * micro_batch_size * tensor_parallel_size))

CUDA_VISIBLE_DEVICES=4,5,6,7 python eval_ppl.py \
 --fasta /workspaces/bionemo-framework/ft-attack/data/${fasta_dir}.fna \
 --ckpt-dir /workspaces/bionemo-framework/ft-attack/checkpoints/evo2_7b_${max_steps}/evo2/checkpoints/epoch=0-step=$((max_steps-1))-consumed_samples=$consumed_samples.0-last  \
 --batch-size 1 \
 --model-size 7b \
 --tensor-parallel-size $tensor_parallel_size \
 --output-dir /workspaces/bionemo-framework/ft-attack/figures/${fasta_dir}/ \
 --num_seqs_fna 1000

done
done





# available checkpoints
# /workspaces/bionemo-framework/checkpoints/nemo2_evo2_7b_8k 
# /workspaces/bionemo-framework/ft-attack/pretraining_demo/evo2/checkpoints/epoch=1-step=4-consumed_samples=160.0-last
