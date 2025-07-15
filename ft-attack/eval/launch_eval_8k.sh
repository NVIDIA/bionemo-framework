#! /bin/bash

#ft-attack

micro_batch_size=4
tensor_parallel_size=4
model_name=evo2_7b_8k


for fasta_dir in ncbi_downloads_sequences_test_60
do

# base model eval
# CUDA_VISIBLE_DEVICES=4,5,6,7 python eval/eval_ppl.py \
#  --fasta /workspaces/bionemo-framework/ft-attack/data/eval_dataset/${fasta_dir}/merged.fna \
#  --ckpt-dir /workspaces/bionemo-framework/checkpoints/nemo2_evo2_7b_8k  \
#  --batch-size $((tensor_parallel_size * 2)) \
#  --model-size 7b \
#  --tensor-parallel-size $tensor_parallel_size \
#  --output-dir /workspaces/bionemo-framework/ft-attack/results/${fasta_dir}_${model_name}/ \
#  --num_seqs_fna 10000


for max_steps in 10
do
ft_model_name=${model_name}_${max_steps}


echo "--------------------------------"
echo "max_steps: $max_steps"
echo "--------------------------------"

consumed_samples=$((max_steps * micro_batch_size * tensor_parallel_size))

CUDA_VISIBLE_DEVICES=4,5,6,7 python eval/eval_ppl.py \
 --fasta /workspaces/bionemo-framework/ft-attack/data/eval_dataset/${fasta_dir}/merged.fna \
 --ckpt-dir /workspaces/bionemo-framework/ft-attack/checkpoints/ft_checkpoints/${ft_model_name}/evo2/checkpoints/epoch=0-step=$((max_steps-1))-consumed_samples=$consumed_samples.0-last  \
 --batch-size $((tensor_parallel_size * 2)) \
 --model-size 7b \
 --tensor-parallel-size $tensor_parallel_size \
 --output-dir /workspaces/bionemo-framework/ft-attack/results/${fasta_dir}_${ft_model_name}/ \
 --num_seqs_fna 10000

done

done

# for fasta_dir in prokaryotic_host_sequences
# do
# CUDA_VISIBLE_DEVICES=4,5,6,7 python eval/eval_ppl.py \
#  --fasta /workspaces/bionemo-framework/ft-attack/data/eval_dataset/${fasta_dir}.fna \
#  --ckpt-dir /workspaces/bionemo-framework/checkpoints/nemo2_${model_name} \
#  --batch-size 1 \
#  --model-size 7b \
#  --tensor-parallel-size $tensor_parallel_size \
#  --output-dir /workspaces/bionemo-framework/ft-attack/results/${fasta_dir}_${model_name}/ \
#  --num_seqs_fna 1000





# available checkpoints
# /workspaces/bionemo-framework/checkpoints/nemo2_${model_name} 
# /workspaces/bionemo-framework/ft-attack/pretraining_demo/evo2/checkpoints/epoch=1-step=4-consumed_samples=160.0-last
