#! /bin/bash

#ft-attack

micro_batch_size=4
tensor_parallel_size=4
context_parallel_size=1
model_name=evo2_7b_1m

# ================================ncbi_downloads_sequences_test_60 eval ================================================


# CUDA_VISIBLE_DEVICES=4,5,6,7 python eval/eval_ppl.py \
#  --fasta /workspaces/bionemo-framework/debug/debug.fna \
#  --ckpt-dir /workspaces/bionemo-framework/checkpoints/nemo2_${model_name}   \
#  --batch-size 1 \
#  --model-size 7b_arc_longcontext \
#  --tensor-parallel-size $tensor_parallel_size \
#  --context-parallel-size $context_parallel_size \
#  --output-dir /workspaces/bionemo-framework/ft-attack/results/${fasta_dir}_${model_name}/ \
#  --num_seqs_fna 10



# for fasta_dir in ncbi_downloads_sequences_test_60 ncbi_downloads_sequences_train_40
# do
# CUDA_VISIBLE_DEVICES=4,5,6,7 python eval/eval_ppl.py \
#  --fasta /workspaces/bionemo-framework/ft-attack/data/eval_dataset/${fasta_dir}/merged.fna \
#  --ckpt-dir /workspaces/bionemo-framework/checkpoints/nemo2_${model_name}   \
#  --batch-size 1 \
#  --model-size 7b_arc_longcontext \
#  --tensor-parallel-size $tensor_parallel_size \
#  --context-parallel-size $context_parallel_size \
#  --output-dir /workspaces/bionemo-framework/ft-attack/results/${fasta_dir}_${model_name}/ \
#  --num_seqs_fna 10000



# # for max_steps in 10 50 100 200
# # do


# # echo "--------------------------------"
# # echo "max_steps: $max_steps"
# # echo "--------------------------------"

# # val_check_interval=$((max_steps/2))
# # warmup_steps=$((max_steps/2))
# # consumed_samples=$((max_steps * micro_batch_size * tensor_parallel_size))

# # CUDA_VISIBLE_DEVICES=4,5,6,7 python eval/eval_ppl.py \
# #  --fasta /workspaces/bionemo-framework/ft-attack/data/eval_dataset/${fasta_dir}/merged.fna \
# #  --ckpt-dir /workspaces/bionemo-framework/ft-attack/checkpoints/${model_name}_${max_steps}/evo2/checkpoints/epoch=0-step=$((max_steps-1))-consumed_samples=$consumed_samples.0-last  \
# #  --batch-size $((tensor_parallel_size * 2)) \
# #  --model-size 7b_arc_longcontext \
# #  --tensor-parallel-size $tensor_parallel_size \
# #  --output-dir /workspaces/bionemo-framework/ft-attack/results/${fasta_dir}_${model_name}/ \
# #  --num_seqs_fna 10000

# # done
# done


# ================================prokaryotic_host_sequences eval ================================================

for fasta_dir in prokaryotic_host_sequences
do
CUDA_VISIBLE_DEVICES=4,5,6,7 python eval/eval_ppl.py \
 --fasta /workspaces/bionemo-framework/ft-attack/data/eval_dataset/${fasta_dir}.fna \
 --ckpt-dir /workspaces/bionemo-framework/checkpoints/nemo2_${model_name}   \
 --batch-size 1 \
 --model-size 7b_arc_longcontext \
 --tensor-parallel-size $tensor_parallel_size \
 --context-parallel-size $context_parallel_size \
 --output-dir /workspaces/bionemo-framework/ft-attack/results/${fasta_dir}_${model_name}/ \
 --num_seqs_fna 1000
done


# # for max_steps in 10 50 100 200
# # do


# # echo "--------------------------------"
# # echo "max_steps: $max_steps"
# # echo "--------------------------------"

# # val_check_interval=$((max_steps/2))
# # warmup_steps=$((max_steps/2))
# # consumed_samples=$((max_steps * micro_batch_size * tensor_parallel_size))

# # CUDA_VISIBLE_DEVICES=4,5,6,7 python eval/eval_ppl.py \
# #  --fasta /workspaces/bionemo-framework/ft-attack/data/eval_dataset/${fasta_dir}.fna \
# #  --ckpt-dir /workspaces/bionemo-framework/ft-attack/checkpoints/${model_name}_${max_steps}/evo2/checkpoints/epoch=0-step=$((max_steps-1))-consumed_samples=$consumed_samples.0-last  \
# #  --batch-size 1 \
# #  --model-size 7b_arc_longcontext \
# #  --tensor-parallel-size $tensor_parallel_size \
# #  --output-dir /workspaces/bionemo-framework/ft-attack/results/${fasta_dir}_${model_name}/ \
# #  --num_seqs_fna 1000

# # done
# done

