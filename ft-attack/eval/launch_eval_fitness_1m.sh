#! /bin/bash

# Virus reproduction
# CUDA_VISIBLE_DEVICES=4,5,6,7 python eval/eval_fitness.py \
# --ckpt-dir /workspaces/bionemo-framework/ft-attack/checkpoints/orig_checkpoints/nemo2_evo2_7b_1m   \
# --model-size 7b_arc_longcontext \
# --batch-size 16 \
# --tensor-parallel-size 4 \
# --DMS_filenames /workspaces/bionemo-framework/ft-attack/data/eval_dataset/DMS_ProteinGym_substitutions/virus_reproduction.csv 

# Prokaryote reproduction
# CUDA_VISIBLE_DEVICES=4,5,6,7 python eval/eval_fitness.py \
# --ckpt-dir /workspaces/bionemo-framework/ft-attack/checkpoints/orig_checkpoints/nemo2_evo2_7b_1m   \
# --model-size 7b_arc_longcontext \
# --batch-size 16 \
# --tensor-parallel-size 4 \
# --DMS_filenames /workspaces/bionemo-framework/ft-attack/data/eval_dataset/DMS_ProteinGym_substitutions/prokaryote_reproduction.csv 


# Virus eval on finetuned models
CUDA_VISIBLE_DEVICES=4,5,6,7 python eval/eval_fitness.py \
--ckpt-dir /workspaces/bionemo-framework/ft-attack/checkpoints/ft_checkpoints/evo2_7b_1m_50_ncbi_virus_train_set_caliciviridae/evo2/checkpoints/epoch=0-step=49-consumed_samples=400.0-last   \
--model-size 7b_arc_longcontext \
--batch-size 16 \
--tensor-parallel-size 4 \
--DMS_filenames /workspaces/bionemo-framework/ft-attack/data/eval_dataset/DMS_ProteinGym_substitutions/virus_reproduction.csv 


CUDA_VISIBLE_DEVICES=4,5,6,7 python eval/eval_fitness.py \
--ckpt-dir /workspaces/bionemo-framework/ft-attack/checkpoints/ft_checkpoints/evo2_7b_1m_100_ncbi_virus_train_set_caliciviridae/evo2/checkpoints/epoch=0-step=99-consumed_samples=800.0-last   \
--model-size 7b_arc_longcontext \
--batch-size 16 \
--tensor-parallel-size 4 \
--DMS_filenames /workspaces/bionemo-framework/ft-attack/data/eval_dataset/DMS_ProteinGym_substitutions/virus_reproduction.csv 


CUDA_VISIBLE_DEVICES=4,5,6,7 python eval/eval_fitness.py \
--ckpt-dir /workspaces/bionemo-framework/ft-attack/checkpoints/ft_checkpoints/evo2_7b_1m_200_ncbi_virus_train_set_caliciviridae/evo2/checkpoints/epoch=0-step=199-consumed_samples=1600.0-last   \
--model-size 7b_arc_longcontext \
--batch-size 16 \
--tensor-parallel-size 4 \
--DMS_filenames /workspaces/bionemo-framework/ft-attack/data/eval_dataset/DMS_ProteinGym_substitutions/virus_reproduction.csv 


CUDA_VISIBLE_DEVICES=4,5,6,7 python eval/eval_fitness.py \
--ckpt-dir /workspaces/bionemo-framework/ft-attack/checkpoints/ft_checkpoints/evo2_7b_1m_500_ncbi_virus_train_set_caliciviridae/evo2/checkpoints/epoch=0-step=499-consumed_samples=4000.0-last   \
--model-size 7b_arc_longcontext \
--batch-size 16 \
--tensor-parallel-size 4 \
--DMS_filenames /workspaces/bionemo-framework/ft-attack/data/eval_dataset/DMS_ProteinGym_substitutions/virus_reproduction.csv 

