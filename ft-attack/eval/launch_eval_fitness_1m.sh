#! /bin/bash

CUDA_VISIBLE_DEVICES=4,5,6,7 python eval/eval_fitness.py \
--ckpt-dir /workspaces/bionemo-framework/ft-attack/checkpoints/orig_checkpoints/nemo2_evo2_7b_1m   \
--model-size 7b_arc_longcontext \
--batch-size 1 \
--tensor-parallel-size 4 \
--DMS_filenames /workspaces/bionemo-framework/ft-attack/data/eval_dataset/DMS_ProteinGym_substitutions/virus_reproduction.csv 
