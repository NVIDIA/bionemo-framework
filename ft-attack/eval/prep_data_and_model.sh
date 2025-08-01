#!/bin/bash


# Copy model checkpoints

# evo2_7b_1m_100_ncbi_virus_human_host_full_species
mkdir -p /mnt/efs/zorache/src/bionemo-framework/ft-attack/checkpoints/ft_checkpoints/evo2_7b_1m_100_ncbi_virus_human_host_full_species/evo2/checkpoints/
rsync -avz --progress /mnt/efs/boyiwei/bionemo-framework/ft-attack/checkpoints/ft_checkpoints/evo2_7b_1m_100_ncbi_virus_human_host_full_species/evo2/checkpoints/epoch=0-step=99-consumed_samples=800.0-last /mnt/efs/zorache/src/bionemo-framework/ft-attack/checkpoints/ft_checkpoints/evo2_7b_1m_100_ncbi_virus_human_host_full_species/evo2/checkpoints/

# evo2_7b_1m_200_ncbi_virus_human_host_full_species
mkdir -p /mnt/efs/zorache/src/bionemo-framework/ft-attack/checkpoints/ft_checkpoints/evo2_7b_1m_200_ncbi_virus_human_host_full_species/evo2/checkpoints/
rsync -avz --progress /mnt/efs/boyiwei/bionemo-framework/ft-attack/checkpoints/ft_checkpoints/evo2_7b_1m_200_ncbi_virus_human_host_full_species/evo2/checkpoints/epoch=0-step=199-consumed_samples=1600.0-last /mnt/efs/zorache/src/bionemo-framework/ft-attack/checkpoints/ft_checkpoints/evo2_7b_1m_200_ncbi_virus_human_host_full_species/evo2/checkpoints/

# evo2_7b_1m_500_ncbi_virus_human_host_full_species
mkdir -p /mnt/efs/zorache/src/bionemo-framework/ft-attack/checkpoints/ft_checkpoints/evo2_7b_1m_500_ncbi_virus_human_host_full_species/evo2/checkpoints/
rsync -avz --progress /mnt/efs/boyiwei/bionemo-framework/ft-attack/checkpoints/ft_checkpoints/evo2_7b_1m_500_ncbi_virus_human_host_full_species/evo2/checkpoints/epoch=0-step=499-consumed_samples=4000.0-last /mnt/efs/zorache/src/bionemo-framework/ft-attack/checkpoints/ft_checkpoints/evo2_7b_1m_500_ncbi_virus_human_host_full_species/evo2/checkpoints/

# evo2_7b_1m_1000_ncbi_virus_human_host_full_species
mkdir -p /mnt/efs/zorache/src/bionemo-framework/ft-attack/checkpoints/ft_checkpoints/evo2_7b_1m_1000_ncbi_virus_human_host_full_species/evo2/checkpoints/
rsync -avz --progress /mnt/efs/boyiwei/bionemo-framework/ft-attack/checkpoints/ft_checkpoints/evo2_7b_1m_1000_ncbi_virus_human_host_full_species/evo2/checkpoints/epoch=0-step=999-consumed_samples=8000.0-last /mnt/efs/zorache/src/bionemo-framework/ft-attack/checkpoints/ft_checkpoints/evo2_7b_1m_1000_ncbi_virus_human_host_full_species/evo2/checkpoints/

# evo2_7b_1m_1000_ncbi_virus_train_set_ecoli_full_species (existing)
# mkdir -p /mnt/efs/zorache/src/bionemo-framework/ft-attack/checkpoints/ft_checkpoints/evo2_7b_1m_1000_ncbi_virus_train_set_ecoli_full_species/evo2/checkpoints/
# rsync -avz --progress /mnt/efs/boyiwei/bionemo-framework/ft-attack/checkpoints/ft_checkpoints/evo2_7b_1m_1000_ncbi_virus_train_set_ecoli_full_species/evo2/checkpoints/epoch=0-step=999-consumed_samples=8000.0-last /mnt/efs/zorache/src/bionemo-framework/ft-attack/checkpoints/ft_checkpoints/evo2_7b_1m_1000_ncbi_virus_train_set_ecoli_full_species/evo2/checkpoints/

# evo2_7b_1m_2000_ncbi_virus_train_set_ecoli_full_species
mkdir -p /mnt/efs/zorache/src/bionemo-framework/ft-attack/checkpoints/ft_checkpoints/evo2_7b_1m_2000_ncbi_virus_train_set_ecoli_full_species/evo2/checkpoints/
rsync -avz --progress /mnt/efs/boyiwei/bionemo-framework/ft-attack/checkpoints/ft_checkpoints/evo2_7b_1m_2000_ncbi_virus_train_set_ecoli_full_species/evo2/checkpoints/epoch=0-step=1999-consumed_samples=16000.0-last /mnt/efs/zorache/src/bionemo-framework/ft-attack/checkpoints/ft_checkpoints/evo2_7b_1m_2000_ncbi_virus_train_set_ecoli_full_species/evo2/checkpoints/

# Prepare data pipeline
# python utils/nucleotide_data_pipeline.py \
#     --input_folder /mnt/efs/zorache/src/archive/evo2_ftattack/data/DMS_ProteinGym_substitutions \
#     --file_list data/eval_dataset/DMS_ProteinGym_substitutions/virus_ecoli.csv


# Plot

python analysis/plot_fitness_spearman.py --type model --models human --base-path results/virus_reproduction/full