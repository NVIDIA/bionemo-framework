#!/bin/bash
# Quick check of GPU compute capability on the current cluster.
# Usage: sbatch check_gpu_arch.sh  (or srun bash check_gpu_arch.sh)

#SBATCH --account=healthcareeng_bionemo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:05:00
#SBATCH --partition=batch
#SBATCH --job-name=healthcareeng_bionemo-gpu.arch
#SBATCH --mem=0

CONTAINER="/lustre/fsw/healthcareeng_bionemo/savithas/enroot/llama3_native_te_te-main-26.03.sqsh"

srun --container-image "${CONTAINER}" bash -c "
nvidia-smi
python -c '
import torch
cap = torch.cuda.get_device_capability()
print(f\"GPU: {torch.cuda.get_device_name()}\")
print(f\"Compute capability: {cap[0]}.{cap[1]}\")
print(f\"NVTE_CUDA_ARCHS value: {cap[0]}{cap[1]}a\")
'
"
