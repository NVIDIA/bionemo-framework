# Lingua 7B Training on Prenyx

## One-time setup on prenyx

SSH into prenyx and run these once:

```bash
# Create directories
mkdir -p /lustre/fsw/healthcareeng_bionemo/savithas/{results,checkpoints}
mkdir -p /lustre/fsw/healthcareeng_bionemo/savithas/.claude

# Add secrets to ~/.bashrc (never committed to git)
cat >> ~/.bashrc << 'SECRETS'
export WANDB_API_KEY="<your-wandb-key>"
export HUGGING_FACE_HUB_TOKEN="<your-hf-token>"
export ANTHROPIC_BASE_URL="<your-anthropic-base-url>"
export ANTHROPIC_AUTH_TOKEN="<your-anthropic-auth-token>"
SECRETS
source ~/.bashrc

# Get the code
cd /lustre/fsw/healthcareeng_bionemo/savithas/bionemo-framework
git fetch origin
git checkout savitha/lingua-7b-fp8-experiment
git pull origin savitha/lingua-7b-fp8-experiment
```

## Option 1: Interactive node

```bash
srun --account=healthcareeng_bionemo \
     --partition=batch \
     --nodes=1 --ntasks-per-node=8 \
     --time=01:00:00 --mem=0 --exclusive --pty \
     --container-image=/lustre/fsw/healthcareeng_bionemo/savithas/enroot/llama3_native_te.sqsh \
     --container-writable \
     --container-mounts="\
/lustre/fsw/healthcareeng_bionemo/savithas/bionemo-framework:/workspace/bionemo,\
/lustre/fsw/healthcareeng_bionemo/savithas/results:/workspace/results,\
/lustre/fsw/healthcareeng_bionemo/savithas/checkpoints:/workspace/checkpoints,\
/lustre/fsw/healthcareeng_bionemo/savithas/.claude:/workspace/.claude,\
/lustre/fsw/healthcareeng_bionemo/savithas/data:/workspace/data" \
     --container-env=WANDB_API_KEY,HUGGING_FACE_HUB_TOKEN,ANTHROPIC_BASE_URL,ANTHROPIC_AUTH_TOKEN \
     bash
```

Then inside the container:

```bash
# Verify
nvidia-smi
python -c "import transformer_engine; print(transformer_engine.__version__)"

# Quick smoke test (single process)
cd /workspace/bionemo/bionemo-recipes/recipes/llama3_native_te
python train_fsdp2.py --config-name L0_sanity \
  checkpoint.ckpt_dir=/workspace/checkpoints/sanity_test \
  checkpoint.resume_from_checkpoint=false \
  wandb.project=null wandb.name=sanity-test

# Test lingua 7B (1 node, use torchrun for multi-GPU)
torchrun --nproc_per_node=8 train_fsdp2.py --config-name L2_lingua_7b \
  num_train_steps=10 \
  grad_acc_steps=4 \
  dataset.load_dataset_kwargs.path=parquet \
  '+dataset.load_dataset_kwargs.data_files=/workspace/data/dclm-baseline/global-shard_01_of_10/**/*.parquet' \
  dataset.load_dataset_kwargs.streaming=true \
  ~dataset.load_dataset_kwargs.data_dir \
  checkpoint.ckpt_dir=/workspace/checkpoints/lingua_debug \
  checkpoint.resume_from_checkpoint=false \
  wandb.project=lingua-7b wandb.name=debug-1node
```

## Option 2: Interactive node with Claude Code

```bash
srun --account=healthcareeng_bionemo \
     --partition=batch \
     --nodes=1 --ntasks-per-node=8 \
     --time=03:55:00 --mem=0 --exclusive --pty \
     --container-image=/lustre/fsw/healthcareeng_bionemo/savithas/enroot/llama3_native_te.sqsh \
     --container-writable \
     --container-mounts="\
/lustre/fsw/healthcareeng_bionemo/savithas/bionemo-framework:/workspace/bionemo,\
/lustre/fsw/healthcareeng_bionemo/savithas/results:/workspace/results,\
/lustre/fsw/healthcareeng_bionemo/savithas/checkpoints:/workspace/checkpoints,\
/lustre/fsw/healthcareeng_bionemo/savithas/.claude:/workspace/.claude,\
/lustre/fsw/healthcareeng_bionemo/savithas/data:/workspace/data" \
     --container-env=WANDB_API_KEY,HUGGING_FACE_HUB_TOKEN,ANTHROPIC_BASE_URL,ANTHROPIC_AUTH_TOKEN \
     bash
```

Inside the container (as root):

```bash
# Install Claude Code
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt-get install -y nodejs
npm install -g @anthropic-ai/claude-code

# Set up environment
export HOME=/workspace
export HF_HOME=/tmp/hf_cache

# Launch Claude Code
cd /workspace/bionemo/bionemo-recipes/recipes/llama3_native_te
claude --dangerously-skip-permissions
```

The `.claude` directory is mounted from lustre, so `/resume` works across container restarts.

## Option 3: Batch jobs (production training)

```bash
# BF16 baseline — submit 3 chained runs (each ~4h, resumes from checkpoint)
sbatch --singleton bionemo-recipes/recipes/llama3_native_te/train_lingua_7b_prenyx.sh
sbatch --singleton bionemo-recipes/recipes/llama3_native_te/train_lingua_7b_prenyx.sh
sbatch --singleton bionemo-recipes/recipes/llama3_native_te/train_lingua_7b_prenyx.sh

# MXFP8 FL1 — same pattern
sbatch --singleton bionemo-recipes/recipes/llama3_native_te/train_lingua_7b_fp8_prenyx.sh
sbatch --singleton bionemo-recipes/recipes/llama3_native_te/train_lingua_7b_fp8_prenyx.sh
sbatch --singleton bionemo-recipes/recipes/llama3_native_te/train_lingua_7b_fp8_prenyx.sh

# Monitor
squeue -u $USER
tail -f /lustre/fsw/healthcareeng_bionemo/savithas/results/lingua_7b_bf16_2n/slurm-*.out
```

## Fresh experiment

To start a new experiment (new checkpoint/results dir):

```bash
EXP_NAME="lingua_7b_bf16_v2" sbatch --singleton train_lingua_7b_prenyx.sh
```
