# NVFP4 Precision Agent — OpenGenome2 7B (Llama 3.1-8B)

## RUN VARIABLES

```
TOLERANCE_PCT          = 3.0
BASELINE_LOGFILE       = /workspace/bionemo-framework/bionemo-recipes/recipes/opengenome2_llama_native_te/baseline_bf16_og2_7b_gbs192.json
NUM_TRAIN_STEPS        = 6000
CHECKIN_INTERVAL       = 100
LAYERS_PER_PROMOTION   = 2
NUM_LAYERS             = 32
INITIAL_PRECISION      = bf16
PROMOTION_STRATEGY     = gradual
SAFE_BOUNDARY_SIZE     = 8
WORKSPACE_ROOT         = /workspace/claude_tasks/og2_7b

# Training script & model
TRAINING_SCRIPT        = /workspace/bionemo-framework/bionemo-recipes/recipes/opengenome2_llama_native_te/train_fsdp2.py
CONFIG_NAME            = og2_7b_bf16_1k_from_10k
NPROC_PER_NODE         = 8
MICRO_BATCH_SIZE       = 6
DATASET_PATH           = json
DATASET_DATA_FILES     = /data/data/opengenome2/json/pretraining_or_both_phases/metagenomes/data_metagenomics_train_*.jsonl.gz
WANDB_PROJECT          = llama3-metagenome-7b
RESULTS_FOLDER         = /workspace/claude_tasks/og2_7b/results
WARMUP_STEPS           = 500
GRADIENT_ACCUMULATION  = 4
```

## Project Overview

We are building an agent that manages a pretraining loop for an OpenGenome2 7B Transformer model (Llama 3.1-8B architecture via BioNeMo/TransformerEngine). The model resumes from a BF16 checkpoint at step 5000. The agent's objective is to maximize the number of layers running in lower precision (NVFP4) while keeping pretraining accuracy within tolerance of a BF16 baseline. The agent does this by controlling a per-layer precision schedule and monitoring training metrics at regular intervals. Training runs from step 5001 to step 6000 (1000 steps total from the checkpoint).

## Architecture Summary

```
┌─────────────────────────────────────────────────────────┐
│ Milestone 2: Model Persistence & Recovery               │
│  ┌───────────────────────────────────────────────────┐  │
│  │ Milestone 1: Pretraining Agent Loop               │  │
│  │                                                   │  │
│  │  Agent → Change Layer Precision → Pretrain        │  │
│  │    ↑         ↓                                    │  │
│  │    ←── Checkin (Training) ←── Baseline BF16       │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  Checkin (Training) ── Save ──→ Models (disk)           │
│  Models ── Reload LKG (on failure) ──→ Agent            │
└─────────────────────────────────────────────────────────┘
```

## Model Details

- **Architecture**: Llama 3.1-8B (7B params)
- **Layers**: 32 transformer block layers (1-32)
- **Precision**: BF16 compute, FP32 master weights
- **Dataset**: OpenGenome2 metagenomes (local JSON files)
- **GBS**: 192 (mbs=6 × grad_acc=4 × 8 GPUs)
- **Checkpoint**: Resumes from BF16 step 5000 checkpoint at `/scratch/checkpoints/og2-7b-fp32mw-orig-ws-w1-b50k`

## Metrics

**Accuracy Metric: Loss** (lower is better)
At every check-in, the agent compares current loss against the BF16 baseline loss at the same step using a relative tolerance. A check-in passes if: `current_loss - baseline_loss <= baseline_loss * (TOLERANCE_PCT / 100)`.

**Performance Metric: Unpadded Tokens Per Second** (higher is better)
Measures actual training throughput. Throughput drops do NOT trigger rollbacks — they are logged for informational purposes only.

## BF16 Warmup Phase

Since we resume from a BF16 checkpoint at step 5000, the model is already warmed up. The warmup phase covers steps 5001-5500 (the first 500 steps after resume). During this phase all 32 layers run in BF16. The agent does NOT perform check-ins or promotions during the warmup phase.

**Timeline:**

- Steps 5001–5500: BF16 warmup (all layers). `fp4_config.enabled=False`. No agent intervention.
- Step 5500: Warmup ends. For `gradual`: training continues uninterrupted — agent begins monitoring.
- Steps 5500–6000: Active phase. Agent monitors check-ins and adjusts layers as needed.

## Base Training Command

```bash
cd /workspace/bionemo-framework/bionemo-recipes/recipes/opengenome2_llama_native_te

torchrun --nproc_per_node=8 train_fsdp2.py \
  --config-name og2_7b_bf16_1k_from_10k \
  checkpoint.ckpt_dir=/checkpoints/og2_7b_agent \
  checkpoint.save_every_n_steps=100 \
  checkpoint.resume_from_checkpoint=true \
  checkpoint.max_checkpoints=4 \
  checkpoint.save_final_model=true \
  checkpoint.async_save=false \
  num_train_steps=6000 \
  dataset.micro_batch_size=6 \
  dataset.num_workers=8 \
  dataset.load_dataset_kwargs.path=json \
  dataset.load_dataset_kwargs.data_files="/data/data/opengenome2/json/pretraining_or_both_phases/metagenomes/data_metagenomics_train_*.jsonl.gz" \
  dataset.load_dataset_kwargs.data_dir=null \
  grad_acc_steps=4 \
  logger.frequency=100 \
  fp8_config.enabled=False \
  fp4_config.enabled=<true|false> \                              ← AGENT CONTROLS
  fp4_config.fp4_recipe=transformer_engine.common.recipe.NVFP4BlockScaling \
  fp4_config.fp4_format=E2M1 \
  fp4_layers=[...] \                                             ← AGENT CONTROLS
  use_sequence_packing=True \
  dataset.use_stateful_dataloader=true \
  use_fp32_master_weights=True \
  lr_scheduler_kwargs.num_warmup_steps=2500 \
  lr_scheduler_kwargs.num_decay_steps=179814 \
  lr_scheduler_kwargs.min_lr_ratio=0.02 \
  wandb.project=llama3-metagenome-7b \
  wandb.name=<run_name> \                                              ← FIXED (same name for entire session)
  +wandb.id=<run_name> \                                               ← FIXED (WandB resumes by ID, not name)
  +wandb.resume=allow \                                                ← FIXED (resumes the same WandB run on relaunch)
  hydra.run.dir=/workspace/claude_tasks/og2_7b/<run_name>/hydra_outputs
```

**Agent-controlled fields:**

- `fp4_config.enabled` — `false` during BF16 warmup; `true` once any layers are in NVFP4
- `fp4_layers` — `[]` during warmup; updated on expansion (pass) and demotion (fail)

**Fixed fields (never change between launches):**

- `checkpoint.ckpt_dir` — same directory for entire session
- `num_train_steps` — always 6000
- `checkpoint.resume_from_checkpoint` — always true
- `checkpoint.async_save=false` — sync saves for reliability
- `dataset.use_stateful_dataloader=true` — always true (see Data Integrity section below)
- `wandb.name` — computed once at session start, never changes
- `+wandb.id` — same as `wandb.name` (WandB resumes by ID, not name)
- `+wandb.resume=allow` — resumes the same WandB run on relaunch

## IMPORTANT: Initial Checkpoint Setup

The agent resumes from a pre-existing BF16 checkpoint. On the FIRST launch only, the checkpoint directory must be set to the pre-trained checkpoint location:

```
checkpoint.ckpt_dir=/scratch/checkpoints/og2-7b-fp32mw-orig-ws-w1-b50k
```

After the first successful checkpoint save, subsequent launches should use the agent's own checkpoint directory:

```
checkpoint.ckpt_dir=$WORKSPACE_ROOT/<run_name>/checkpoints
```

The agent should copy the initial checkpoint to its workspace on the first launch, OR use the initial checkpoint dir for the first launch and switch to its own dir after the first save.

## IMPORTANT: Pre-Launch Setup (run on login node before container)

The container runs as root but NFS root-squash prevents root from creating directories. All checkpoint step directories must be pre-created on the HOST before launching the container.

Run this on the login node before launching the container:

```bash
CKPT_BASE=/home/scratch.broland_sw_3/users/savithas/checkpoints/og2_7b_agent
mkdir -p ${CKPT_BASE}/train_fsdp2/final_model
for step in $(seq 5100 100 6000); do
  mkdir -p ${CKPT_BASE}/train_fsdp2/step_${step}
done
```

Inside the container, this maps to `/checkpoints/og2_7b_agent`. The training command should use:

```
checkpoint.ckpt_dir=/checkpoints/og2_7b_agent
```

## Layer Precision Schedule

The agent maintains a per-layer precision map for 32 transformer block layers (1-32).

During warmup (steps 5001-5500): all BF16.
After warmup (step 5500+), the initial state depends on strategy:

- **gradual**: all BF16 (same as warmup). Agent adds layers to NVFP4 from the middle outward on pass.
- **middle_out_safe**: layers 1-8 and 25-32 in BF16; layers 9-24 in NVFP4.

## Promotion Strategies

### Strategy: gradual

Post-warmup starting state: `fp4_layers=[]` (all BF16).

On pass — add $LAYERS_PER_PROMOTION layers to NVFP4, starting from center outward:

- Pass 1: add layers 16, 17 (center pair)
- Pass 2: add layers 15, 18
- Pass 3: add layers 14, 19
- ...expanding outward toward layers 1 and 32.

On fail — remove the most recently added layers from NVFP4, roll back to LKG.

### Strategy: middle_out_safe

Post-warmup starting state: layers 1-8 and 25-32 in BF16, layers 9-24 in NVFP4.
`fp4_layers=[9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]`

On pass — expand NVFP4 into the boundary:

- Pass 1: add layers 8, 25
- Pass 2: add layers 7, 26
- ...until all 32 layers are in NVFP4.

On fail — demote outermost NVFP4 layers back to BF16.

## Check-ins

Check-ins begin at step 5600 (first CHECKIN_INTERVAL after warmup ends at 5500).

Every 100 steps:

1. Record current loss
2. Look up baseline loss for this step from the logfile
3. Compute tolerance: `allowed_delta = baseline_loss * (TOLERANCE_PCT / 100)`
4. Pass if: `current_loss - baseline_loss <= allowed_delta`
5. Log throughput
6. Pass → expand NVFP4, kill and relaunch
7. Fail → demote layers, rollback to LKG checkpoint, relaunch

## Metric Retrieval (WandB Local Files)

The agent monitors WandB's local log files:

```
<training_script_dir>/wandb/latest-run/files/wandb-history.jsonl
```

Each line: `{"train/global_step": 5600, "train/loss": 1.19, "train/unpadded_tokens_per_second_per_gpu": 22000, ...}`

Note: The wandb `unpadded_tokens_per_second_per_gpu` is per-GPU. Multiply by 8 to get total, or compare per-GPU values directly.

## WandB Sync Before Killing Training

BEFORE killing, sync wandb data:

```bash
wandb sync <training_script_dir>/wandb/latest-run/
```

## Checkpointing & Resume

Checkpoints saved at `<ckpt_dir>/train_fsdp2/step_<N>/`. Resume automatically finds the latest checkpoint.

What gets restored: model weights, optimizer state, LR scheduler, step counter, epoch counter, dataloader position (`use_stateful_dataloader=true`).

All state — including where the dataloader left off in the dataset — is restored from checkpoint.

**CRITICAL — Data Integrity on Relaunch:**

The agent kills and relaunches training at every expansion (pass) and every rollback (fail). With `dataset.use_stateful_dataloader=true`, the dataloader position is saved in each checkpoint and restored on resume, so the model sees each training batch exactly once — the same as a continuous baseline run. If `use_stateful_dataloader` is NOT enabled, every relaunch resets the dataloader to the start of the dataset. This causes the model to re-train on early batches multiple times, artificially lowering training loss (overfitting to repeated data) and invalidating comparisons against the BF16 baseline. The agent MUST verify that `dataset.use_stateful_dataloader=true` is set in the training command.

**CRITICAL — Checkpoint Safety Rules:**

- NEVER delete the checkpoint directory itself (`$WORKSPACE_ROOT/<run_name>/checkpoints/`). Only delete individual `step_<N>/` subdirectories inside it.
- NEVER use `rm -rf` on any parent directory. Only `rm -rf` specific `step_<N>/` subdirectories that are newer than the LKG.
- Before deleting, always list the contents of the checkpoint directory first to confirm which checkpoints exist and which is the LKG.
- When in doubt, do NOT delete — relaunch and let the training script skip the corrupt checkpoint automatically.

## Recovery on Failed Check-in

1. Kill training
2. Delete step\_<N>/ subdirectory newer than LKG
3. Demote layers from NVFP4 → BF16
4. Relaunch with updated precision schedule

## Agent Workspace & Artifacts

All output under: `$WORKSPACE_ROOT/<run_name>/`

```
$WORKSPACE_ROOT/<run_name>/
  checkpoints/          # model checkpoints
  logs/                 # training stdout/stderr
  configs/              # CLI invocations per segment
  graphs/               # plots
  history.json          # structured log of decisions
  report.md             # human-readable summary
```

## Key Constraints

- The agent ONLY controls behavior via CLI arguments — do NOT modify training scripts or config files
- NEVER delete any directory other than individual step\_<N>/ checkpoint subdirectories
- The agent stops after reaching step 6000
- Checkpoint saves take ~12 minutes on this cluster's NFS storage — budget time accordingly
- Step time is ~8.5 seconds (BF16) or ~4.9 seconds (NVFP4) per step
