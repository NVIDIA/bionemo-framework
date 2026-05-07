# Claude Code — ComputeLab Benchmarking Agent

## Purpose

You are running on the ComputeLab login node to poll for GPU availability and run memory profiler benchmarks for the BioNeMo Lingua 7B/70B MXFP8 experiments.

## Directory Restrictions

- ONLY operate within these directories:
  - `/home/scratch.savithas_other/` — code, results, checkpoints, memory snapshots
  - `/home/scratch.savithas_other_1/` — TransformerEngine, data, container images
- NEVER read, write, or execute anything outside these two directories
- NEVER access other users' directories or `/home/scratch.*` directories belonging to others

## Forbidden Actions

- NEVER run `rm -rf` on any directory. If you need to delete something, list exactly what will be deleted and ask for confirmation first.
- NEVER run `git push`, `git push --force`, or any push commands
- NEVER amend or rewrite git history
- NEVER print, log, or echo environment variables containing secrets (WANDB_API_KEY, HUGGING_FACE_HUB_TOKEN, HF_TOKEN, ANTHROPIC_API_KEY)
- NEVER submit unbounded job chains (always set a max step count)
- NEVER modify system-wide configs, install system packages, or use sudo
- NEVER leave Docker containers running — always use `docker run --rm`

## What You CAN Do

- Poll `sinfo` for node availability
- Run `srun` to allocate nodes
- Run `docker run --rm` to launch benchmarks
- Read/edit files within the allowed directories
- Run `nvidia-smi`, `ls`, `cat`, `grep` for inspection
- Run `git status`, `git log`, `git diff`, `git pull` (read-only git operations)

## Key Paths on ComputeLab

Two scratch directories are used:

```
/home/scratch.savithas_other/           # Code + outputs
├── bionemo-framework/                  # Code repo (launch Claude from here)
├── checkpoints/                        # Training checkpoints
├── claude_tasks/                       # Claude task outputs
├── results/                            # Training results/logs (mounted into container)
└── memory_snapshots/                   # Torch memory profiler snapshots (mounted into container)
    └── results/                        # Captured wandb logs per run

/home/scratch.savithas_other_1/         # Large assets
├── TransformerEngine/                  # TE repo (built for B300 arch 103a)
├── data/                               # Training data (DCLM parquet)
└── enroot/                             # Container images (.sqsh)
```

### Directory Setup (run once before benchmarks)

These directories must exist on the HOST before Docker mounts them, otherwise Docker creates them as root-owned and you get permission errors:

```bash
mkdir -p /home/scratch.savithas_other/results
mkdir -p /home/scratch.savithas_other/memory_snapshots/results
```

The benchmark script creates per-run subdirs automatically, but verify these base directories exist first.

## Benchmark Configs

6 total benchmarks: 3 for 8B, 3 for 70B. All with memory profiler enabled, 100 steps.

### 8B Benchmarks (mbs=4, uses `train_fsdp2.py`)

| Benchmark      | Config                     | Extra Args                                                                      |
| -------------- | -------------------------- | ------------------------------------------------------------------------------- |
| 8B BF16        | `L2_lingua_7b`             | (none)                                                                          |
| 8B MXFP8       | `L2_lingua_7b_mxfp8`       | `fp8_layers=null`                                                               |
| 8B MXFP8 qinit | `L2_lingua_7b_mxfp8_qinit` | `fp8_config.quantized_model_init_kwargs.preserve_high_precision_init_val=false` |

### 70B Benchmarks (mbs=1, CP=4, uses `train_fsdp2_cp.py`)

The 70B model uses Context Parallelism (CP=4) on a single 8-GPU node. Use the `L2_lingua_70b_computelab` base config which sets CP=4, absolute model path, and appropriate defaults for single-node ComputeLab runs.

| Benchmark       | Config                     | Extra Args                                                                                                                                                                                                                                                      |
| --------------- | -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 70B BF16        | `L2_lingua_70b_computelab` | (none)                                                                                                                                                                                                                                                          |
| 70B MXFP8       | `L2_lingua_70b_computelab` | `fp8_config.enabled=true fp8_config.fp8_recipe=transformer_engine.common.recipe.MXFP8BlockScaling fp8_config.fp8_format=E4M3 fp8_layers=null`                                                                                                                   |
| 70B MXFP8 qinit | `L2_lingua_70b_computelab` | `fp8_config.enabled=true fp8_config.fp8_recipe=transformer_engine.common.recipe.MXFP8BlockScaling fp8_config.fp8_format=E4M3 fp8_config.quantized_model_init_kwargs.enabled=true fp8_config.quantized_model_init_kwargs.preserve_high_precision_init_val=false` |

**Important:** 70B uses `train_fsdp2_cp.py` (NOT `train_fsdp2.py`) because it requires Context Parallelism.

## Docker Launch Template

Common docker args for all benchmarks:

```bash
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /home/scratch.savithas_other/bionemo-framework:/workspace/bionemo \
  -v /home/scratch.savithas_other_1/TransformerEngine:/workspace/transformer_engine \
  -v /home/scratch.savithas_other_1/data:/workspace/data \
  -v /home/scratch.savithas_other/memory_snapshots:/workspace/memory_snapshots \
  -e WANDB_API_KEY -e HUGGING_FACE_HUB_TOKEN -e HF_TOKEN \
  -e HYDRA_FULL_ERROR=1 \
  -e WANDB_CONSOLE_SUMMARY_MAX_ROWS=20 \
  nvcr.io/nvidia/pytorch:26.03-py3 \
  bash -c "<COMMAND>"
```

For 8B benchmarks, the inner command is:

```bash
export PYTHONPATH=/workspace/transformer_engine:${PYTHONPATH:-}
cd /workspace/bionemo/bionemo-recipes/recipes/llama3_native_te
torchrun --nproc_per_node=8 train_fsdp2.py --config-name <CONFIG> \
  dataset.micro_batch_size=4 dataset.use_stateful_dataloader=true \
  grad_acc_steps=1 num_train_steps=100 \
  checkpoint.save_every_n_steps=999999 checkpoint.resume_from_checkpoint=false \
  logger.frequency=10 \
  memory_profiler.enabled=true memory_profiler.snapshot_dir=/workspace/memory_snapshots/<RUN_NAME> \
  wandb.name=<RUN_NAME> wandb.id=<RUN_NAME> wandb.project=lingua-memory \
  hydra.run.dir=/tmp/hydra_outputs \
  <EXTRA_ARGS>
```

For 70B benchmarks, the inner command is:

```bash
export PYTHONPATH=/workspace/transformer_engine:${PYTHONPATH:-}
cd /workspace/bionemo/bionemo-recipes/recipes/llama3_native_te
torchrun --nproc_per_node=8 train_fsdp2_cp.py --config-name L2_lingua_70b_computelab \
  num_train_steps=100 \
  checkpoint.save_every_n_steps=999999 checkpoint.resume_from_checkpoint=false \
  logger.frequency=10 \
  memory_profiler.enabled=true memory_profiler.snapshot_dir=/workspace/memory_snapshots/<RUN_NAME> \
  wandb.name=<RUN_NAME> wandb.id=<RUN_NAME> wandb.project=lingua-memory \
  hydra.run.dir=/tmp/hydra_outputs \
  <EXTRA_ARGS>
```

## Important Environment Variables

Set these INSIDE the Docker container for proper wandb output capture:

```bash
export WANDB_CONSOLE_SUMMARY_MAX_ROWS=20
```

This ensures the full wandb run summary is visible in stdout so you can read metrics like `train/gpu_memory_allocated_max_gb`, `train/step_time`, `train/tokens_per_second_per_gpu`, etc.

Also use `hydra.run.dir=/tmp/hydra_outputs` to avoid Hydra permission issues on read-only mounts.

## Capturing Results

1. You can only run a single benchmark at a time (one Docker container).
2. After each run, capture the wandb summary output (step_time, tokens_per_second_per_gpu, gpu_memory_allocated_max_gb, etc.) and save it to a results file.
3. Create a results directory: `/home/scratch.savithas_other/memory_snapshots/results/`
4. After ALL 6 runs complete, create a `results.md` summary with:
   - Peak memory (GB) per config
   - Tokens per second per GPU per config
   - Step time per config
   - Memory breakdown from snapshot analysis (weights vs optimizer vs activations)
   - Comparison tables for 8B and 70B separately

## Polling Workflow

We need a **B300 node with 8 GPUs**. To find one:

```bash
# Check for idle B300 8-GPU nodes
sinfo -N --noheader -o "%N %P %G %t" | grep -i b300 | grep "gpu:8" | grep idle
```

If the partition name is not `b300`, first run `sinfo` to discover the correct partition name, then poll that.

1. Poll every 60 seconds: `sinfo -N --noheader -o "%N %P %G %t" | grep "gpu:8" | grep idle` (filter for B300)
2. When found, allocate with `srun --partition=<B300_PARTITION> --nodes=1 --ntasks=1 --gpus-per-node=8 --exclusive --time=04:00:00 --pty bash`
3. Verify 8 GPUs: `nvidia-smi | grep B300` (should show 8 GPUs)
4. Run all 6 benchmarks sequentially: 8B BF16, 8B MXFP8, 8B MXFP8+qinit, 70B BF16, 70B MXFP8, 70B MXFP8+qinit
5. After each run, save the wandb summary to `/home/scratch.savithas_other/memory_snapshots/results/<run_name>.txt`
6. After all benchmarks complete, analyze the memory snapshots (see below)
7. Create `results.md` with comparison tables
8. Memory snapshots are `.pickle` files also viewable at https://pytorch.org/memory_viz

## Analyzing Memory Snapshots

After benchmarks complete, analyze the `.pickle` snapshots to compare peak memory across BF16 / MXFP8 / MXFP8+qinit. Since torch is only available inside the container, run analysis via Docker:

```bash
docker run --rm \
  -v /home/scratch.savithas_other/memory_snapshots:/snapshots \
  nvcr.io/nvidia/pytorch:26.03-py3 \
  python /snapshots/analyze.py
```

### Analysis Script Template

Write a Python script to `/home/scratch.savithas_other/memory_snapshots/analyze.py` that:

1. Loads each snapshot:

   ```python
   import pickle

   with open("/snapshots/<run_name>/memory_snapshot.pickle", "rb") as f:
       snap = pickle.load(f)
   ```

2. Extracts key metrics from each snapshot:

   - `snap["device_configuration"]` — GPU info
   - `snap["segments"]` — memory segments (each has `total_size`, `blocks`)
   - `snap["allocation_records"]` — individual allocations with stack traces

3. Computes and compares:

   - Total allocated memory at snapshot time
   - Peak memory (sum of all active segments)
   - Largest individual allocations (sort by size, show top 10 with stack traces)
   - Breakdown by category: model weights, optimizer states, activations, gradients

4. Prints comparison tables for both model sizes:

   ```
   === 8B Model ===
   | Metric              | BF16    | MXFP8 no-qinit | MXFP8 qinit |
   |---------------------|---------|-----------------|-------------|
   | Peak allocated (GB) | ...     | ...             | ...         |
   | Model weights (GB)  | ...     | ...             | ...         |
   | Optimizer (GB)      | ...     | ...             | ...         |
   | Activations (GB)    | ...     | ...             | ...         |

   === 70B Model ===
   | Metric              | BF16    | MXFP8 no-qinit | MXFP8 qinit |
   |---------------------|---------|-----------------|-------------|
   | Peak allocated (GB) | ...     | ...             | ...         |
   | Model weights (GB)  | ...     | ...             | ...         |
   | Optimizer (GB)      | ...     | ...             | ...         |
   | Activations (GB)    | ...     | ...             | ...         |
   ```

The goal is to answer Jonathan's question: why does qinit not show memory savings on the 8B model? Compare where the memory goes in each configuration. The 70B comparison should show a clearer benefit since model weights are a larger fraction of total memory.

## Debugging and Error Handling

When a benchmark fails, you MUST diagnose and fix the issue before moving on. Do not skip failed benchmarks.

### General Debugging Approach

1. Read the full error output carefully — the root cause is usually near the bottom of the traceback.
2. If the error is in training code (Python), you can edit files in `/home/scratch.savithas_other/bionemo-framework/` since they're mounted into the container.
3. Test fixes with a quick 5-step run (`num_train_steps=5`) before re-running the full 100-step benchmark.
4. If you edit training code to fix an issue, note what you changed in the `results.md` so we can review later.

### Common Errors and Fixes

**CUDA Out of Memory:**

- Reduce `micro_batch_size` (e.g., 4 → 2 for 8B, 1 is already minimum for 70B)
- For 70B, increase `cp_size` (e.g., 4 → 8) if memory is tight
- Check if another process is using GPU memory: run `nvidia-smi` on the node

**Hydra config errors (MissingMandatoryValue, ConfigAttributeError):**

- Check that config overrides are valid: run a quick `python train_fsdp2.py --help --config-name <CONFIG>` inside the container
- If a config key doesn't exist, check `hydra_config/defaults.yaml` and the specific config file for the correct key name
- Use `HYDRA_FULL_ERROR=1` (already set) to see the full traceback

**Permission denied on mounted directories:**

- Check that the directory exists on the host and is owned by you: `ls -la /home/scratch.savithas_other/memory_snapshots/`
- If Docker created it as root, fix with: `docker run --rm -v /home/scratch.savithas_other/memory_snapshots:/fix alpine chown -R $(id -u):$(id -g) /fix`
- Or create directories on the host before running Docker

**TransformerEngine import errors:**

- Verify TE is mounted: run `ls /workspace/transformer_engine/` inside container
- Verify PYTHONPATH is set: `echo $PYTHONPATH` should include `/workspace/transformer_engine`
- If TE C extensions are missing: the `.so` files in `/home/scratch.savithas_other_1/TransformerEngine/` must be built for B300 (arch 103a). If they were built for a different GPU, rebuild:
  ```bash
  docker run --rm --gpus all \
    -v /home/scratch.savithas_other_1/TransformerEngine:/workspace/te \
    nvcr.io/nvidia/pytorch:26.03-py3 \
    bash -c "cd /workspace/te && NVTE_CUDA_ARCHS='103a' pip install -e . --no-build-isolation"
  ```

**Docker: image not found / pull errors:**

- The image `nvcr.io/nvidia/pytorch:26.03-py3` should already be cached. If not, pull it:
  ```bash
  docker pull nvcr.io/nvidia/pytorch:26.03-py3
  ```

**torchrun NCCL errors (timeout, connection refused):**

- Ensure `--ipc=host` is in Docker args
- Ensure all 8 GPUs are visible: `nvidia-smi -L` should show 8 GPUs
- Try setting `NCCL_DEBUG=INFO` to get more details:
  ```bash
  -e NCCL_DEBUG=INFO
  ```

**Dataloader errors (no data files found):**

- Check that DCLM data exists: `ls /home/scratch.savithas_other_1/data/`
- The 8B config expects data at `/workspace/data/dclm-baseline/global-shard_01_of_10/**/*.parquet`
- If the data path is different, override with `dataset.load_dataset_kwargs.data_files=<correct_path>`

**Memory profiler snapshot not saved:**

- Check that the snapshot dir exists inside the container and is writable
- The snapshot is only saved on rank 0 after the first training step
- Look for the log line: `Memory snapshot saved to ...`
- If missing, check that `memory_profiler.enabled=true` and `memory_profiler.snapshot_after_first_step=true` are set

### Retry Strategy

- If a benchmark fails, try to fix the root cause and re-run it.
- Do NOT retry the same command more than 2 times without changing something.
- If you cannot fix an error after 2 attempts, save the error output to `/home/scratch.savithas_other/memory_snapshots/results/<run_name>_ERROR.log` and move on to the next benchmark. Report the failure in `results.md`.

## GPU Architecture

- ComputeLab B300 nodes: CUDA arch `103a`
- TE is pre-built for this architecture in `/home/scratch.savithas_other_1/TransformerEngine/`
