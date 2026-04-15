# Running Local Performance Benchmarks

Instructions for running throughput/MFU benchmarks on a bare-metal GPU machine (e.g., 8x B200).

## Prerequisites

1. Clone the repo and check out the benchmark branch:

   ```bash
   git clone https://github.com/NVIDIA/bionemo-framework.git
   cd bionemo-framework
   git checkout gkaushik/config-generation
   ```

2. Install recipe dependencies (repeat for each recipe you want to benchmark):

   ```bash
   cd bionemo-recipes/recipes/esm2_native_te
   pip install -r requirements.txt
   cd ../../..
   ```

## Step 1: Generate local benchmark configs

From the repo root, generate Hydra configs from the Blackwell matrix:

```bash
python ci/benchmarks/generate_benchmark_configs.py \
    --mode local \
    --csv ci/benchmarks/blackwell_matrix.csv
```

This creates benchmark Hydra configs in each recipe's `hydra_config/` directory.
The configs follow the naming pattern `bench_{hardware}_{variant}_{precision}[_cp{N}].yaml`.

To regenerate after changes: re-run the same command (idempotent).

## Step 2: Run benchmarks

Each benchmark is a `torchrun` command. The key rule:

- **CP=1** (no context parallelism): use `train_fsdp2.py`
- **CP>1** (context parallelism): use `train_fsdp2_cp.py`

### ESM2 benchmarks

```bash
cd bionemo-recipes/recipes/esm2_native_te

# ESM2 3B — BF16
torchrun --nproc_per_node=8 train_fsdp2.py --config-name bench_b200_3b_bf16
torchrun --nproc_per_node=8 train_fsdp2_cp.py --config-name bench_b200_3b_bf16_cp4
torchrun --nproc_per_node=8 train_fsdp2_cp.py --config-name bench_b200_3b_bf16_cp8

# ESM2 3B — FP8
torchrun --nproc_per_node=8 train_fsdp2.py --config-name bench_b200_3b_fp8
torchrun --nproc_per_node=8 train_fsdp2_cp.py --config-name bench_b200_3b_fp8_cp4
torchrun --nproc_per_node=8 train_fsdp2_cp.py --config-name bench_b200_3b_fp8_cp8

# ESM2 3B — MXFP8
torchrun --nproc_per_node=8 train_fsdp2.py --config-name bench_b200_3b_mxfp8
torchrun --nproc_per_node=8 train_fsdp2_cp.py --config-name bench_b200_3b_mxfp8_cp4
torchrun --nproc_per_node=8 train_fsdp2_cp.py --config-name bench_b200_3b_mxfp8_cp8

# ESM2 3B — NVFP4
torchrun --nproc_per_node=8 train_fsdp2.py --config-name bench_b200_3b_nvfp4
torchrun --nproc_per_node=8 train_fsdp2_cp.py --config-name bench_b200_3b_nvfp4_cp4
torchrun --nproc_per_node=8 train_fsdp2_cp.py --config-name bench_b200_3b_nvfp4_cp8

# ESM2 15B — BF16
torchrun --nproc_per_node=8 train_fsdp2.py --config-name bench_b200_15b_bf16
torchrun --nproc_per_node=8 train_fsdp2_cp.py --config-name bench_b200_15b_bf16_cp4
torchrun --nproc_per_node=8 train_fsdp2_cp.py --config-name bench_b200_15b_bf16_cp8

# ESM2 15B — FP8
torchrun --nproc_per_node=8 train_fsdp2.py --config-name bench_b200_15b_fp8
torchrun --nproc_per_node=8 train_fsdp2_cp.py --config-name bench_b200_15b_fp8_cp4
torchrun --nproc_per_node=8 train_fsdp2_cp.py --config-name bench_b200_15b_fp8_cp8

# ESM2 15B — MXFP8
torchrun --nproc_per_node=8 train_fsdp2.py --config-name bench_b200_15b_mxfp8
torchrun --nproc_per_node=8 train_fsdp2_cp.py --config-name bench_b200_15b_mxfp8_cp4
torchrun --nproc_per_node=8 train_fsdp2_cp.py --config-name bench_b200_15b_mxfp8_cp8

# ESM2 15B — NVFP4
torchrun --nproc_per_node=8 train_fsdp2.py --config-name bench_b200_15b_nvfp4
torchrun --nproc_per_node=8 train_fsdp2_cp.py --config-name bench_b200_15b_nvfp4_cp4
torchrun --nproc_per_node=8 train_fsdp2_cp.py --config-name bench_b200_15b_nvfp4_cp8
```

### Llama3 benchmarks

```bash
cd bionemo-recipes/recipes/llama3_native_te

# Llama-3.1-8B — BF16
torchrun --nproc_per_node=8 train_fsdp2.py --config-name bench_b200_8b_bf16
torchrun --nproc_per_node=8 train_fsdp2_cp.py --config-name bench_b200_8b_bf16_cp4
torchrun --nproc_per_node=8 train_fsdp2_cp.py --config-name bench_b200_8b_bf16_cp8

# Llama-3.1-8B — FP8
torchrun --nproc_per_node=8 train_fsdp2.py --config-name bench_b200_8b_fp8
torchrun --nproc_per_node=8 train_fsdp2_cp.py --config-name bench_b200_8b_fp8_cp4
torchrun --nproc_per_node=8 train_fsdp2_cp.py --config-name bench_b200_8b_fp8_cp8

# Llama-3.1-8B — MXFP8
torchrun --nproc_per_node=8 train_fsdp2.py --config-name bench_b200_8b_mxfp8
torchrun --nproc_per_node=8 train_fsdp2_cp.py --config-name bench_b200_8b_mxfp8_cp4
torchrun --nproc_per_node=8 train_fsdp2_cp.py --config-name bench_b200_8b_mxfp8_cp8

# Llama-3.1-8B — NVFP4 (depends on PR #1500)
torchrun --nproc_per_node=8 train_fsdp2.py --config-name bench_b200_8b_nvfp4
torchrun --nproc_per_node=8 train_fsdp2_cp.py --config-name bench_b200_8b_nvfp4_cp4
torchrun --nproc_per_node=8 train_fsdp2_cp.py --config-name bench_b200_8b_nvfp4_cp8
```

### OpenGenome2 benchmarks

```bash
cd bionemo-recipes/recipes/opengenome2_llama_native_te

# OpenGenome2-7B — BF16
torchrun --nproc_per_node=8 train_fsdp2.py --config-name bench_b200_og2-7b_bf16
torchrun --nproc_per_node=8 train_fsdp2_cp.py --config-name bench_b200_og2-7b_bf16_cp4
torchrun --nproc_per_node=8 train_fsdp2_cp.py --config-name bench_b200_og2-7b_bf16_cp8

# OpenGenome2-7B — FP8
torchrun --nproc_per_node=8 train_fsdp2.py --config-name bench_b200_og2-7b_fp8
torchrun --nproc_per_node=8 train_fsdp2_cp.py --config-name bench_b200_og2-7b_fp8_cp4
torchrun --nproc_per_node=8 train_fsdp2_cp.py --config-name bench_b200_og2-7b_fp8_cp8

# OpenGenome2-7B — MXFP8
torchrun --nproc_per_node=8 train_fsdp2.py --config-name bench_b200_og2-7b_mxfp8
torchrun --nproc_per_node=8 train_fsdp2_cp.py --config-name bench_b200_og2-7b_mxfp8_cp4
torchrun --nproc_per_node=8 train_fsdp2_cp.py --config-name bench_b200_og2-7b_mxfp8_cp8

# OpenGenome2-7B — NVFP4 (depends on PR #1500)
torchrun --nproc_per_node=8 train_fsdp2.py --config-name bench_b200_og2-7b_nvfp4
torchrun --nproc_per_node=8 train_fsdp2_cp.py --config-name bench_b200_og2-7b_nvfp4_cp4
torchrun --nproc_per_node=8 train_fsdp2_cp.py --config-name bench_b200_og2-7b_nvfp4_cp8
```

### CodonFM benchmarks

CodonFM only supports CP=1 (no `train_fsdp2_cp.py` yet). CP>1 configs are generated but disabled.

```bash
cd bionemo-recipes/recipes/codonfm_native_te

# CodonFM 5B — BF16 (CP=1 only)
torchrun --nproc_per_node=8 train_fsdp2.py --config-name bench_b200_5b_bf16

# CodonFM 5B — FP8 (CP=1 only)
torchrun --nproc_per_node=8 train_fsdp2.py --config-name bench_b200_5b_fp8

# CodonFM 5B — MXFP8 (CP=1 only)
torchrun --nproc_per_node=8 train_fsdp2.py --config-name bench_b200_5b_mxfp8

# CodonFM 5B — NVFP4 (CP=1 only)
torchrun --nproc_per_node=8 train_fsdp2.py --config-name bench_b200_5b_nvfp4
```

### Evo2 benchmarks

Evo2 uses Megatron (not Hydra recipes). Local config generation is not supported.
Use the Evo2 CLI directly or the Lepton CI configs.

## Metrics collected

Each benchmark logs these metrics to W&B (and stdout):

| Metric            | W&B key                              | Description                           |
| ----------------- | ------------------------------------ | ------------------------------------- |
| Step time         | `train/step_time`                    | Wall-clock seconds per optimizer step |
| Tokens/sec        | `train/tokens_per_second_per_gpu`    | Throughput per GPU                    |
| MFU               | `train/mfu_percent`                  | Model FLOPs Utilization (%)           |
| TFLOPS/GPU        | `train/tflops_per_gpu`               | Achieved TFLOPS per GPU               |
| GPU memory (peak) | `train/gpu_memory_allocated_max_gb`  | Peak GPU memory (GB)                  |
| GPU memory (mean) | `train/gpu_memory_allocated_mean_gb` | Average GPU memory (GB)               |
| Loss              | `train/loss`                         | Training loss                         |

Set `log_mfu: true` in the config (default for benchmarks) to enable MFU tracking.

## Tips

- Run benchmarks one at a time to get clean measurements.
- The first few steps are warmup — ignore them when analyzing throughput.
- Configs default to 500 steps for ESM2/CodonFM and 250 for Llama3/OG2.
- Override at runtime: `torchrun ... --config-name bench_b200_3b_bf16 num_train_steps=100`
- W&B defaults to online mode. To disable: append `wandb_init_args.mode=offline` (ESM2/CodonFM) or `wandb.mode=offline` (Llama3/OG2).
