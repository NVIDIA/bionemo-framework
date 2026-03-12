# Eden Recipe

Eden is a family of genomic models that use the Llama 3.1 architecture, developed by Basecamp Research. Models range from 100M to 35B parameters.

Reference: [**Eden**](https://www.biorxiv.org/content/10.64898/2026.01.12.699009v2) by Basecamp Research.

## Installation

```bash
./.ci_build.sh  # build the virtualenv
source ./.ci_test_env.sh  # source the virtualenv
```

## CLI tools

| Command                         | Description                                          |
| ------------------------------- | ---------------------------------------------------- |
| `train_eden`                    | Train or fine-tune Eden models                       |
| `infer_eden`                    | Autoregressive text generation (greedy/sampling)     |
| `predict_eden`                  | Batch log-likelihood scoring on FASTA sequences      |
| `eden_convert_nemo2_to_mbridge` | Convert NeMo2 checkpoints to MBridge DCP format      |
| `eden_export_mbridge_to_hf`     | Export Eden MBridge checkpoint to HuggingFace Llama  |
| `eden_convert_hf_to_mbridge`    | Convert HuggingFace Llama checkpoint to Eden MBridge |

## Quick start

### Training with mock data

```bash
torchrun --nproc-per-node 1 --no-python \
  train_eden \
  --hf-tokenizer-model-path tokenizers/nucleotide_fast_tokenizer_256 \
  --model-size eden_7b --num-layers 2 --max-steps 5 --eval-interval 5 \
  --eval-iters 1 --mock-data \
  --micro-batch-size 4 --global-batch-size 4 --seq-length 64 \
  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1 \
  --mixed-precision-recipe bf16_mixed \
  --no-activation-checkpointing \
  --decay-steps 1000 --warmup-steps 10 \
  --log-interval 1 --seed 41 --dataset-seed 33 \
  --result-dir eden_test
```

Note: fp32_residual_connection is automatically set to False for Eden/TE layers.

### Autoregressive generation (infer_eden)

```bash
torchrun --nproc_per_node 1 --no-python \
  infer_eden \
  --ckpt-dir /path/to/mbridge/checkpoint \
  --prompt "ATCGATCGATCGATCG" \
  --max-new-tokens 200 \
  --temperature 1.0 \
  --output-file generated.txt
```

Options: --ckpt-dir, --prompt/--prompt-file, --max-new-tokens, --temperature, --top-k/--top-p, --tensor-parallel-size, --max-seq-length (auto-detected by default, override with EDEN_MAX_SEQ_LEN env var).

### Batch sequence scoring (predict_eden)

```bash
torchrun --nproc_per_node 1 --no-python \
  predict_eden \
  --fasta /path/to/sequences.fasta \
  --ckpt-dir /path/to/mbridge/checkpoint \
  --output-dir predictions/ \
  --micro-batch-size 4 \
  --write-interval epoch
```

## Exporting / importing Eden (Llama) checkpoints

### Export: MBridge to HuggingFace

```bash
eden_export_mbridge_to_hf \
  --mbridge-ckpt-dir /path/to/eden_mbridge/iter_0000001 \
  --hf-output-dir /path/to/eden_hf \
  --model-size eden_7b
```

Produces standard HuggingFace directory loadable with `LlamaForCausalLM.from_pretrained()`.

### Import: HuggingFace to MBridge

```bash
eden_convert_hf_to_mbridge \
  --hf-model-dir /path/to/eden_hf \
  --mbridge-ckpt-dir /path/to/eden_mbridge_reimported \
  --model-size eden_7b
```

## Model sizes

| Key        | Description             |
| ---------- | ----------------------- |
| `eden_7b`  | Eden base (~8B params)  |
| `eden_11b` | Eden ~11B               |
| `eden_18b` | Eden ~18B               |
| `eden_21b` | Eden ~21B               |
| `eden_24b` | Eden ~24B (32K context) |
| `eden_27b` | Eden ~27B (32K context) |
| `eden_28b` | Eden ~28B               |
| `eden_35b` | Eden ~35B               |

## Data

Eden uses the ShardedEdenDataset from Basecamp Research, backed by SQLite for fast windowed access to genomic sequences. The data utilities are provided by the `bionemo-recipeutils` sub-package.

## Docker build

```bash
docker build -t eden_megatron_recipe-$(git rev-parse --short HEAD) .
```
