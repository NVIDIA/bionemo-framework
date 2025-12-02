# Example Tiny Llama3 Checkpoint

This directory contains the model and tokenizer configuration for a tiny Llama3 model (~1M parameters) optimized for fast convergence testing on genomic sequences. This checkpoint is designed for quick sanity checks and convergence tests.

## Contents

- **config.json**: Model configuration for a tiny Llama3 model (4 layers, 384 hidden size)
- **tokenizer.json**: Fast tokenizer for nucleotide sequences (256 vocab size)
- **tokenizer_config.json**: Tokenizer configuration
- **special_tokens_map.json**: Special tokens mapping (EOS=0, PAD=1, BOS=2, UNK=3)

## Usage

Use this directory as the `model_tag` in your training configurations:

```yaml
# In your hydra config (e.g., L0_convergence configs)
model_tag: ./example_tiny_llama_checkpoint

dataset:
  tokenizer_path: ./example_tiny_llama_checkpoint  # Same directory for tokenizer
```

This eliminates the need for absolute paths and makes configurations portable across different environments.

## Pretraining Data

This recipe demonstrates pre-training llama3 on the opengenome2 dataset, available on Hugging Face at
[arcinstitute/opengenome2](https://huggingface.co/datasets/arcinstitute/opengenome2).

```python
>>> from datasets import load_dataset
>>> dataset = load_dataset("arcinstitute/opengenome2", data_dir="json/pretraining_or_both_phases", split="train", streaming=True)
Resolving data files: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 194/194 [00:00<00:00, 81158.49it/s]
>>> print({key: (type(val), len(val)) for key, val in next(iter(dataset)).items()})
{'text': (<class 'str'>, 827528)}
```

To download the dataset locally, use the following command:

```bash
export HF_TOKEN=<your_huggingface_token>
hf download arcinstitute/opengenome2 --repo-type dataset --include "json/pretraining_or_both_phases/**/*.jsonl.gz" --local-dir /path/to/download/directory
```

Then pass the downloaded dataset directory to the training script as the `dataset.load_dataset_kwargs.path` configuration parameter.

```bash
HF_DATASETS_OFFLINE=1 python train_fsdp2.py --config-name L0_sanity \
  dataset.load_dataset_kwargs.path=/path/to/download/directory
```

## Model Parameters

- Layers: 4
- Hidden size: 384
- Attention heads: 6
- Intermediate size: 1536
- Vocabulary size: 256 (nucleotide tokenizer)
- Max position embeddings: 8192

Perfect for fast convergence testing where you want to verify the model can overfit on small datasets.
