# PEFT Fine-tuning with TransformerEngine-accelerated ESM-2

This folder demonstrates how to fine-tune a TransformerEngine-accelerated ESM-2 model using PEFT.

## Commands to Launch LoRA Fine-tuning

To run single-process training on one GPU, run:

```bash
python train_lora_ddp.py
```

To run multi-process training locally on 2+ GPUs, run (e.g. 2 GPUs):

```bash
torchrun --nproc_per_node=2 train_lora_ddp.py
```

## Sequence Packing (THD input format)

Sequence packing is handled via the [`DataCollatorWithFlattening`](https://huggingface.co/docs/transformers/v4.47.1/main_classes/data_collator#transformers.DataCollatorWithFlattening) collator from the HuggingFace transformers library that provides input arguments (e.g.
`cu_seq_lens_q`) needed for padding-free attention. To enable sequence packing, set `use_sequence_packing=true`
in the hydra configuration.

```bash
python train_lora_ddp.py --config-name L0_sanity use_sequence_packing=true
```

## Running Inference

To do inference, use `infer.py`. By default, it uses the `hydra_config/L0_sanity_infer.yaml` config, which points to a
previously LoRA fine-tuned checkpoint fine-tuned on `data/porter6_train_dataset_55k.parquet`. It also defaults to running
inference on the FASTA file at `data/input_infer.fasta` (see `hydra_config/defaults_infer.yaml`).

To run inference with the defaults:

```bash
python infer.py
```

You can override the most common settings from the command line:

- **`input_file`**: FASTA input (default: `data/input_infer.fasta`)
- **`output_file`**: Where to write predictions (CSV). If `null`, results print to stdout (default: `preds.csv`)
- **`model_tag`**: Base ESM-2 HF model to load (default: `nvidia/esm2_t6_8M_UR50D`)
- **`base_model_config_dir`**: Directory containing the fine-tuned model config
- **`peft_model_config_dir`**: Directory containing the LoRA adapter weights/config (defaults to `base_model_config_dir`)

Examples:

```bash
# Run on a different FASTA file and write a CSV
python infer.py input_file=/path/to/inputs.fasta output_file=preds.csv

# Point to your own LoRA fine-tuned checkpoint directory
python infer.py base_model_config_dir=/path/to/my_peft_checkpoint peft_model_config_dir=/path/to/my_peft_checkpoint
```

## Datasets

This recipe includes small and medium-sized datasets in `data/` so you can get started quickly without downloading
anything.

- **Quick sanity dataset (used for CI and smoke tests)**: `data/peft_sanity_dataset.parquet` is a **5,000-sample subset**
  of the Hugging Face dataset
  [`lamm-mit/protein_secondary_structure_from_PDB`](https://huggingface.co/datasets/lamm-mit/protein_secondary_structure_from_PDB).
  It is intended for fast local iteration and is also used by the recipe's CI tests.

- **Porter6 paper datasets**:

  - `data/porter6_train_dataset_55k.parquet`: training set.
  - `data/porter6_val_dataset_2024_692.parquet`: 2024 benchmark validation set.

  These originate from the Porter6 secondary-structure prediction work. For details on how they were built, see the
  [Porter6 paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11719765/). The original dataset files were downloaded from
  the [Porter6 repository](https://github.com/WafaAlanazi/Porter6) and then converted to Parquet format for use in this
  recipe.

### Installing Dependencies

The easiest way to get started with this recipe is to use the provided Dockerfile, which uses the latest NVIDIA PyTorch
base image to provide optimized versions of PyTorch and TransformerEngine. To build the container, run:

```bash
docker build -f Dockerfile -t esm2_peft_te ..
```

This uses `..` as the build context so the `esm2_native_te` symlink targets are available during the build.
The Dockerfile only copies the four shared Python files from `esm2_native_te` into the image.

To run the container, run:

```bash
docker run -it --gpus all --network host --ipc=host --rm -v ${PWD}:/workspace/bionemo esm2_peft_te /bin/bash
```

## Developer Guide

### Running tests

To run tests locally, run `recipes_local_test.py` from the repository root with the recipe directory as an argument.

```bash
./ci/scripts/recipes_local_test.py bionemo-recipes/recipes/esm2_peft_te/
```

For more information see [here](../esm2_native_te/README.md).
