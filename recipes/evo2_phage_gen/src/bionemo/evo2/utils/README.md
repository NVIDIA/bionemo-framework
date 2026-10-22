# Evo2 utilities

This directory contains Evo2-specific configuration, checkpoint conversion, and
training helpers. Run the installed commands from the recipe environment; their
entry points are defined in the recipe's `pyproject.toml`.

## Choosing a utility

| Task                                                              | Entry point or module                                                                          |
| ----------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Convert a NeMo2, Savanna, or Vortex checkpoint to Megatron Bridge | [Checkpoint conversion guide](checkpoint/README.md)                                            |
| Export Megatron Bridge weights to Vortex                          | `evo2_export_mbridge_to_vortex`; see the [conversion guide](checkpoint/README.md)              |
| Inspect inverse Hyena filter priors                               | `evo2_analyze_inverse_prior`                                                                   |
| Remove optimizer state from a checkpoint                          | `evo2_remove_optimizer`                                                                        |
| Convert FASTA records to inference prompts                        | `bionemo_fasta_to_jsonl`                                                                       |
| Configure offline FASTA preprocessing                             | `Evo2PreprocessingConfig` in [config.py](config.py), consumed by `preprocess_evo2`             |
| Describe taxonomy for sequence prompts                            | `Evo2TaxonomyLineage` in [config.py](config.py)                                                |
| Compute targeted embedding variance regularization                | `SquaredErrorTargetedVarianceLoss` in [loss/embedding_variance.py](loss/embedding_variance.py) |

Use `--help` on an installed command for its complete argument list.

## Preparing inference prompts

```bash
bionemo_fasta_to_jsonl input.fasta prompts.jsonl --upper
```

Each FASTA record becomes a JSON object with `id` (the first word in the FASTA
header) and `prompt` (the concatenated sequence). `--upper` uppercases the
sequence; without it, the original case is preserved. The JSONL file can be
passed to `infer_evo2 --prompt-file`.

The implementation lives in `bionemo.common.io.fasta_to_jsonl`.
[utils/fasta_to_jsonl.py](fasta_to_jsonl.py) re-exports its Python functions for
backward compatibility.

## Removing optimizer state

```bash
evo2_remove_optimizer \
  --src-ckpt-dir /path/to/training_checkpoint \
  --dst-ckpt-dir /path/to/weights_checkpoint
```

The source is the checkpoint root containing `iter_NNNNNNN` directories.
The default output contains weights without optimizer state. To retain serialized
model objects such as Transformer Engine `_extra_state`, add
`--preserve-model-object-state`. See the [checkpoint guide](checkpoint/README.md)
for the directory layout.

The implementation is `bionemo.common.checkpoint.remove_optimizer`;
[checkpoint/evo2_remove_optimizer.py](checkpoint/evo2_remove_optimizer.py)
provides the Evo2 entry point.

## Legacy conversion scripts

[checkpoint/convert_zero3_to_zero1.py](checkpoint/convert_zero3_to_zero1.py) and
[checkpoint/convert_checkpoint_model_parallel_evo2.py](checkpoint/convert_checkpoint_model_parallel_evo2.py)
operate on older ZeRO/Savanna checkpoint layouts. They are not replacements for
the installed Megatron Bridge converters. Check each script's argument parser and
imports before using it with an older checkpoint environment.

[checkpoint/nemo2_to_hf.py](checkpoint/nemo2_to_hf.py) uses NeMo exporters for
Mamba and Llama; its `hyena` option raises an error. Use the documented Vortex
export path for Evo2 Hyena checkpoints.

## Developing utilities

Keep model-specific helpers here. Reusable recipe utilities live in
`bionemo.common`, and synthetic FASTA test fixtures live in
`bionemo.evo2.data.test_utils`. Preserve compatibility re-exports when moving
existing functions. The repository's `ci/scripts/check_copied_files.py` manages
copies of both packages in other recipes: edit the source under
`recipes/evo2_megatron`, then regenerate the copies with `--fix`.
