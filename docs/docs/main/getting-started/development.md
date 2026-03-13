# Development with BioNeMo

This page covers the current development model for the repository after the NeMo2-era training packages were removed.

## Code overview

BioNeMo Framework now has two main development workflows:

- **Recipe development** in `bionemo-recipes/`, where model implementations and end-to-end training or inference workflows live.
- **Framework library development** in `sub-packages/`, where reusable utilities and biology-oriented workflow libraries live.

Examples of current framework libraries include:

- `bionemo-core`: shared interfaces and data utilities
- `bionemo-recipeutils`: shared utilities used by multiple recipes
- `bionemo-scdl`: single-cell dataset loading and conversion
- `bionemo-moco`: molecular co-design utilities
- `bionemo-webdatamodule`: `WebDataset` helpers

## Package structure

Most framework packages follow the same structure:

- `pyproject.toml`: package metadata and dependencies
- `src/`: importable source code
- `tests/`: package tests
- `examples/` or `notebooks/`: optional tutorial material that is pulled into the docs build
- `README.md`: package overview and usage notes

## Training and fine-tuning

Training entrypoints for supported models live in `bionemo-recipes`.

See also [Training Models](./training-models.md).

Common locations:

- `bionemo-recipes/models/esm2`
- `bionemo-recipes/models/amplify`
- `bionemo-recipes/models/geneformer`
- `bionemo-recipes/recipes/evo2_megatron`
- `bionemo-recipes/recipes/esm2_native_te`
- `bionemo-recipes/recipes/esm2_accelerate_te`
- `bionemo-recipes/recipes/geneformer_native_te_mfsdp_fp8`

When a recipe or local workflow depends on a framework package, install that package into your active environment with an editable install:

```bash
uv pip install -e ./sub-packages/bionemo-core
uv pip install -e ./sub-packages/bionemo-scdl
uv pip install -e "./sub-packages/bionemo-recipeutils[basecamp]"
```

## Data and checkpoints

The `download_bionemo_data` CLI remains the standard way to fetch supported BioNeMo datasets and checkpoints:

```bash
download_bionemo_data --list-resources
```

Set `DATA_SOURCE=ngc` for public resources, or `DATA_SOURCE=pbss` for internal NVIDIA workflows where applicable.

## Advanced developer documentation

For repository-wide development guidance, see the top-level \[README\]({{ github_url }}) and the package or recipe README in the directory you are modifying.
