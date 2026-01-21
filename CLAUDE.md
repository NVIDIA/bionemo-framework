# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

BioNeMo Framework is a comprehensive suite for building and training biological foundation models at scale. The repository is split into three main areas:

1. **bionemo-recipes/** - Lightweight, self-contained training recipes using FSDP (Fully Sharded Data Parallel) with TransformerEngine and megatron-FSDP
2. **sub-packages/** - 5D parallel models (tensor/pipeline/context parallel) using NeMo/Megatron-Core, and dataloading/processing tools
3. **3rdparty/** - Git submodules for NeMo and Megatron-LM dependencies

## Code Architecture

### bionemo-recipes Structure

The recipes directory contains two types of components:

**Models (`bionemo-recipes/models/`)**: HuggingFace-compatible `PreTrainedModel` classes with TransformerEngine layers. These are:

- Distributed via Hugging Face Hub (e.g., nvidia/esm2_t48_15B_UR50D)
- Drop-in replacements for standard transformers, compatible with `AutoModel.from_pretrained()`
- Each model includes: conversion utilities (HF ↔ TE), golden value tests, checkpoint export scripts, and open-source license

**Recipes (`bionemo-recipes/recipes/`)**: Self-contained Docker environments demonstrating training patterns. Each recipe:

- Is completely isolated with no shared dependencies between recipes
- Contains everything needed: Dockerfile, training scripts, Hydra configs, tests, sample data
- Prioritizes KISS (Keep It Simple) over DRY - code duplication is preferred for clarity
- Follows naming: `{model_name}_{framework}_{features}/` (e.g., `esm2_native_te/`)

## Essential Commands

### Pre-commit and Linting

**CRITICAL**: Always run pre-commit hooks after making changes:

```bash
# After editing files
pre-commit run --all-files
# Or for modified files only
pre-commit run
```

Pre-commit includes:

- Ruff linting/formatting (line-length: 119, Google-style docstrings)
- Markdown formatting (mdformat)
- License header checks
- Trailing whitespace/EOF fixes
- YAML validation
- Secret detection

If pre-commit fails, fix issues before considering task complete. Tasks are NOT complete until all linter errors are resolved and pre-commit passes.

### Testing

**For bionemo-recipes:**

```bash
# Test a specific recipe/model inside the devcontainer
cd bionemo-recipes/recipes/{recipe_name}  # or models/{model_name}
pytest -v .
```

## Development Workflow

### Working on bionemo-recipes

1. **Navigate to the recipe/model**: `cd bionemo-recipes/recipes/{name}` or `cd bionemo-recipes/models/{name}`
2. **Make changes** - remember each recipe is self-contained
3. **Test locally inside the devcontainer**:
   ```bash
   pytest -v .
   ```
4. **Run pre-commit**: `pre-commit run --files $(git ls-files -m)`

### Code Quality Standards

- **Line length**: 119 characters
- **Docstrings**: Google-style (pydocstyle convention)
- **Import sorting**: isort configuration (2 lines after imports)
- **Linting**: Ruff for Python, Pyright for type checking
- **Test files and `__init__.py`**: Have relaxed linting rules as configured in pyproject.toml

## Key Configuration Files

- **.pre-commit-config.yaml**: Pre-commit hook configuration
- **.cursorrules**: Cursor AI coding guidelines
- **pyproject.toml (per sub-package)**: Individual package configuration

## Important Patterns

### bionemo-recipes Philosophy

- **Self-contained**: No cross-recipe dependencies, no imports from other recipes
- **Educational**: Code is documentation - prioritize readability over abstraction
- **KISS over DRY**: Duplicate code if it improves clarity
- **One concept per recipe**: Don't try to demonstrate every feature in one script

### Model Conversion Pattern (bionemo-recipes/models)

Models require:

1. Golden value tests proving TE model matches reference model
2. Bidirectional conversion functions: `convert_hf_to_te()` and `convert_te_to_hf()`
3. Export script (`export.py`) bundling all files for Hugging Face Hub
4. Open-source license

### Pre-commit Failures

- Fix all issues before committing
- Ruff will auto-fix many issues - verify fixes are appropriate
- Some files have per-file ignores in pyproject.toml - respect those
