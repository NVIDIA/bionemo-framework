# bionemo-recipeutils

Shared, framework-agnostic utilities for BioNeMo recipes.

## Constraint

**This package must not depend on megatron-core, megatron-bridge, or NeMo.**
Recipe-specific framework adapters (e.g., `DatasetProvider` wrappers) belong
in each recipe, not here.

## Contents

- `bionemo.recipeutils.data.basecamp` — High-performance SQLite-backed genomic
  dataset (`ShardedEdenDataset`) and window pre-computation CLI, contributed by
  [BaseCamp Research](https://basecamp-research.com/).
- `bionemo.recipeutils.io` — File format conversion utilities (FASTA to JSONL).

## Installation

```bash
pip install bionemo-recipeutils            # core
pip install bionemo-recipeutils[basecamp]   # + polars for window pre-computation
```

## CLI tools

```bash
bionemo_fasta_to_jsonl input.fasta output.jsonl
bionemo_precompute_windows precompute split.parquet output.sqlite --window-size 8192
```
