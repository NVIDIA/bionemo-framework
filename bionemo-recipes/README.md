# BioNemo Recipes

BioNemo Recipes provides an easy path for the biological foundation model training community to scale up transformer-based models efficiently. Rather than offering a batteries-included training framework, we provide **model checkpoints** with TransformerEngine layers and **training recipes** that demonstrate how to achieve maximum throughput with popular open-source frameworks.

## Overview

The biological AI community is actively prototyping model architectures and needs tooling that prioritizes extensibility, interoperability, and ease-of-use alongside performance. BioNemo Recipes addresses this by offering:

- **Flexible scaling**: Scale from single-GPU prototyping to multi-node training without complex parallelism configurations
- **Framework compatibility**: Works with popular frameworks like HuggingFace Accelerate, PyTorch Lightning, and vanilla PyTorch
- **Performance optimization**: Leverages TransformerEngine and megatron-fsdp for state-of-the-art training efficiency
- **Research-friendly**: Hackable, readable code that researchers can easily adapt for their experiments

### Use Cases

- **Foundation Model Developers**: AI researchers and ML engineers developing novel biological foundation models who need to scale up prototypes efficiently
- **Foundation Model Customizers**: Domain scientists looking to fine-tune existing models with proprietary data for drug discovery and biological research

## Repository Structure

This repository contains two types of components:

### Models (`models/`)

Huggingface-compatible `PreTrainedModel` classes that use TransformerEngine layers internally. These are designed to be:

- **Distributed via Hugging Face Hub**: Pre-converted checkpoints available at [huggingface.co/nvidia](https://huggingface.co/nvidia)
- **Drop-in replacements**: Compatible with `AutoModel.from_pretrained()` without additional dependencies
- **Performance optimized**: Leverage TransformerEngine features like FP8 training and context parallelism

Example models include ESM-2, Geneformer, and AMPLIFY.

### Recipes (`recipes/`)

Self-contained training examples demonstrating best practices for scaling biological foundation models. Each recipe is a complete Docker container with:

- **Framework examples**: Vanilla PyTorch, HuggingFace Accelerate, PyTorch Lightning
- **Feature demonstrations**: FP8 training, megatron-fsdp, context parallelism, sequence packing
- **Scaling strategies**: Single-GPU to multi-node training patterns
- **Benchmarked performance**: Validated throughput and convergence metrics

Recipes are **not pip-installable packages** but serve as reference implementations that users can adapt for their own research.

## Quick Start

### Using Models

```python
from transformers import AutoModel, AutoTokenizer

# Load a BioNemo model directly from Hugging Face
model = AutoModel.from_pretrained("nvidia/AMPLIFY_120M")
tokenizer = AutoTokenizer.from_pretrained("nvidia/AMPLIFY_120M")
```

### Running Recipes

```bash
# Navigate to a recipe
cd recipes/esm2_native_te_mfsdp

# Build and run
docker build -t esm2_recipe .
docker run --rm -it --gpus all esm2_recipe python train.py
```

______________________________________________________________________

## Developer Guide

### Setting Up Development Environment

1. **Install pre-commit hooks:**

   ```bash
   pre-commit install
   ```

   Run hooks manually:

   ```bash
   pre-commit run --all-files
   ```

2. **Test your changes:**
   Each model and recipe has its own build and test setup following this pattern:

   ```bash
   cd models/my_model  # or recipes/my_recipe
   docker build . -t my_tag
   docker run --rm -it --gpus all my_tag pytest -v .
   ```

### Coding Guidelines

We prioritize **readability and simplicity** over comprehensive feature coverage:

- **KISS over DRY**: It's better to have clear, duplicated code than complex abstractions
- **One thing well**: Each recipe should demonstrate specific features clearly rather than trying to cover everything
- **Self-contained**: Recipes cannot depend on cutting-edge code from other parts of the repository

### Testing Strategy

We use a three-tier testing approach:

#### L0 Tests (Pre-merge)

- **Purpose**: Fast validation that code works
- **Runtime**: \<10 minutes, single GPU
- **Frequency**: Run automatically on PRs
- **Scope**: Basic functionality, checkpoint creation/loading

#### L1 Tests (Performance Monitoring)

- **Purpose**: Performance benchmarking and partial convergence validation
- **Runtime**: Up to 4 hours, up to 16 GPUs
- **Frequency**: Nightly/weekly
- **Scope**: Throughput metrics, scaling validation

#### L2 Tests (Release Validation)

- **Purpose**: Full convergence and large-scale validation
- **Runtime**: Multiple days, hundreds of GPUs
- **Frequency**: Monthly or before releases
- **Scope**: Complete model convergence, cross-platform validation

### Adding New Components

#### Adding a New Model

Models should be pip-installable packages that can export checkpoints to Hugging Face. See the
[models README](models/README.md) for detailed guidelines on:

- Package structure and conventions
- Checkpoint export procedures
- Testing requirements
- CI/CD integration

#### Adding a New Recipe

Recipes should be self-contained Docker environments demonstrating specific training patterns. See
the [recipes README](recipes/README.md) for guidance on:

- Directory structure and naming
- Hydra configuration management
- Docker best practices
- SLURM integration examples

### CI/CD Contract

All components must pass this basic validation:

```bash
docker build -t {component_tag} .
docker run --rm -it --gpus all {component_tag} pytest -v .
```

#### Running CI/CD

To run the CI/CD pipeline locally, run the following command:

```bash
./ci/build_and_test.py
```

### Performance Expectations

We aim to provide the fastest available training implementations for biological foundation models, with documented benchmarks across NVIDIA hardware (A100, H100, H200, B100, B200, etc.).

## Contributing

We welcome contributions that advance the state of biological foundation model training. Please ensure your contributions:

1. Follow our coding guidelines emphasizing clarity
2. Include appropriate tests (L0 minimum, L1/L2 as applicable)
3. Provide clear documentation and examples
4. Maintain compatibility with our supported frameworks

For detailed contribution guidelines, see our individual component READMEs:

- [Models Development Guide](models/README.md)
- [Recipes Development Guide](recipes/README.md)

## License

[Add appropriate license information]

## Support

For technical support and questions:

- Check existing issues before opening a new one
- Review our training recipes for implementation examples
- Consult the TransformerEngine and megatron-fsdp documentation for underlying technologies
