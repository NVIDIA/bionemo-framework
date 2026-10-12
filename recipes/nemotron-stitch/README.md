# nemotron-stitch

**STITCH**: Soft-Token Integration, Training, Checkpointing, and Hosting.

Nemotron Stitch connects a frozen external encoder to a Nemotron language
model or any other base reasoning model. The encoder turns an
application-specific input into representation vectors, and a small trainable
**projector** maps those vectors into the
language model's embedding space. The projected vectors are **soft tokens**:
they replace reserved token positions in the prompt and are then processed like
ordinary language-model embeddings.

The package supplies the framework integration around that idea: projector
construction and scatter, portable artifacts, NeMo AutoModel training, NeMo RL
rollouts, and vLLM serving. Applications keep ownership of their encoder, data,
prompt format, and evaluation logic.

## Start here

- [`examples/llava/README.md`](examples/llava/README.md) is a complete,
  qualified run of projector alignment, LoRA SFT, GRPO, and vLLM inference
  using a frozen CLIP encoder and a text-only Nemotron model.
- [`examples/llava/quickstart.md`](examples/llava/quickstart.md) explains how to
  replace the image-specific pieces with your own encoder and paired data.

The example is intentionally small enough to read as one application module,
while exercising the same package paths used by larger training jobs.

## The model in one minute

```text
domain input
    |
    v
frozen encoder
    |
    |  features [T, F]
    v
trainable projector
    |
    |  soft tokens [T, H]
    v
reserved positions in the text prompt
    |
    v
Nemotron language model
```

`T` is the number of encoder positions, `F` is the width of each encoder
representation, and `H` is the language model's embedding width. A
token-preserving projector keeps `T`; a resampling projector may emit a
different number of soft tokens. The package's flat scatter contract also
supports different token counts across examples in one batch.

The encoder does not need to return NumPy arrays. Its native result may be a
PyTorch tensor, a byte payload, or a service response. The application data
path decodes that result into the tensor consumed by the projector. Nemotron
Stitch includes a NumPy-backed feature cache for applications that find it
useful, but that cache is not the encoder interface.

## What the package provides

### Projectors and prompt scatter

- Named projectors that map encoder features to the LM embedding width.
- Token-preserving MLP projectors and a fixed-output Perceiver projector for
  3D feature grids.
- Flat feature and token-index contracts for uniform or ragged batches.
- Validation that projected rows land on the correct placeholder tokens and
  have the correct LM width.

Applications can define projector kinds outside this package and register
them before model or artifact construction:

```python
import torch

from nemotron_stitch.projector import Projector, register_projector


class MyProjector(Projector):
    def __init__(self, mm_hidden_size: int, output_size: int, hidden_size: int):
        super().__init__()
        self.mm_hidden_size = mm_hidden_size
        self.output_size = output_size
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(mm_hidden_size, hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_size, output_size),
        )

    def reset_parameters(self) -> None:
        for layer in self.layers:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


register_projector("my_projector", MyProjector)
```

The corresponding config uses `kind: my_projector`; all other fields are
passed to the class as keyword arguments, along with the LM's `output_size`.
Registration is process-local, so applications must perform it before loading
the model or a projector artifact in every worker process that constructs the
projector. Built-in kinds cannot be replaced.

### Portable projector artifacts

Projector weights are stored separately from the base model and LoRA weights:

```text
mm-projector.safetensors
mm-projector-manifest.json
```

The manifest records projector configuration, tensor metadata, checksums, and
optional provenance. Artifact loading fails on incompatible structure or a
checksum mismatch. Applications can opt into stricter provenance checks when
they have stable encoder, dataset, or model identities worth enforcing.

### Feature utilities

The `features` package contains reusable encoder registration, chunking,
pooling, content-addressed caching, resumable cache filling, and bounded data
preparation. These utilities are optional. An application with an existing
cache, online encoder, or native binary format can provide its own data-loading
boundary without changing the projector or framework contracts.

### Framework integration

- **NeMo AutoModel**: out-of-tree model registration, multimodal forward-input
  handling, projector-aware training recipes, trainability policy, optimizer
  configuration, and distributed-checkpoint helpers.
- **NeMo RL**: feature transport, projector loading, policy-worker extensions,
  and rollout wiring for GRPO.
- **vLLM**: plugin construction for either pre-projected soft tokens or
  application-defined encoding inside the worker.
- **Testing**: reusable conformance fixtures for projector construction,
  scatter behavior, artifact round trips, and trainability rules.

Framework imports are lazy. The base package can be imported without NeMo
AutoModel, NeMo RL, or vLLM installed; applications supply versions of those
frameworks that match their runtime.

## What an application provides

Nemotron Stitch deliberately does not define a universal modality interface.
Each application owns:

- encoder construction, preprocessing, and output decoding;
- dataset access and row normalization;
- prompt text and the number of reserved soft-token positions per input;
- base-model registration and framework configuration;
- rewards, evaluation, and domain-specific qualification; and
- any cache or online-serving behavior that is specific to its encoder.

This keeps modality code out of the package and avoids forcing every encoder
through the same storage format or geometry.

## Training and serving

The usual workflow has three stages:

| Stage     | Trainable state                 | Purpose                                             |
| --------- | ------------------------------- | --------------------------------------------------- |
| Alignment | Projector                       | Teach the LM to interpret encoder representations.  |
| SFT       | Projector and LoRA              | Adapt the model to application prompts and answers. |
| GRPO      | LoRA; projector normally frozen | Optimize responses with an automatic reward.        |

Here **projector** always means the encoder-to-LM module. **Adapter** means
LoRA. Keeping those names distinct matters because both can be present in the
same checkpoint and have different trainability rules.

At inference time, the application runs or queries the frozen encoder, loads
the trained projector, and sends the projected soft tokens to vLLM together
with the prompt and selected LoRA weights. vLLM does not need to understand the
original domain input when projected mode is used.

## Core contracts

| Boundary             | Contract                                                                                      |
| -------------------- | --------------------------------------------------------------------------------------------- |
| Encoder to projector | Application-decoded PyTorch tensor; token-wise features are normally `[N, F]` or `[B, T, F]`. |
| Projector to LM      | Soft tokens `[N, H]` or `[B, T, H]`, where `H` is the LM embedding width.                     |
| Model forward kwargs | `mm_features__<name>` and optional `mm_token_indices__<name>`.                                |
| Prompt routing       | Each named projector has a distinct placeholder token ID.                                     |
| Flat indices         | Positions index the flattened `[B, S]` token grid and support ragged batches.                 |
| Ownership            | `sidecar` for an independently managed projector or `module` for framework-managed sharding.  |
| Artifacts            | Safetensors weights plus a versioned, checksummed manifest.                                   |

Construction-time and collation-time validation catch most configuration and
geometry mistakes. The final scatter also fails closed rather than silently
placing embeddings at the wrong prompt positions.

## Installation

Nemotron Stitch is shipped as source in BioNeMo Recipes; it is not published to
PyPI. From a BioNeMo Recipes checkout, install it in a virtual environment:

```bash
cd recipes/nemotron-stitch
python -m pip install -e .
```

To install an immutable BioNeMo Recipes revision without cloning the full repository:

```bash
python -m pip install \
  "nemotron-stitch @ git+https://github.com/NVIDIA-BioNeMo/bionemo-recipes.git@<revision>#subdirectory=recipes/nemotron-stitch"
```

For development, use the checked-in lockfile:

```bash
uv sync --locked  # installs the package and development tools into .venv
```

Base dependencies are PyTorch, safetensors, NumPy, and PyYAML. The framework
extras are intentionally unpinned because NeMo AutoModel, NeMo RL, and vLLM
must come from one mutually compatible runtime stack. The
[`LLaVA Dockerfile`](examples/llava/Dockerfile) is the qualified reference
environment.

## Package layout

```text
src/nemotron_stitch/
  prompt.py               reserved-token prompt rendering
  lm_config.py            language-model configuration helpers
  contracts.py            forward, ownership, and artifact constants
  provenance.py           canonical metadata and checksums

  projector/              projector implementations, scatter, trainability,
                          and the projector artifact (schema and codec)
  features/               optional encoder, preparation, and cache utilities
  automodel/              NeMo AutoModel integration
  nemo_rl/                NeMo RL policy, rollout, and transport integration
  vllm/                   vLLM plugin and processor construction
  testing/                exported conformance suite
```

## Development

The checkout uses Python 3.12 from `.python-version`; the package continues to
support Python 3.10–3.14. Commit `uv.lock` when changing dependencies with
`uv add` (or `uv add --dev` for development tools). The `uv_build` backend uses
the version in `pyproject.toml`; update it with `uv version --bump patch`. The package
exposes installed metadata as `__version__` (`0+unknown` in an uninstalled
source checkout).

Run the CPU checks and build distributions with:

```bash
uv run --locked pre-commit run --all-files
uv run --locked pytest
uv build
```

The lockfile covers local development. In framework containers, install with
`uv pip install --no-deps .` into the existing runtime; `uv sync` would replace
its framework dependencies.

### CI image

Recipe CI uses a public mirror of NVIDIA's multi-platform NeMo RL nightly
image at `svcbionemo023/bionemo-framework:nemo-rl-ci-b03da0f4-amd64`, pinned
to digest `sha256:08fd971c29f8e76f0d589bc7195104231594e859707fda6fac82388770732140`.
It contains the qualified NeMo RL, AutoModel, and vLLM environments, so CI
only installs the checked-out package. The mirror is copied verbatim from
`gitlab-master.nvidia.com/dl/joc/nemo-ci/main/rl:latest` to make it accessible
to public Docker Hub consumers and GitHub Actions runners.

Once published, run the same folder-scoped path as CI from the repository root:

```bash
./ci/scripts/recipes_local_test.py recipes/nemotron-stitch
```

The framework seam tests must also run in the pinned environment from
[`examples/llava/Dockerfile`](examples/llava/Dockerfile):

```bash
docker run --rm \
  -v "$PWD:/opt/nemotron-stitch" \
  -w /opt/nemotron-stitch \
  -e PYTHONPATH=/opt/nemotron-stitch/src \
  -e PYTHONDONTWRITEBYTECODE=1 \
  svcbionemo023/bionemo-framework:nemo-rl-ci-b03da0f4-amd64 \
  /opt/nemo_rl_venv/bin/python -m pytest tests/ -q -p no:cacheprovider
```

See [`AGENTS.md`](AGENTS.md) for repository conventions, the full worker-venv
test matrix, and the rule for documenting framework workarounds. Current
upstream limitations are tracked in [`docs/upstream-gaps.md`](docs/upstream-gaps.md).

This recipe imports upstream Stitch commit
`c3f389af8316eacaec1906637afe6da81fe99bf0`,
with BioNeMo-specific CI, packaging, documentation, and topology/refit
admission fixes.

## Full decoder SFT

The shared `ProjectorFinetuneRecipe` can use full decoder training in stage 2.
Omit the AutoModel `peft` section and select the decoder parameter names explicitly:

```yaml
projector:
  train_decoder: true
  trainability_decoder_patterns: ["model.", "lm_head."]
  train_projector: true
```

These example substrings match AutoModel's native Nemotron decoder; inspect
your runtime model's names, which can differ from HF checkpoint keys. Every
selector must match at least one decoder parameter.
Projector and extra-family classification takes precedence. Decoder parameters
retain AutoModel's existing trainability so protected parameters stay frozen;
unnamed parameters remain frozen. Missing decoder matches and models containing
LoRA parameters fail closed. Alignment and default LoRA SFT retain their existing
behavior. The optimizer remains `ProjectorAdamWConfig`; upstream AutoModel owns
full checkpoint serialization. CPU optimizer/recipe tests qualify the selection
contract only: full-model distributed capacity and stage handoffs still require
qualification on the target model and hardware.
