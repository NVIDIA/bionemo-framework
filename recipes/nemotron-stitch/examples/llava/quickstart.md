# Quickstart: adapt the example to your modality

This guide shows how to connect your own encoder and paired dataset to Nemotron,
then run the alignment, SFT, and GRPO workflow in `examples/llava/`. You provide
the domain-specific data and encoder code; `nemotron-stitch` supplies the shared
training and serving integration for NeMo AutoModel, NeMo RL, and vLLM.

The example uses CLIP and images, but the same structure works for other
encoders, including encoders that return a different number of features for
each input. To run the image example unchanged, follow the
[`README`](README.md).

## How the recipe works

A frozen encoder turns each domain input into one or more representation
vectors, which this package calls **features**. A small trainable **projector**
maps those representations into the language model's embedding space. The
projected vectors are **soft tokens**, which occupy reserved positions in the
prompt alongside ordinary text.

In the image example, CLIP returns 49 representation vectors, each of width
768\. The MLP projector maps the resulting `[49, 768]` tensor to 49 soft tokens
of width 3136 for the 4B language model. More generally, `[T, F]` means `T`
encoder positions, each represented by `F` values. A token-preserving projector
maps that tensor to `[T, H]`, where `H` is the language model's embedding width.
`T` may vary by input. A resampling projector may instead change the number of
rows.

The image example runs the encoder once per training row and caches the result.
Alignment, SFT, and GRPO then read that cache, while inference runs the same
frozen encoder online for each new input. An application can use a different
feature-loading path. In either case, vLLM receives projected soft tokens, not
the raw input or the encoder itself.

## What you provide

Application-specific Python belongs in one module. The reference is
[`llava_example.py`](llava_example.py). It contains:

- encoder and dataset access;
- row normalization and prompt construction;
- base-model registration; and
- projected-token registration for vLLM.

You also provide an AutoModel config for alignment and SFT, plus a NeMo RL
config if you use GRPO. The example uses the package's NumPy-backed feature
cache, dataset, and collator. Applications with another storage format can
provide their own data-loading boundary and still use the package's projector
lifecycle, rollout transport, and serving plugin.

## Before you start

You need:

- A frozen encoder and a data path that presents its output to the projector as
  a PyTorch tensor. The encoder's native result may be a tensor, a byte payload,
  or a service response; it does not need to be a NumPy array.
- Paired examples containing a domain input, a user prompt, and a desired
  assistant answer.
- Three unused, single-token entries in the language model's tokenizer: a
  start token, a repeated placeholder token, and an end token.
- Descriptive examples for alignment and task-style question/answer examples
  for SFT.

The example's 64-row slices and short schedules are integration smoke tests,
not useful training settings.

GRPO is optional. Add it only when generated answers can be scored
automatically, such as by comparing them with exact numeric answers.

## Step 1: write the application module

Copy the structure of [`llava_example.py`](llava_example.py) and replace the
application-specific pieces described below.

### Encoder and dataset access

The image example loads both its encoder and dataset from the Hugging Face Hub,
so its access functions use pinned repository revisions:

```python
ENCODER_REPO_ID = "openai/clip-vit-base-patch32"
ENCODER_REVISION = "..."
DATASET_REPO_ID = "lmms-lab/LLaVA-OneVision-Data"
DATASET_REVISION = "..."
```

Your encoder may instead use a local checkpoint, a service, or a web API. Your
dataset may be any iterable of source rows. Cache metadata is optional; when
you have stable identifiers, pass them to `prepare_features` so they are
recorded with the cache.

Choose a short projector name for prompts, model configuration, and serving:

```python
PROJECTOR_NAME = "image"
```

### Reserved tokens

Describe the three tokenizer tokens that delimit a run of soft-token slots:

```python
PLACEHOLDER_SPEC = PlaceholderSpec(
    PROJECTOR_NAME,
    "<SPECIAL_30>",  # start
    "<SPECIAL_32>",  # repeated once per soft token
    "<SPECIAL_31>",  # end
)
```

Verify that each string already maps to one token in the base model's
tokenizer. The model and collator configs use the numeric ID of the repeated
placeholder token. The start and end tokens are ordinary vocabulary tokens;
only the repeated positions receive projected embeddings.

### Row normalization

Implement the source iterator and row normalizer:

```python
def open_dataset_stream(configuration: str):
    """Return an iterable of source rows without loading the whole dataset."""


def normalize_row(row, *, configuration: str, source_index: int):
    """Return (record, encoder_input) for one valid source row."""
```

`record` has this shape:

```python
{
    "sample_id": "stable-unique-id",
    "prompt": "start + T placeholders + end, followed by the user's text",
    "target": "the desired assistant answer",
    "ground_truth": "optional value used by the GRPO reward",
}
```

Use `render_soft_token_prompt` to expand the `{mm:<projector-name>}` marker.
Pass the number of soft tokens that this row's projector will emit. For a
token-preserving projector, that is the feature-row count; for a resampling
projector, use its output geometry. The CLIP example uses 49 for every image.

The count must be available while normalizing the row. For variable-length
inputs, derive it from source metadata or input geometry before the encoder is
run. The collator checks each token-preserving `[T, F]` row against its prompt.

Raise `InvalidRow` for an expected bad row that should be skipped. Raise
`ValueError` when a row violates the dataset schema.

Update `PARTITIONS` to select the examples used by each stage and split. Its
intervals refer to valid-row ordinals; skipped rows do not consume an ordinal.

### Encoder and inference input

Implement these two functions:

```python
def build_encoder(device: str):
    """Return the application-specific encoder callable."""


def load_input(path):
    """Load one inference input in the form expected by the encoder."""
```

Keep preprocessing intrinsic to the encoder—resizing, tokenization, pooling,
and similar transforms—inside `build_encoder`. Decode dataset payloads in
`normalize_row`, and decode user-supplied inference files in `load_input`.

The LLaVA callable returns a NumPy array because `prepare_features` writes the
package's `.npy` cache. That is a property of this preparation helper, not a
requirement on encoders. If your encoder returns another representation, such
as BF16 bytes, decode it directly to a PyTorch tensor in your data-loading code;
there is no need to round-trip through NumPy.

### Base-model registration

Declare each base model that the application can register:

```python
NANO_4B = ModelSpec(
    architecture="LlavaExampleNemotronForCausalLM",
    repo_id="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
    revision="...",
    hidden_size=3136,
    hub_remote_code=True,
)
```

`architecture` is the name registered with AutoModel. `hidden_size` is the
language model's embedding width. Set `hub_remote_code` when the snapshot
provides the model implementation; use the native AutoModel class otherwise.

Snapshot paths, tokenizer options, LoRA targets, and framework-specific model
overrides belong in the training configs.

To add another base model:

1. Add its `ModelSpec` and include it in `MODEL_SPECS`.
2. Copy an alignment/SFT config pair and update the model block and paths.
   AutoModel configs are self-contained, so each stage needs a complete model
   block.
3. For GRPO, add a model contract under `configs/models/` and select it from
   the stage config's `defaults` list.

The `NANO_4B` and `LIGHTNING_30B` paths in the example provide a concrete pair
to compare. The package supplies `build_multimodal_host` and `build_mm_plugin`;
the application's `build_host_cls`, `register_models`, and `register_vllm` only
bind those shared implementations to your names and base classes.

## Step 2: write the training configs

Use `configs/` as the reference. Alignment and SFT are plain
AutoModel configs. GRPO is a NeMo RL config composed with a per-model contract.

### Alignment and SFT

Both stages use the package dataset, collator, and recipe:

```yaml
recipe: nemotron_stitch.automodel.recipe.ProjectorFinetuneRecipe

dataset:
  _target_: nemotron_stitch.automodel.data.ManifestFeatureDataset
  manifest_path: outputs/data/manifest.jsonl
  cache_root: outputs/data/feature-cache
  stage: alignment
  split: train

dataloader:
  collate_fn:
    _target_: nemotron_stitch.automodel.data.collate_fn
    tokenizer_name_or_path: artifacts/models/your-model
    projector_name: image
    placeholder_token_id: 32
```

The model block supplies the registered architecture, projector configuration,
and placeholder-token map. The dataset and collator infer feature geometry and
per-row token counts from the data.

The `projector` block connects the recipe to your application and controls the
portable projector artifact:

```yaml
projector:
  model_registry_callback: your_application.register_models
  train_projector: true
  initial_artifact: outputs/alignment/projector  # SFT only
  artifact_dir: outputs/sft/projector
```

Alignment trains the projector with the language model frozen. SFT loads the
alignment artifact and trains the projector together with a LoRA adapter. Add
`provenance` and `expected_provenance` to this block if you want artifact loads
to enforce application-specific identities.

Configure model paths, LoRA, the optimizer, the schedule, checkpoints, and
distributed resources with the usual AutoModel settings.

### GRPO

NeMo RL supports config composition through `defaults`. Put model-specific
settings in one file under `configs/models/`, then compose it after the
upstream recipe:

```yaml
defaults:
  - <nemo-rl checkout>/examples/configs/vlm_grpo_3B.yaml
  - models/<your-model>.yaml
```

The model contract contains the snapshot, registered architecture, projector
configuration, warm-start paths, LoRA targets, sentinel strings, and vLLM
limits. Keep the projector frozen by default (`mm_projector_trainable: false`,
`mm_projector_ownership: sidecar`); to train it during GRPO instead, follow
the llava example's `grpo-trainable-projector.yaml` overlay — module
ownership, an explicit `projector_lr`, an unforced ratio, and a nonzero KL
penalty (see its README section for why each is required). Configure the
callbacks in the stage file:

```yaml
policy:
  model_registry_callback: your_application.register_models
  generation:
    mm_plugin_callback: your_application.register_vllm
    vllm_cfg:
      enforce_eager: false
    vllm_kwargs:
      compilation_config:
        mode: none
        cudagraph_mode: full_decode_only
      max_num_seqs: ${policy.generation_batch_size}
```

vLLM needs an upper bound for profiling and dummy inputs. Set
`mm_encoder_max_tokens` to the largest projected payload you will serve. Each
request may use any positive token count up to that limit. Full decode CUDA
graphs leave projected-embedding prefill eager and amortize their capture cost
over sustained training; `mode: none` keeps Inductor from changing the rollout
kernels.

The GRPO data block reads cached features, applies the frozen projector for
rollout, and carries `ground_truth` into the reward environment. Configure the
reward functions under `env` for your answer format. If you do not have a
meaningful automatic reward, stop after SFT.

### Keep shared names aligned

Use the same names at each framework boundary:

- The `ModelSpec.architecture` value must match `architectures` in the model
  config.
- `PROJECTOR_NAME` must match the `mm_projectors` entry, the placeholder-token
  map, the collator's `projector_name`, and the GRPO adapter name.
- The collator and model configs must use the numeric ID of
  `PLACEHOLDER_SPEC.placeholder_token`.
- GRPO's dataset and vLLM overrides must use the same start, placeholder, and
  end strings.

Bad names fail during model construction, collation, or soft-token scatter.

## Step 3: prepare and train

Build and start the qualified container as described in the
[`README`](README.md#setup). These commands run the image example from its
`/opt/llava-example` working directory; substitute your module and config paths
for another application.

Prepare the bounded dataset snapshot and feature cache:

```bash
docker exec llava-example python -m llava_example
```

Run alignment, then SFT:

```bash
docker exec llava-example \
  python -m nemo_automodel.cli.app configs/alignment.yaml --nproc-per-node 1
docker exec llava-example \
  python -m nemo_automodel.cli.app configs/sft.yaml --nproc-per-node 1
```

If the application has an automatic reward, continue with GRPO. The example
loads the latest SFT adapter through its `checkpoints/LATEST` link:

```bash
docker exec llava-example \
  python -m nemotron_stitch.nemo_rl.runner --config configs/grpo.yaml
```

Alignment exports a projector. SFT exports the updated projector and a LoRA
adapter. GRPO keeps the projector frozen and updates the LoRA adapter. The base
language-model snapshot is unchanged throughout.

For inference, load a new domain input, run the frozen encoder, project its
features, and serve the resulting soft tokens with the selected LoRA adapter.
The image example's command is in the
[`README`](README.md#inference).

## Runtime checks

For token-preserving `[T, F]` features, the collator checks that each row has
`T` placeholder positions and emits explicit flat scatter indices. The model
checks that projector outputs fit those positions and that their width matches
the language model. Other feature layouts are stacked normally and validated
by the projector and final scatter.

Projector artifacts always carry checksums and their projector configuration.
Provenance locks are optional: add them when your encoder, dataset, or model has
a stable identity that later stages should verify.
