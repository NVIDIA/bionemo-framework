---
name: nemotron-stitch-extend-example
description: Add or adapt a Nemotron Stitch example for an external encoder and paired data, using the LLaVA recipe as the reference for training and serving integration.
---

# Extend a Stitch example

## Locate the recipe checkout

Prefer the BioNeMo Recipes checkout supplied by the user, then a compatible
nearby checkout. Let `$BIONEMO_RECIPES` denote its root. It must contain:

- `$BIONEMO_RECIPES/recipes/nemotron-stitch/AGENTS.md`;
- `$BIONEMO_RECIPES/recipes/nemotron-stitch/examples/llava/README.md`;
- `$BIONEMO_RECIPES/recipes/nemotron-stitch/examples/llava/quickstart.md`; and
- `$BIONEMO_RECIPES/recipes/nemotron-stitch/src/nemotron_stitch/contracts.py`.

If no compatible checkout is available, acquire
`https://github.com/NVIDIA-BioNeMo/bionemo-recipes` without overwriting existing
work. Record the checkout root, recipe root, and revision when available.
The skill may be installed separately: do not resolve the recipe relative to
this skill's installation path.

Read the selected recipe's `AGENTS.md`, LLaVA README, and quickstart before
editing. Resolve the paths below from the selected recipe root. Check the
current recipe implementation and image pins before applying version-specific
recommendations; they can differ from the snapshot described here.

For DNA or molecular inputs, also read
`$BIONEMO_RECIPES/recipes/nemotron-stitch/examples/README.md` and the selected
`$BIONEMO_RECIPES/recipes/nemotron-stitch/examples/genetics/README.md` or
`$BIONEMO_RECIPES/recipes/nemotron-stitch/examples/chemistry/README.md`.
These examples use perception endpoints and typed chat blocks, packed
alignment/SFT, and full decoder SFT. LLaVA remains the LoRA and inference
reference. Each example is an independently installed vLLM plugin; install it
in every generation-worker interpreter, including executor subprocesses.

## Build the application boundary

Keep each example independently installable, testable, and copyable. Examples
must not import from one another. Prefer a small application module and direct
public APIs over a shared example framework.

The application supplies:

- encoder construction, preprocessing, and output decoding into feature tensors;
- dataset normalization and paired text;
- prompt rendering, placeholder IDs, and soft-token counts;
- base-model registration and framework configuration; and
- rewards and domain evaluation when reinforcement learning is needed.

Use `build_multimodal_host` for AutoModel and `build_mm_plugin` for vLLM. The
application owns their callbacks. Preserve the flat feature/index contracts,
including ragged batches and projector ownership. Use the package's portable
projector artifacts for stage handoffs.

Keep the learning path visible: preparation, projector alignment, SFT,
evaluation, and the requested RL or inference stage. Match only the parts of
LLaVA that the use case needs. Start with one runnable configuration and state
its model, data, framework revisions, and hardware assumptions.

Keep typed input blocks in chat order, including multiple inputs, histories,
and text-only records. DNA spans derive from sequence length and the configured
pooling window; use a matching cache and encoder width. Gene preparation uses
`python prepare.py --config configs/alignment.yaml`: YAML supplies
`model.mm_projectors[0].mm_hidden_size` (8192) and
`dataset.chat_template_kwargs.pooling_window` (64). Set stage YAML widths to
1920 for Evo2-1B and keep supervised/GRPO settings aligned; the removed
`STITCH_ENCODER_WIDTH` and `STITCH_POOLING_WINDOW` variables no longer configure
them. New caches are required when geometry changes. Chemistry preserves
canonical atom/bond indices and separate molecule, atom, and bond projectors.
Per-record `chat_template_kwargs` override collator defaults for native
reasoning/direct-answer mixtures. `PackedManifestFeatureDataset` accepts a
sample `collate_fn` for multiple projectors and rebases their scatter indices.
Packed loading retains the complete planning metadata alongside cached features;
the sample collator must reproduce the planned next-token IDs and labels.
A differing encoding is rejected rather than padded to an obsolete budget.
`projected_span_processor` retains per-input rollout items in chat order.
The projected vLLM plugin admits explicitly empty start/end sentinel strings;
its placeholder string must remain nonempty. Keep policy spans and rollout
placeholder counts aligned.

The domain examples use production optimization settings and multinode GRPO
topologies. Their eight synthetic records only exercise integration; bounded
runs require explicit step/data overrides. The eight-row manifest can pack
into fewer than eight entries, leaving every rank empty under the pinned
AutoModel sampler. Use the separate repeated manifest view documented in the
shared example README for eight-rank sanity runs. More epochs or a loader
`drop_last: false` setting cannot repair the sampler's dropped rank groups.
Prefix every AutoModel dotted CLI override with `--`; bare assignments are
silently ignored by the pinned parser. Do not present them as quality
measurements or launch production configs on an undersized local allocation.
Domain processor wrappers delegate `save_pretrained` to their native base
processor; reconstruct the wrapper from YAML when loading checkpoints. Verify
that wrappers survive pickle and deepcopy for worker transport. Keep GRPO
variant checkpoint directories separate from each other, alongside their log
directories. Gene SFT retains one recent checkpoint plus checkpoints referenced
by native pointers such as `LATEST` and `LOWEST_VAL`. The
gene mean-loss alignment defaults require the AutoModel fix identified by its
README and qualified candidate image; the stock CI image does not admit that
configuration. Gene GRPO uses EP8 with `dp_replicate_size: 1` and an inferred
four-way DP shard axis on 32 GPUs. For a bounded smoke from a barely trained
donor, disable `grpo.use_dynamic_sampling` to avoid uniform-reward resampling
that never reaches an update; production sampling remains unchanged.

Super has a separate runtime with AutoModel 0.8.0, Transformers 5.18, and compatible NeMo RL APIs; the default CI image does not qualify Super.

For GRPO the projector is frozen by default. Joint projector training requires
module ownership, `mm_projector_trainable: true`, and an admitted policy mesh.
Use the LLaVA trainable-projector configs as the reference. The pinned stack
admits a constant-only `ConstantLR(factor=1.0, total_iters=0)` schedule for late
projector parameter groups; milestone schedules fail at construction. Combined
dense replication and expert parallelism is rejected. Module-owned projectors use vLLM IPC or NCCL refit; SGLang and checkpoint-engine
refit are rejected even when the projector is frozen. Keep colocated external
DP generation's `async_scheduling: false` explicit for vLLM 0.29.0 pending GPU qualification.

Full-parameter GRPO disables LoRA and adapter restoration and uses native
consolidated decoder weights with frozen projectors. Async generation selects
`EncoderVllmAsyncGenerationWorker` and its matching frozen interpreter through
`ENCODER_GENERATION_WORKER_FQN`. External-DP async generation requires a
separate framework backport; native reload correctness and resume remain
qualification targets. The LLaVA full-parameter Lightning control is unsupported
pending resolution of upstream refit corruption; retain it for failure reproduction and do not
use it for training. Do not infer qualification from finite smoke updates.

## Keep framework fixes in the integration layer

An example owns domain normalization, encoder semantics, token geometry, and
evaluation. Feature transport, artifacts, registration, distributed runtime,
and general training or generation fixes belong in Stitch or upstream.

When a concrete upstream bug or missing capability obstructs adding a custom
modality, record the affected revision, reproduction, integration path, and
required behavior in
`$BIONEMO_RECIPES/recipes/nemotron-stitch/docs/upstream-gaps.md`. Cite its stable
`U-` identifier in the workaround. General backend bugs, performance tuning,
minor API inconveniences, qualification reminders, and package bugs stay outside
this tracker. Continue independent example work. Do not hide a framework monkey
patch or compatibility shim in an application helper.

## Verify and explain the recipe

Run tests and training inside the image pinned by
`$BIONEMO_RECIPES/recipes/nemotron-stitch/examples/Dockerfile`, using the driver,
AutoModel worker, and sync/async vLLM worker interpreters appropriate to the seam.
The maintained stack uses NeMo RL `9a842983`, AutoModel `b916107a`, and vLLM
`0.29.0`, with `AutomodelPolicyWorker` environments. Cover feature alignment,
placeholder geometry, masking, packed loss weights, KL-reference donor state,
and CPU-offloaded artifact export. Host static checks supplement container tests.

The README should give runnable commands and explain stage handoffs, shapes,
trainable state, and supported topology. Report measured results with the
split, hardware, training budget, checkpoint, and metric definition. Separate
smoke checks from quality measurements and state any verification limitations.
