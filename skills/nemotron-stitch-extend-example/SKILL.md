---
name: nemotron-stitch-extend-example
description: Add or adapt a Nemotron Stitch example for an external encoder and paired data, using the LLaVA recipe as the reference for training and serving integration.
---

# Extend a Stitch example

## Locate the recipe checkout

Prefer the BioNeMo Recipes checkout supplied by the user, then a compatible
nearby checkout. The selected checkout must contain:

- `recipes/nemotron-stitch/AGENTS.md`;
- `recipes/nemotron-stitch/examples/llava/README.md`;
- `recipes/nemotron-stitch/examples/llava/quickstart.md`; and
- `recipes/nemotron-stitch/src/nemotron_stitch/contracts.py`.

If no compatible checkout is available, acquire
`https://github.com/NVIDIA-BioNeMo/bionemo-recipes` without overwriting existing
work. Record the checkout root, recipe root, and revision when available.
The skill may be installed separately: do not resolve the recipe relative to
this skill's installation path.

Read the selected recipe's `AGENTS.md`, LLaVA README, and quickstart before
editing. Resolve the paths below from the selected recipe root. Check the
current recipe implementation and image pins before applying version-specific
recommendations; they can differ from the snapshot described here.

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

For GRPO the projector is frozen by default. Joint projector training requires
module ownership, `mm_projector_trainable: true`, and an admitted policy mesh.
Use the LLaVA trainable-projector configs as the reference. The pinned stack
admits a constant-only `ConstantLR(factor=1.0, total_iters=0)` schedule for late
projector parameter groups; milestone schedules fail at construction. Combined
dense replication and expert parallelism is rejected. Module-owned projectors use vLLM IPC or NCCL refit; SGLang and checkpoint-engine
refit are rejected even when the projector is frozen. Keep colocated external
DP generation's `async_scheduling: false` explicit for vLLM 0.25.1.

## Keep framework fixes in the integration layer

An example owns domain normalization, encoder semantics, token geometry, and
evaluation. Feature transport, artifacts, registration, distributed runtime,
and general training or generation fixes belong in Stitch or upstream.

When a needed public capability is missing, record the framework and pinned
revision, a minimal reproduction, and the required behavior in
`docs/upstream-gaps.md` only when an open issue is explicitly cited by `U-`
identifier in implementation, example configuration, or build files. Package
bugs do not belong in this tracker. Continue independent example work. Do not hide a
framework monkey patch or compatibility shim in an application helper.

## Verify and explain the recipe

Run tests and training inside the image pinned by the example Dockerfile,
using the driver, AutoModel worker, or vLLM worker interpreter appropriate to
the seam. When testing U-41 memory fixes, use the optional
`examples/llava/Dockerfile.candidates` overlay and report it as a candidate
stack separately from default-image results. Cover feature alignment, placeholder geometry, masking, trainability,
and artifact handoffs. Host static checks supplement container tests.

The README should give runnable commands and explain stage handoffs, shapes,
trainable state, and supported topology. Report measured results with the
split, hardware, training budget, checkpoint, and metric definition. Separate
smoke checks from quality measurements and state any verification limitations.
