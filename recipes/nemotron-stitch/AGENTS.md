# Recipe development

The BioNeMo repository's root `AGENTS.md` also applies.

## Scope

Keep the integration thin. Use the public APIs of NeMo AutoModel, NeMo RL,
vLLM, Transformers, PEFT, and PyTorch before introducing owned code. Package
code removes modality-independent friction in connecting an encoder,
projector, or soft tokens to those frameworks. Applications own their encoder,
data, prompts, rewards, and evaluation.

No implementation under `src/nemotron_stitch/` may depend on one modality or
encoder. `examples/` contains application code; use `examples/llava/` as the
reference. Guidance for adding an example lives in
the repository-level `nemotron-stitch-extend-example` skill in
`skills/nemotron-stitch-extend-example/SKILL.md` (from the BioNeMo checkout root).

Framework imports must stay lazy. The base package must import without NeMo
AutoModel, NeMo RL, or vLLM installed. Match the existing code and the recipe's
Ruff configuration. Public signatures need type annotations. Validate geometry
and provenance at construction or collation, and fail on ambiguous inputs.

Use **projector** for the encoder-to-LM module, **adapter** for LoRA,
**features** for encoder output, and **soft tokens** for projector output.

## Framework workarounds

Track open framework limitations in [`docs/upstream-gaps.md`](docs/upstream-gaps.md).
Workaround comments should explain the affected revision and state what
upstream capability would let us delete the code. Cite an open `U-` identifier
while the framework issue remains open. For merged fixes awaiting a runtime pin
bump, retain the deletion condition in the comment without a tracker row.
Keep only open framework issues explicitly cited by `U-` identifier in
implementation, example configuration, or build files. Documentation and test
mentions alone do not qualify. Package bugs do not belong in this tracker.
Remove closed or unreferenced entries; never reuse an identifier. When adopting an upstream fix, delete the workaround.

Preserve numerical parity during refactors. Changes to forward kwargs, index
geometry, or artifact manifests require a schema version change and converter.

## Source synchronization

This BioNeMo recipe is maintained at
`NVIDIA-BioNeMo/bionemo-recipes/recipes/nemotron-stitch`. When importing package
and example updates, use a pinned source snapshot and retain BioNeMo-specific
CI, image, packaging, and discovery changes.
Record the imported commit in the recipe README and BioNeMo PR description.
The repositories have independent histories; do not use a bidirectional subtree.

## Verification

Run framework seam tests in the pinned image declared by
`examples/llava/Dockerfile`. From the BioNeMo repository root, the driver suite is:

```bash
./ci/scripts/recipes_local_test.py recipes/nemotron-stitch
```

Full framework coverage also needs the AutoModel policy-worker and vLLM
worker interpreters. Mount this recipe at `/opt/nemotron-stitch`, set
`PYTHONPATH=/opt/nemotron-stitch/src:/opt/nemotron-stitch/examples/llava`, and run
`python -m pytest tests/ examples/llava/tests/ -q -p no:cacheprovider` with each
interpreter. The worker paths in the pinned image are:

- `/opt/ray_venvs/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2/bin/python`
- `/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker/bin/python`

Install pytest into a throwaway test container if those environments lack it.
CUDA-only tests may skip on a host without GPUs; report these skips and which
framework environments were tested.
