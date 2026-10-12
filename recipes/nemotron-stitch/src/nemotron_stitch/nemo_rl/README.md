# `nemo_rl/`

NeMo RL integration: the seams that carry encoder features and soft tokens
through GRPO — policy training, rollout transport, and vLLM generation.

- `data.py` — GRPO dataset and a DPO preference processor over modality-neutral
  encoder tensors, with placeholder-count validation and the
  chat-template-kwargs tokenizer proxy. DPO keeps NeMo RL's stock collator.
- `dpo.py` — temporary U-54 launcher selecting an external preference processor
  through NeMo RL's public `setup_preference_data` hook. It otherwise uses the
  stock DPO setup and trainer.
- `policy.py` — DTensor policy worker bridge for arbitrary encoder payloads,
  with the donor-adapter provenance gate over NeMo RL's own compact-PEFT warm
  start. The projector is frozen by default;
  `policy.hf_config_overrides.mm_projector_trainable: true` (with module
  ownership) opts into training it: the worker re-enables gradients after
  AutoModel's post-wrap PEFT freeze, appends the projector to the policy
  optimizer, anchors the KL reference and the drift report to the warm start,
  and persists the trained projector beside each checkpoint as the portable
  artifact. Sidecar ownership does not support joint projector training.
- `runner.py` — owned GRPO bootstrap: registers the worker runtimes behind
  NeMo RL's config-driven worker-extension FQNs.
- `transport.py` — modality-neutral callback helpers moving encoder payloads
  between data and policy planes.
- `vllm_worker.py` — vLLM generation worker: plugin registration,
  generation-only `hf_overrides` composition, and the resident/offload
  lifecycle policy.

Framework imports are lazy; the base wheel never imports NeMo RL.

## Upstream gaps

See the [open framework limitations](../../../docs/upstream-gaps.md) for the
pinned runtime and the conditions for deleting framework workarounds.
