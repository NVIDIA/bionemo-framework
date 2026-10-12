# `projector/`

The modality-agnostic core of the package: everything between the encoder's
feature tensor and the LM's input embeddings.

- `projectors.py` — the projector ABC, `build_projector`, external
  `register_projector` seam, and the zoo (`mlp2x_gelu`, `mlp2x_gelu_norm`,
  `perceiver3d`).
- `multimodal.py` — `MultimodalProjector`, the named registry that validates
  and dispatches per-projector soft tokens onto their scatter targets.
- `scatter.py` — the flat index contract primitives: soft tokens `[N, H_lm]`
  plus int64 indices into the flattened `B*S` sequence, with the dense
  `[B, T]` form lowered at the boundary.
- `trainability.py` — the exact trainability policy for the two training
  stages (families: `projector`, `extra`, `lora`).
- `artifact.py` — the projector sidecar: `ProjectorManifest` schema,
  DTensor-safe save/load codec, and `FrozenProjector`, the model-detached
  frozen form used by the GRPO data plane.

Applications reuse these contracts across training and serving.

## Upstream gaps

None. These are package-owned contracts (the zoo, the index geometry, the
artifact schema), not framework workarounds — per AGENTS.md they are the code
that may remain here permanently. A runtime adopting AutoModel's public freeze configuration changes how the
trainability policy is expressed to AutoModel (`freeze_config` selectors), but
the policy itself is ours; see `automodel/README.md`.
