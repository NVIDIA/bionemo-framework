# `testing/`

The exported conformance surface. A shared library that only shares *code*
drifts, so the contracts the package promises are executable:

- `conformance.py` — `ProjectorContractSuite`: consumers subclass it in their
  own `tests/` with their projector config, and it pins the contract the ports
  rely on — scatter semantics, projector construction/init, sidecar
  round-trip with provenance enforcement.
- `fixtures.py` — synthetic ragged batches for the flat index contract:
  uneven per-row soft-token counts, rows with none, non-contiguous
  placeholder positions. No real collator produces that shape at this layer,
  so the fixture is synthetic by design.
- `recipe.py` — `CheckpointTrackerFake`, a framework-free double for the
  narrow `BaseRecipe` state-tracker slice that `ProjectorSidecarState`
  (automodel/recipe.py) relies on, so the suite runs without importing
  AutoModel.

## Upstream gaps

None owned here. `CheckpointTrackerFake` deliberately mirrors a pinned
AutoModel `BaseRecipe` slice (`24b47e8`); it tracks the `automodel/recipe.py`
seams (see `automodel/README.md`) and shrinks when they close — it is test
infrastructure, not a framework seam.
