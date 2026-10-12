# `features/`

The encoder side of the package: a validated encoder registry and an
immutable, content-addressed feature cache, so alignment, SFT, and GRPO never
re-run a frozen encoder over the same data.

- `registry.py` — `EncoderSpec`/`EncoderRegistry`: one encoder configuration
  pinned by an immutable `(repository, revision, implementation_revision, layer)` identity; structural validation only.
- `cache.py` — fail-closed pooled-feature cache with atomic publication.
  `cache_id()` is the frozen hash of the cache contract and the identity of
  every published directory; changing a field invalidates existing caches by
  construction.
- `preparation.py` — the bounded walk from application-owned source rows to a
  published cache (partitioning, publication, manifest files).
- `chunking.py`, `pooling.py`, `manifest.py`, `types.py`, `protocol.py` —
  the cache's field types, pooling geometry, and manifest files.

Applications own source access, row normalization, and encoder construction;
this subpackage owns everything shareable past that line.

## Upstream gaps

None. The cache identity contract and encoder registry are package-owned
contracts, not framework workarounds — no upstream framework has a seam here
to close.
