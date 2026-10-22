# Package design and contracts

Nemotron Stitch connects external encoders, projectors, and soft tokens to
NeMo AutoModel, NeMo RL, and vLLM. Applications own their modality, encoder,
data sources, prompts, and evaluation. Training and generation lifecycles
belong to the upstream frameworks. See [AGENTS.md](../AGENTS.md) for the scope
and change rules, and the [package README](../README.md) for usage.

The package was extracted from
[ct-nemotron](https://github.com/NVIDIA-dev/ct-nemotron) and
[genome-research](https://github.com/NVIDIA-dev/genome-research).
Section numbers below preserve the `design §…` references in code and tests.

## 1.5 Vocabulary

- **Encoder**: the frozen upstream model producing features.
- **Features**: encoder output, typically `[N, mm_hidden_size]`.
- **Projector**: the trainable encoder-to-LM module.
- **Soft tokens**: projector output, `[N, H_lm]`.
- **Adapter**: LoRA only.

## 3. Contracts

### 3.1 Forward inputs and routing

Per-projector inputs use flat keyword names:

```text
mm_features__<name>
mm_token_indices__<name>   # optional explicit scatter indices
```

`config.mm_placeholder_token_ids` maps each projector name to a distinct
placeholder token ID. `MultimodalProjector` derives targets from `input_ids`
or validates explicit indices against the configured placeholders. Routing
uses token identity; interleaved projector inputs must retain their positions.

### 3.2 Flat index geometry

Soft tokens are flattened to `[N, H_lm]`. Their int64 indices `[N]` address the
flattened `[B, S]` prompt. Negative indices are dropped; retained indices must
be unique, in range, and match the configured placeholder. Dense `[B, T]`
indices lower to this representation at the projector boundary. Ragged counts
and multiple projectors share the same contract.

See [the projector module](../src/nemotron_stitch/projector/README.md) and
`projector/scatter.py` for the implementation and validation.

### 3.3 Projector ownership

`module` ownership registers projectors in the model tree so distributed
frameworks can wrap and shard them. `sidecar` ownership holds them outside the
module tree; artifact serialization and explicit device placement handle their
state. Sidecar training is admitted only for a single rank. Ownership is
configured explicitly; the host factory defaults to sidecar unless its argument
or `config.mm_projector_ownership` selects module ownership.

Both modes use the same scatter, trainability, and portable artifact contracts.
A distributed topology must pass the relevant integration's admission guards;
module ownership alone does not establish support for every mesh.

### 3.4 Portable artifacts

[`contracts.py`](../src/nemotron_stitch/contracts.py) defines the format
discriminator `nemotron-add-modality/mm-projector-manifest`, schema version,
and filenames. A bare schema-version integer does not identify the format.

```text
mm-projector-manifest.json
mm-projector.safetensors
extra/<name>.safetensors    # optional project state
adapter_model.safetensors  # optional PEFT weights
adapter_config.json        # optional PEFT configuration
```

The manifest records projector configuration, tensor metadata, checksums,
optional provenance, and configuration snapshots. Loading validates structure
and checksums; provenance locks fail on mismatches. Writers emit the package
format; legacy conversion is handled by the artifact migration code.

### 3.5 Trainability

The policy distinguishes projector, extra project state, and LoRA families.
Alignment trains projectors and configured extra state; SFT may add LoRA or
explicitly selected full-decoder parameters. Projector freezing is configurable.
Full-decoder selection preserves upstream-required freezes and rejects
ambiguous LoRA/full-decoder requests or selectors that match nothing.

Apply the policy after PEFT freezing and before optimizer construction. Validate
the complete trainable set and write its deterministic manifest. Framework
workarounds supporting that ordering are tracked as
[U-7](../docs/upstream-gaps.md).

### 3.6 vLLM modes

`build_mm_plugin` provides `projected` mode for already-projected soft tokens
and `encode` mode with application-supplied model and processing callbacks
for raw encoder payloads. The factory supplies the processing and registration
integration. Domain encoding stays in the application.

See [the vLLM integration](../src/nemotron_stitch/vllm/README.md).

### 3.7 Lazy framework imports

The base wheel must import with no vLLM and Transformers 4.48.1. Framework
integration uses lazy imports with sentinel fallbacks. Framework versions are
pinned by the shared [example Dockerfile](../examples/Dockerfile); supported
seams must be checked in that environment.

## 4. Package layout

`projector/` owns projector construction, scatter, trainability, and artifacts;
`artifact/` owns migration; `features/` owns optional encoder/cache utilities;
`automodel/`, `nemo_rl/`, and `vllm/` own framework integration; `testing/` exports
conformance helpers. Domain implementations belong in applications or examples.

## 6. Conformance and parity

[`ProjectorContractSuite`](../src/nemotron_stitch/testing/conformance.py)
checks consumer projector wiring against shared contracts, including scatter,
ownership, artifact round trips, and trainability. Synthetic fixtures include
ragged batches. Refactors must preserve the relevant bit-exact forward-parity
fixtures. Run framework seam tests in the pinned image as described in
[AGENTS.md](../AGENTS.md).

## 7. Encoder and cache contracts

The optional feature layer supplies immutable encoder identities, a unit
protocol, chunking, coverage-weighted pooling, bounded preparation, and an
immutable content-addressed cache. Applications supply source access, encoder
construction, units, and domain policy validators.

`CacheContract.cache_id()` hashes the serialized contract. Its field set and
hashing define published cache identities; changing them requires an explicit
refill plan. Cache publication validates content and fails closed.
See [the feature utilities](../src/nemotron_stitch/features/README.md).

## 8. Schema changes

Changing forward kwargs, index geometry, or the manifest requires bumping
`SCHEMA_VERSION` and shipping a converter in the same change. Preserve artifact
state and the relevant numerical parity through migration. Cover package bugs with focused regressions; framework workarounds belong
in [upstream-gaps.md](upstream-gaps.md), with stable identifiers
and explicit deletion conditions.
