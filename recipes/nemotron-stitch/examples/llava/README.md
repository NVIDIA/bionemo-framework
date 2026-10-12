# Minimal LLaVA example

A small, reproducible vertical slice that adds a frozen CLIP encoder and a
LLaVA-style projector to several base models, then runs all three
post-training stages end to end. The qualified control is text-only
Nemotron-3-Nano-4B. The Qwen paths demonstrate the same seams with non-NVIDIA
bases: dense Qwen3.5-4B and Qwen3.6-35B-A3B using AutoModel's native MoE
backend. Both keep Qwen's built-in vision tower frozen and unused.

1. **alignment** — train only the image projector (`mlp2x_gelu`, 768 → the
   base model's hidden width) against the frozen language model;
2. **SFT** — warm-start the projector from the alignment artifact and train
   it jointly with a LoRA adapter on the language model's attention and
   mamba-input projections, plus the shared expert on the MoE target — every
   projection family that survives the GRPO handoff; the SFT configs document
   the excluded ones (the LLaVA finetune stage); and
3. **GRPO** — warm-start SFT's projector and LoRA adapter and keep updating
   only the adapter, so the whole recipe carries a single adapter, never a
   merged copy. The opt-in
   [`grpo-trainable-projector.yaml`](configs/grpo-trainable-projector.yaml)
   variant trains the projector jointly with the adapter instead; see
   [the variant's contract](#grpo-with-a-trainable-projector).

### DPO preference demonstration

The optional [`configs/dpo.yaml`](configs/dpo.yaml) exercises preference
training with the same frozen CLIP features and SFT projector/LoRA handoff.
It makes deterministic chosen/rejected pairs from the CLEVR manifest: the
recorded numeric answer is chosen, and the next integer is rejected. This is
an integration demonstration, not a preference-quality experiment. After data
preparation and SFT, run from `/opt/llava-example`:

```bash
python -c 'from llava_example import prepare_dpo_preferences; print(prepare_dpo_preferences())'
python run_dpo.py --config configs/dpo.yaml
```

[`run_dpo.py`](run_dpo.py) selects the package's U-54 DPO launcher, which calls
NeMo RL's public `setup_preference_data(..., processor_fn=...)` and its
unchanged DPO setup and training functions. The package processor loads one raw feature tensor per
pair, attaches it independently to the chosen and rejected message logs, and
rejects a branch whose placeholder count differs from the feature count. The
stock preference collator interleaves the branches and carries their packed
feature tensors in that order. The feature loader is a callback: it can read a
cache, run a local encoder, or call a service, provided it returns a nonempty
`[tokens, hidden]` tensor and keeps the two branches aligned with the prompt.
An online source should be pinned and deterministic across reference and
policy passes.

NeMo RL already has a VLM preference processor for supported image processors
and a multimodal-capable DPO collator. Its pinned DPO launcher chooses the text
processor and has no configuration key to select the VLM processor or an
external one. U-54 tracks that launcher seam; the LLaVA entry point is
temporary until NeMo RL exposes processor selection. The two-pair data path
has been checked against NeMo RL `4d969c93`; a DPO model update has not yet
been qualified for this example.

Serving is part of the design. GRPO's rollout engine is plain vLLM: at every
update the current adapter is merged into the base weights and refit into the
running engine (CUDA-IPC colocated, NCCL across two GPUs), and each prompt's
image reaches vLLM as 49 already-projected soft tokens — the projector
(frozen again at this stage) runs in the rollout data path over cached CLIP
features. The CLIP encoder runs once per training row during preparation, not
inside training or GRPO rollouts. vLLM never imports CLIP, sees a raw image, or
holds a separate adapter object. Inference on a new image runs the same frozen
encoder once before calling vLLM.

Everything modality-neutral — preparation, cache, dataset, collator, projector,
scatter, artifact codec, training recipe, RL transport, and workers — is in the
[`nemotron-stitch`](../../README.md) package. The common application code is
[`llava_example.py`](llava_example.py): encoder and dataset access, row
normalization, and the Nemotron bindings. [`llava_qwen.py`](llava_qwen.py) is
the small Qwen-specific model-registration boundary. Configs hold the
values their frameworks consume directly — model/projector construction,
token IDs, LoRA settings, paths, and resources. Shapes, per-row token counts,
and artifact provenance are not duplicated into YAML. Scripts run or qualify
the example but contain no application contract to copy.

**Adding your own modality?** Read [`quickstart.md`](quickstart.md) first: it
walks the three stages with the exact pieces you would change, written for
domain scientists rather than framework experts.

| Role                        | Selection                                                                                                                            | Pinned revision                            |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------ |
| Language model (4B control) | `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`                                                                                              | `dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f` |
| Language model (non-NVIDIA) | `Qwen/Qwen3.5-4B`                                                                                                                    | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| Language model (MoE smoke)  | `Qwen/Qwen3.6-35B-A3B`                                                                                                               | `995ad96eacd98c81ed38be0c5b274b04031597b0` |
| Language model (target)     | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`                                                                                  | `b3caaabed0263651a17dc1f2d4ce97e794f76c44` |
| Frozen encoder              | `openai/clip-vit-base-patch32`                                                                                                       | `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` |
| Data                        | `lmms-lab/LLaVA-OneVision-Data`, configurations `diagram_image_to_text(cauldron)` (alignment) and `CLEVR-Math(MathV360K)` (SFT/GRPO) | `7ca5e5bf8b2006d5dfa0549756198474f0897f63` |

## Data terms

The OneVision repository is labeled Apache-2.0 on its dataset card **and**
separately limits use to academic research and education; this example
surfaces both statements and does not try to reconcile them. The selected
components are `diagram_image_to_text(cauldron)` (from the CAULDRON
collection) and `CLEVR-Math(MathV360K)` (from MathV360K). Confirm the terms
fit your use before running preparation. No source image or dataset row is
committed to this repository.

## Setup

The environment (unmodified NeMo RL main, nesting AutoModel r0.6.0, vLLM
`0.25.1`, torch `2.11.0+cu130`, transformers `5.12.1`) is built from the
`recipes/nemotron-stitch` directory:

```bash
docker build -f examples/llava/Dockerfile -t llava-example .

# Any GPU SKU: --gpus all exposes every device (restrict a subset with
# --gpus '"device=0,1"'). --cap-add SYS_PTRACE is for the colocated GRPO
# stage's CUDA-IPC weight refit (pidfd_getfd). The mounts persist the HF
# cache, the model snapshots, and the training outputs on the host.
docker run -d --name llava-example --ipc=host \
  --gpus all --cap-add SYS_PTRACE \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$PWD/examples/llava/artifacts:/opt/llava-example/artifacts" \
  -v "$PWD/examples/llava/outputs:/opt/llava-example/outputs" \
  llava-example
```

The base is a public mirror of NVIDIA's NeMo RL nightly at `b03da0f4`, pinned
in the Dockerfile to the AMD64 digest
`sha256:08fd971c29f8e76f0d589bc7195104231594e859707fda6fac82388770732140`.
The public mirror is currently AMD64, matching BioNeMo CI runners. The example
uses the base's prebuilt
AutoModel worker environment for preparation, alignment/SFT, and the GRPO
controller; NeMo RL selects separate environments for its Ray workers.

On hosts whose NVIDIA driver predates CUDA 13, also mount the CUDA
forward-compat shim (`-e LD_LIBRARY_PATH=/opt/cuda-compat -v /path/to/cuda-compat:/opt/cuda-compat`; any CUDA-13 container's
`/usr/local/cuda/compat` provides it). For a quick host-side setup without
Docker, install the root package with `pip install -e .` in an environment
carrying those pins and put `examples/llava` on `PYTHONPATH`; the Dockerfile is
the qualified path.

Inside the container, fetch the immutable snapshots and build the data cache.
The language model downloads straight to the artifact path every stage's
config references (`--local-dir` writes real, resumable files); the encoder
stays in the HF cache, where `llava_example.py` resolves it by repo id and
pinned revision:

```bash
docker exec llava-example hf download nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 \
  --revision dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f \
  --local-dir /opt/llava-example/artifacts/models/nemotron-nano-4b-bf16
docker exec llava-example hf download openai/clip-vit-base-patch32 \
  --revision 3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268
docker exec llava-example python -m llava_example
```

Preparation streams a bounded sample (the `PARTITIONS` table in
`llava_example.py`: valid-row ordinals 0–79 per training configuration, plus
80–159 for the GRPO slice; scan capped at 512 rows), runs the frozen CLIP
encoder once, and retains only the package feature cache plus a JSONL manifest — no
image files. It writes `outputs/data/{feature-cache,manifest.jsonl,...}`,
streams through a preparation-owned temporary datasets cache, and never
touches your global one. Budgets: 512 MiB transient, 64 MiB retained
(measured: 36.5 MB retained). The Nemotron targets share `outputs/data`.
Qwen uses `outputs/data-qwen` because its tokenizer needs a different set of
single-token sentinels. Pass those values and the output path directly to the
same preparation entry point; the underlying CLIP features and row selection
are otherwise identical.

## Training

The 4B control's three stages run on one GPU with at least 80 GB HBM:

```bash
docker exec llava-example python -m nemo_automodel.cli.app configs/alignment.yaml --nproc-per-node 1
docker exec llava-example python -m nemo_automodel.cli.app configs/sft.yaml --nproc-per-node 1
docker exec llava-example python -m nemotron_stitch.nemo_rl.runner --config configs/grpo.yaml
```

The SFT checkpointer maintains `checkpoints/LATEST`, which the GRPO config reads
directly; there is no checkpoint path to copy by hand.

### GRPO with a trainable projector

The default stage 3 keeps the projector frozen. The opt-in
[`configs/grpo-trainable-projector.yaml`](configs/grpo-trainable-projector.yaml)
overlay trains it jointly with the adapter:

```bash
docker exec llava-example python -m nemotron_stitch.nemo_rl.runner \
  --config configs/grpo-trainable-projector.yaml
```

The projector must be module-owned on this path (`mm_projector_ownership: module`): sidecar parameters live outside the module tree, so DCP cannot save
their optimizer state and FSDP2 never manages their gradients. The
worker re-enables gradients after AutoModel's post-wrap PEFT freeze, appends
the projector to the policy optimizer as one group (`policy.projector_lr`,
1e-5 here — SFT trained the projector at 1e-4, the adapter trains at 1e-6),
and keeps the audit fail-closed: nothing outside the adapter and the projector
may train. The optimizer's LR schedule must be the constant-only passthrough
(`ConstantLR(factor: 1.0)`, what every shipped GRPO config carries): NeMo RL
builds the scheduler before the worker extension runs, so any other schedule
either rescales the projector group wrongly or crashes at a milestone, and
the worker rejects it at construction (U-65) instead of mid-run. The trainable
projector is mesh-agnostic within the package's admitted topologies (TP=1,
CP=1, and not both `dp_replicate_size` and `expert_parallel_size` above one —
the combined replicate×EP mesh is inexpressible on fused-EP layouts on this
stack, U-66): expert-parallel meshes were qualified at
EP=8 on the Lightning-30B target ([`configs/grpo-lightning-trainable-ep8.yaml`](configs/grpo-lightning-trainable-ep8.yaml),
8×H100, 2026-10-01: three finite updates with monotone projector drift and a
nonzero KL leash through an EP=8 policy mesh and an EP=8 colocated vLLM
rollout, warm-started from the EP=8 Lightning alignment/SFT chain), and
data-replicated HSDP meshes were qualified on the 4B control ([`configs/grpo-trainable-projector-hsdp.yaml`](configs/grpo-trainable-projector-hsdp.yaml),
dp_replicate 4 × dp_shard 2, 2026-10-01). AutoModel switches to HSDP at
dp_replicate > 1, so FSDP2 all-reduces projector gradients across replicas;
the worker enforces that every update starts from bit-identical replicas —
each rank's drift is all-gathered and any divergence fails the step closed
— which is also why EP>1 and HSDP meshes share the shard-aware drift and
KL-reference paths.

Each checkpoint carries the trained projector beside the policy shards as the
package's portable artifact (`<step>/policy/weights/mm_projector/`), and the
projector's Adam moments ride the ordinary optimizer state save keyed by FQN.
Resume restores both — the projector's KL reference stays anchored to the
warm start across resumes, mirroring NeMo RL's own anchor-across-resumes
policy for the language model, so a resume never grants a fresh drift budget.
A frozen-stage checkpoint tree resumed into this variant (it keeps a distinct
one) has no projector sidecar and the run says so:
`ENCODER_POLICY_PROJECTOR_RESUME projector_state: warm_start`. The next
stage — DPO warm start, serving, or another GRPO run — consumes the
checkpoint's `mm_projector/` directory directly as `projector_artifact_path`.

Two loss-side changes come with it. First, the rollout data plane still
embeds prompts through the SFT-artifact sidecar, one gradient step behind the
trainable projector; vLLM scores its samples under exactly the embeddings it
sampled with, so the unforced importance ratio (`force_on_policy_ratio: false`) absorbs the projector drift instead of assuming the stale sampling
distribution equals the current one. Second, the KL reference is anchored to
the warm-started projector (the worker patches NeMo RL's reference snapshot,
which is captured before the artifact load), so a nonzero
`reference_policy_kl_penalty` measures real drift including the projector —
relevant because earlier projector+LoRA runs surfaced runaway projector drift
only through a KL spike. The worker logs `ENCODER_PROJECTOR_DRIFT` (relative
L2 against the warm start) before every update as the direct signal; watch it
alongside NeMo RL's reported KL.

The single-GPU GRPO configs use a resident-weight policy:
policy and vLLM base weights stay in HBM after the mandatory
startup handoff, vLLM gets an explicit 4 GiB KV cache, and synchronous
one-update rollouts use `force_on_policy_ratio` instead of recomputing current
policy log-probabilities. Each rollout batch is submitted to vLLM whole. The
per-step full-policy refit is consequently CUDA-IPC-local between resident
workers rather than a repeated base-model round trip through host memory. The
two-GPU config explicitly disables these colocated overrides and uses NeMo
RL's ordinary dedicated-worker lifecycle. GRPO keeps prefill and its projected
embeddings eager, but captures full decode CUDA graphs for sustained rollout
throughput. It explicitly disables Inductor compilation, retaining vLLM's eager
kernels because Lightning's compiled-kernel path measured worse policy/rollout
parity. Graph capture is bounded to the configured rollout batch; its one-time
startup cost is expected to dominate very short smoke runs.

Alignment and SFT use token-budget packing on the native Lightning and Qwen
tracks. The shared iterable dataset scans the JSONL manifest lazily, retains a
bounded buffer of raw rows for shuffling, and tokenizes and loads cached
features only as rows are drawn into the current pack. Its checkpoint state
includes the source offset, shuffle RNG and buffer, and partial pack for exact
resume. Position IDs reset at every sample boundary, while cached CLIP
features are concatenated and their projector indices are rebased into the
packed row.

Lightning uses padding-free THD with Transformer Engine. Qwen uses AutoModel's
supported NEAT form: one indexed token row, greedily filled and then padded to
the configured budget, with a block-causal mask preventing attention across
source-sample boundaries. Dense Qwen and Lightning use 2048-token packs;
Qwen3.6 uses 512 for its one-step GB300 smoke. The Nano configs intentionally
remain unpacked because their decorated Hub implementation does not accept
packed-sequence boundaries. This is online next-fit packing over the shuffled
stream; unlike the old eager path, it does not scan the full corpus for
globally tighter bins.

> **Qwen3.5-4B.** The requested checkpoint is itself a native VLM, not a
> text-only LM. This path decorates AutoModel's native
> `Qwen3_5ForConditionalGeneration` class, loads the downloaded Hugging Face
> snapshot directly, leaves its vision tower frozen and unused, and names the
> external CLIP route `clip_image` so it cannot collide with Qwen/vLLM's
> native `image` modality. No checkpoint conversion or augmented init
> directory is involved.
>
> ```bash
> docker exec llava-example hf download Qwen/Qwen3.5-4B \
>   --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
>   --local-dir /opt/llava-example/artifacts/models/qwen3.5-4b
> docker exec llava-example python -m llava_example \
>   --output-dir outputs/data-qwen \
>   --projector-name clip_image \
>   --start-token '<|vision_start|>' \
>   --placeholder-token '<|vision_pad|>' \
>   --end-token '<|vision_end|>'
> docker exec llava-example python -m nemo_automodel.cli.app configs/alignment-qwen.yaml --nproc-per-node 1
> docker exec llava-example python -m nemo_automodel.cli.app configs/sft-qwen.yaml --nproc-per-node 1
> docker exec llava-example python -m nemotron_stitch.nemo_rl.runner --config configs/grpo-qwen.yaml
> ```
>
> The placeholder spec reserves the 49 token positions where the external
> CLIP projector scatters its soft tokens. Qwen already defines
> `<|vision_start|>`, `<|vision_pad|>`, and `<|vision_end|>` as single tokens;
> its native image and video processors use the distinct `<|image_pad|>` and
> `<|video_pad|>` payload tokens, so this does not masquerade as a native Qwen
> image. The same boundaries are consumed by the policy collator and vLLM
> prompt replacement.
>
> These configs exercise the upstream Qwen3.5 implementations already present
> in AutoModel and vLLM. Packed one-step GPU smoke runs of alignment and SFT
> were recorded on 2026-09-02, and a GRPO smoke was recorded on 2026-09-01;
> unlike the two Nemotron tracks, a full configured training run has not yet
> been recorded.

> **Qwen3.6-35B-A3B native MoE smoke.** This is the same three-stage contract,
> but the training host directly decorates AutoModel's native
> `Qwen3_5MoeForConditionalGeneration`; it does not use Transformers' model
> implementation or convert the checkpoint. The deliberately small 512-token,
> one-step configs are an end-to-end integration run for one 256 GiB GB300,
> not a convergence recipe. GRPO uses 4 prompts × 4 generations so the fixed
> refit is amortized across 16 rollouts. They reuse the Qwen feature cache
> created above.
>
> ```bash
> docker exec llava-example hf download Qwen/Qwen3.6-35B-A3B \
>   --revision 995ad96eacd98c81ed38be0c5b274b04031597b0 \
>   --local-dir /opt/llava-example/artifacts/models/qwen3.6-35b-a3b
> docker exec llava-example python -m nemo_automodel.cli.app configs/alignment-qwen3.6-35b.yaml --nproc-per-node 1
> docker exec llava-example python -m nemo_automodel.cli.app configs/sft-qwen3.6-35b.yaml --nproc-per-node 1
> docker exec llava-example python -m nemotron_stitch.nemo_rl.runner --config configs/grpo-qwen3.6-35b.yaml
> ```
>
> Alignment and SFT explicitly select AutoModel's portable torch expert and
> dispatcher backends for the single-GPU run. GRPO uses the same native model;
> AutoModel falls back to its standard grouped-expert implementation because
> expert parallelism is one. The policy remains GPU-resident beside vLLM on
> the qualified 256 GiB card.

> **Lightning 30B target.** The same three stages with the `-lightning`
> configs, on one GPU with ~100 GiB HBM (measured peak ~92 GiB). Lightning
> loads the pinned Hugging Face snapshot directly. The host composes the
> native family state-dict adapter with the package's projector filter, so the
> fresh module-owned projector is initialized outside the base checkpoint:
>
> ```bash
> docker exec llava-example hf download nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 \
>   --revision b3caaabed0263651a17dc1f2d4ce97e794f76c44 \
>   --local-dir /opt/llava-example/artifacts/models/lightning-30b-bf16
> docker exec llava-example python -m nemo_automodel.cli.app configs/alignment-lightning.yaml --nproc-per-node 1
> docker exec llava-example python -m nemo_automodel.cli.app configs/sft-lightning.yaml --nproc-per-node 1
> docker exec llava-example bash -c \
>   'CUDA_VISIBLE_DEVICES=0 \
>      python -m nemotron_stitch.nemo_rl.runner --config configs/grpo-lightning.yaml'
> ```
>
> The Lightning config also reads its automatically maintained SFT
> `checkpoints/LATEST` link. It uses the same prepared `outputs/data` as the
> 4B configs; the `-lightning` files track the `LIGHTNING_30B` spec and keep
> Lightning-specific output directories and run recipe, with the GRPO
> per-model contract composed from `configs/models/lightning-30b.yaml`.
>
> On an 8×H100 80 GB node (x86_64), alignment and SFT run expert-parallel
> across all eight GPUs — EP=8 shards the 128 routed experts (16/rank) and
> FSDP2 shards the rest, for ~13.6/~12.8 GiB per-rank peaks (qualified
> 2026-09-03; EP=8 is now the config default, so no CLI override is needed):
>
> ```bash
> docker exec llava-example torchrun --nproc-per-node 8 -m nemo_automodel.cli.app configs/alignment-lightning.yaml
> docker exec llava-example torchrun --nproc-per-node 8 -m nemo_automodel.cli.app configs/sft-lightning.yaml
> ```
>
> (Single-GPU runs now pass `--distributed.ep_size=1` to override the EP=8
> default. Batch geometry: the packed microbatch is 1, so the global batch of
> 16 yields 2 accumulation steps at 8 ranks — pass
> `--step_scheduler.global_batch_size=8` for the accumulation-free 8-rank
> geometry. Multi-pack microbatches are unsupported by the packed collator.)
>
> The default sidecar Lightning GRPO policy needs more memory than an 80 GB
> GPU provides. For an eight-H100 module-owned policy, use
> [`configs/grpo-lightning-trainable-ep8.yaml`](configs/grpo-lightning-trainable-ep8.yaml)
> and its documented projector handoff.

### Projected-payload smoke check

With the downloaded base snapshot available at
`artifacts/models/nemotron-nano-4b-bf16`, run this from `/opt/llava-example`
inside the example container:

```bash
/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker/bin/python \
  scripts/qualify_vllm_projected.py
```

The probe generates from random projected embeddings and checks that invalid
feature widths and excessive token counts are rejected. `VLLM-PROJECTED-OK`
indicates the payload checks passed; this probe does not measure model quality.

## Optional candidate image

The default Dockerfile uses the pinned, unmodified NeMo RL stack. To test the
open U-41 memory fixes, build the optional overlay after building `llava-example`:

```bash
docker build -f examples/llava/Dockerfile.candidates \
  --build-arg CANDIDATE_IMAGE=llava-example \
  -t llava-example:rl41-candidates .
```

Run this command from `recipes/nemotron-stitch`. The overlay applies the
[training logprob chunking PR](https://github.com/NVIDIA-NeMo/RL/pull/4114) and
[FSDP output precision PR](https://github.com/NVIDIA-NeMo/RL/pull/4115).
It is a candidate stack with separate qualification requirements; it does not
change the default CI image. Delete each patch layer when the runtime pin
includes that fix.

## Distributed Training

Same files, no second implementation — two GPUs instead of one. Alignment/SFT
run FSDP2 data parallel (global batch and seed unchanged). GRPO runs a
one-rank policy cluster on one GPU and a one-rank vLLM cluster on the other.
That GRPO topology is pipeline/resource parallelism across GPUs, **not**
model sharding — each cluster stays one rank, inside the package's qualified
sidecar-worker boundary. The two-rank AutoModel runs use `torchrun` directly:
the CLI's `--nproc-per-node` wrapper maps out-of-tree recipe targets to
in-repo script paths.

```bash
docker exec llava-example torchrun --nproc-per-node 2 -m nemo_automodel.cli.app configs/alignment.yaml
docker exec llava-example torchrun --nproc-per-node 2 -m nemo_automodel.cli.app configs/sft.yaml
docker exec llava-example python -m nemotron_stitch.nemo_rl.runner --config configs/grpo-2gpu.yaml
```

Eight GPUs (qualified on one 8×H100 node): alignment/SFT stay FSDP2 data
parallel and shrink the micro-batch so the global batch stays divisible by
`local_batch_size × world_size` — a CLI override, not a config fork. GRPO
keeps the policy at its qualified one-rank boundary and scales rollout to
seven TP=1 vLLM replicas, with NCCL refit across all eight GPUs
(`configs/grpo-8gpu.yaml`):

```bash
docker exec llava-example torchrun --nproc-per-node 8 -m nemo_automodel.cli.app configs/alignment.yaml --step_scheduler.local_batch_size=2
docker exec llava-example torchrun --nproc-per-node 8 -m nemo_automodel.cli.app configs/sft.yaml --step_scheduler.local_batch_size=2
docker exec llava-example python -m nemotron_stitch.nemo_rl.runner --config configs/grpo-8gpu.yaml
```

`scripts/run_matrix.sh <container>` runs the whole matrix end to end
(prepare; 1-, 2-, and 8-GPU stages; vLLM probe).

## Artifact chain

```text
base snapshot     artifacts/models/nemotron-nano-4b-bf16/           (pristine; training never writes here)
prepared data     outputs/data/                                     (feature-cache/ + manifest.jsonl + stage slices)
alignment writes  outputs/alignment/projector/                      (mm-projector.safetensors + manifest)
SFT reads it,     outputs/sft/checkpoints/epoch_<K>_step_<N>/model/ (PEFT adapter)
                  outputs/sft/projector/                            (projector, updated — NOT in the PEFT dir)
GRPO reads SFT's projector + adapter,
                  outputs/grpo/checkpoints/step_<N>/policy/weights/ (updated LoRA only)
```

Every artifact is plain Hugging Face format, so nothing needs an export or
conversion step. The SFT step directory also carries optimizer/RNG/dataloader
state beside `model/` for native resume; `model/` itself is a standard PEFT
directory whose `adapter_config.json` records the resolved target modules.
GRPO's layout is NeMo RL's checkpointer (`step_<N>`; `save_period` and
`keep_top_k` in `configs/grpo.yaml`). The GRPO policy loads SFT's projector
export into a sidecar-held projector; the package suite verifies that handoff.
AutoModel's `checkpoints/LATEST` link selects the SFT adapter for GRPO.

## Inference

The trained model is three artifacts — the pristine base snapshot, the SFT
projector export (warm-started from alignment, then jointly trained with the
adapter), and the LoRA adapter — and the base is the one training never
touches. `scripts/infer.py` serves all three together on any image you hand
it: the pinned frozen CLIP encodes the image online (resolved from the HF
cache by repo id and revision, exactly as `llava_example.py` encoded the
training data), the trained projector turns the features into soft tokens, the
adapter is merged into the base weights (the same merge GRPO's refit
performs in memory at every update), and the example's vLLM plugin
generates, with the prompt rendered byte-identically to the GRPO rollout:

```bash
docker exec llava-example \
  /opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker/bin/python \
  scripts/infer.py --adapter outputs/grpo/checkpoints \
    --input /opt/llava-example/outputs/my-image.jpg \
    --question "How many objects are in the image?"
```

`--adapter` accepts the checkpoints root (the latest step is selected and
printed) or one specific PEFT directory — point it at
`outputs/sft/checkpoints` to compare the SFT adapter against the GRPO one.
The merge is pure safetensors on CPU (~8 GiB for the 4B), no peft or model
instantiation. The `outputs/` mount is host-visible, so dropping an image
there is the easy path into the container. The SFT/GRPO slices were
CLEVR-Math, so counting-style questions (answered "The answer is N") show
the trained behavior best.

## Tests

```bash
pytest examples/llava/tests            # CPU: application contracts
LLAVA_EXAMPLE_HUB_TEST=1 pytest examples/llava/tests -k hub   # opt-in pinned-Hub check
pyrefly check                          # types: package + this example (from the repo root)
```
