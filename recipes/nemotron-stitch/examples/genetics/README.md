# Genetics: DNA features into Nemotron

Start with [data/sanity.jsonl](data/sanity.jsonl): eight hand-authored
base-count questions over synthetic ATCG sequences, including a prompt with two
sequences. [prepare.py](prepare.py) sends the raw sequences to an Evo2
perception endpoint and builds the feature cache and all three stage manifests.
No biological data download or corpus builder is needed.

Each JSONL record uses a chat message with interleaved content blocks:

```json
{
  "sample_id": "dna-pair",
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "Consider this DNA: "},
    {"type": "dna", "sequence": "ACGT"},
    {"type": "text", "text": "\nAnd this DNA: "},
    {"type": "dna", "sequence": "AA"},
    {"type": "text", "text": "\nHow many bases in total? Answer in \\boxed{...}."}
  ]}],
  "answer": "6"
}
```

Each `dna` block supplies both a raw encoder input and its position in the
prompt. The block stores only the raw sequence; it does not store an encoded
token count. `format_dna` derives the span length from the sequence and Evo2
pooling configuration, then emits the start token, placeholders,
and end token when the native `apply_chat_template` is called. Other content
types pass through unchanged to the base processor. Alignment, SFT, and GRPO preserve the surrounding
text and its order. The raw sequence is sent to the encoder, not inserted as
text into the language-model prompt. Each sequence is encoded independently
(window pooling never crosses sequence boundaries), then its features are
concatenated in content order. Repeated sequences reuse encoding while keeping
every occurrence and its own start/end span.

`Evo2Processor(ProcessorMixin)` wraps the checkpoint's existing processor,
loaded with `AutoProcessor.from_pretrained`. It delegates native inputs and
chat formatting to that instance. An optional `sequence_processor` callable
can supply feature fields through `processor(text=..., sequences=..., **kwargs)`;
the demo's cached training path already carries those features in the dataset.

Follow the [setup, local endpoint, training, and test commands](../README.md).
Set `EVO2_URL` and optionally `EVO2_TOKEN`, then run `python prepare.py` from
this folder. The client requests `decoder.layers.12` with mean pooling in
64-base windows by default. `STITCH_POOLING_WINDOW` configures the window in
both preparation and all alignment/SFT/GRPO configs; preparation also accepts
`--pooling-window`. Features must be finite
`[ceil(bases / pooling_window), encoder_width]`. Changing the pooling window
requires a matching encoder request and a new cache, not editing dataset rows.
The collator and GRPO transport still reject spans that disagree with actual
cached feature rows. The default
width is 1,920 for Evo2-1B. To use the production Evo2-20B endpoint, set
`export STITCH_ENCODER_WIDTH=8192` before both preparation and training.
Preparation also accepts `--encoder-width`; the YAML projector dimensions read
the same environment variable. Use a new cache/output directory when switching
encoders, and align the model/layer/pooling provenance with your endpoint. The final partial window contains only valid
bases.

[genetics_example.py](genetics_example.py) registers a Lightning host with a
`mlp2x_gelu_norm` projector (`1920 → 4096 → 2688`). DNA features replace token
ID 32 between start/end IDs 30/31. As in chemistry, alignment freezes the
language model; SFT trains its native vocabulary embeddings along with the
rest of the decoder, so boundary tokens learn directly. GRPO uses those saved
checkpoint embeddings and a frozen DNA projector.

Older gene runs with `dna_marker_deltas` need conversion before reuse: fold the
deltas into the input embedding rows and remove their checkpoint/artifact state.
If the input embeddings and output head share weights, preserve the original
output head by untying them before folding. Validate converted model outputs
against the original run, or regenerate alignment/SFT outputs with this demo.

The configs are [alignment](configs/alignment.yaml), [SFT](configs/sft.yaml),
and [GRPO](configs/grpo.yaml), with production training budgets; use native overrides for a bounded sanity run.
Stitch owns the manifest dataset, tokenization, slot validation, projection,
scatter, optimizer selection, and artifact handoffs. Both supervised stages use THD packing. The same eight records serve every
stage; two include short `reasoning_content` fields and enable thinking.
Preparation preserves optional `target` and `chat_template_kwargs`; it defaults
the thinking flag from whether a reasoning field is supplied. GRPO retains
per-record settings while the exact-answer reward scores only the final answer.

To supply your own examples, keep the raw JSONL fields `sample_id`, `messages`,
and `answer`. IDs must be unique. Messages retain their roles and order,
including prior assistant turns. Content can be a plain text string or a list of
text and `dna` blocks; each `sequence` must contain only uppercase A/C/G/T/N.
Text-only records are supported. The GRPO config admits up to eight sequence
blocks per prompt through vLLM's `limit_mm_per_prompt.dna`; override that
upstream setting for larger prompts. The separate `answer` is the ground-truth value to score; `target` optionally
overrides the default `\boxed{...}` training response; prior assistant messages remain context. Prepared
manifests and caches go under ignored `artifacts/data/`. The cache header
records the raw dataset hash and requested encoder layer/pooling; feature
digests identify the returned values. It does not claim to verify the endpoint's
checkpoint identity. For experiments, add verified model provenance using
Stitch's artifact configuration.

The bundled questions and shared stage exposures are integration smoke data;
they do not measure gene annotation quality or generalization.

## Super VL

The DNA example also includes native Super VL wiring, sharing the native host
factory, processor wrapper, THD packer, and encoder configuration. `ModelSpec`
entries in `genetics_example.py` name the native classes for both backends; the
same registration callbacks decorate every entry. To add another compatible
model, supply its native classes in an entry and select its architecture in
the training config. The host
wraps AutoModel's `NemotronOmniForConditionalGeneration`; generation wraps
vLLM's native `NemotronH_Nano_VL_V2`. The vLLM config disables image/video/audio inputs with zero modality limits,
so the native profiler handles DNA without custom profiling subclasses.

Use the separately qualified Super runtime rather than the default Lightning
base. From `recipes/nemotron-stitch`, build the same Dockerfile with this base
(requires NVIDIA registry access):

```bash
docker build -f examples/Dockerfile --target super \
  --build-context perception=../bionemo-perception-containers \
  --build-arg BASE_IMAGE=nvcr.io/1015084509601350/nemotron-gene-async-grpo@sha256:3839e0ca88752a78f17bdc53c1bcfd2584931d245b04445ba710a4109c755b29 \
  -t stitch-domain-examples:super .
```

This runtime carries AutoModel 0.8.0, Transformers 5.18.0 in the driver, policy,
and generation interpreters, and the U-80 NeMo RL compatibility patch. It is separate from the
normal demo's pinned framework matrix. In a shell launched from that image
using the shared mount/GPU instructions, download the Super checkpoint and run:

```bash
hf download nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16 \
  --revision a1ccf94d69377bf82c4db39798aac28447d9c16f \
  --local-dir artifacts/models/super-vl-bf16
# Reuse the prepared DNA cache, with STITCH_ENCODER_WIDTH matching its encoder.
torchrun --nproc-per-node 8 -m nemo_automodel.cli.app configs/alignment-super.yaml
torchrun --nproc-per-node 8 -m nemo_automodel.cli.app configs/sft-super.yaml
python -m nemotron_stitch.nemo_rl.runner --config configs/grpo-super.yaml
# Full language-decoder GRPO, with the same frozen DNA projector:
python -m nemotron_stitch.nemo_rl.runner --config configs/grpo-super-full.yaml
```

The Super config inherits the shared GRPO batching, context, and memory settings.
Size the upstream policy and generation settings for your hardware. Local
eight-H100 memory and communication overrides stay in qualification runs.

Super writes separate `outputs/super-alignment` and `outputs/super-sft`
artifacts. SFT selects `language_model.` parameters while keeping native vision
and audio towers frozen. Native host-class construction and real Super processor
reasoning have CPU checks. The local eight-H100 test verified native vLLM
profiling with zero image/video/audio limits, allowing deletion of the custom
profiling helper. The subsequent policy weight-refit metadata preparation ran
out of GPU memory before any optimizer updates; this is not a completed Super
GRPO qualification. Full-parameter Super GRPO has not been GPU-qualified.
