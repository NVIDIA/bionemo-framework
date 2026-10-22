# Chemistry: SMILES features into Nemotron

Start with [data/sanity.jsonl](data/sanity.jsonl): eight heavy-atom-count
questions over simple SMILES strings, including a molecule with no bonds and a
prompt with two molecules. [prepare.py](prepare.py) sends raw SMILES to a KERMT
perception endpoint and builds the feature cache and all three stage manifests.
RDKit 2025.9.1 canonicalizes the encoder input and supplies its atom/bond
indices during preparation; learned featurization stays in the endpoint. The
shared image installs RDKit. In another environment, install this example with
its `[preparation]` extra.

Each JSONL record uses the same chat structure as the genetics example:

```json
{
  "sample_id": "molecule-pair",
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "Consider this molecule: "},
    {"type": "molecule", "smiles": "CC"},
    {"type": "text", "text": "\nAnd this molecule: "},
    {"type": "molecule", "smiles": "CO"},
    {"type": "text", "text": "\nQUESTION\nHow many heavy atoms in total? Answer in \\boxed{...}."}
  ]}],
  "answer": "4"
}
```

For `CC`, the prepared molecule block renders as:

```text
<SPECIAL_30>MOLECULE
<SPECIAL_33>
ATOMS
0 <SPECIAL_32>
1 <SPECIAL_32>
BONDS
0 1 <SPECIAL_34><SPECIAL_31>
```

This follows production's indexed graph grammar: one molecule vector, zero-based
atom IDs, and one vector per undirected bond after its two atom IDs. Bond rows
are sorted lexicographically; a bondless molecule writes `BONDS` followed by
`NONE`. Keep the question in the following text block, prefixed by
`\nQUESTION\n`, as the sanity examples do. The example keeps one boundary pair
around the graph, rather than production's historical per-vector wrappers.
That boundary choice remains intentional; prompt text and tokenization are not
claimed bit-identical to those older checkpoints.

Each `molecule` block supplies both a raw encoder input and its position among
the text. Preparation caches features and adds atom/bond counts, `encoder_smiles`, and
ordered `bond_endpoints` to the block. Indices refer to the canonical SMILES
sent to the encoder, not the original spelling.
`format_molecule` in the processor extension expands it into soft-token slots
when the native `apply_chat_template` is called. Other content types, including
images, are delegated unchanged to the base processor. SMILES goes to the encoder, not into the
language-model prompt.

The example defines `KERMTProcessor(ProcessorMixin)` explicitly. Its
`base_processor` comes from `AutoProcessor.from_pretrained`, so a Super VL
checkpoint keeps its Nemotron Omni processor rather than being reduced to a
text tokenizer. For example:

```python
from transformers import AutoProcessor
from chemistry_example import KERMTProcessor

processor = KERMTProcessor(
    base_processor=AutoProcessor.from_pretrained(
        "artifacts/models/lightning-30b-bf16", trust_remote_code=True
    )
)
processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```

The optional `molecule_processor` callable is used by
`processor(text=..., molecules=..., **kwargs)` to supply encoder feature fields.
Native inputs and processing options go to the base instance; its output fields
are preserved. The cached training path supplies features through the dataset,
so it does not run an encoder during chat formatting.

Follow the [setup, local endpoint, training, and test commands](../README.md).
Set `KERMT_URL` and optionally `KERMT_TOKEN`, then run `python prepare.py` from
this folder. The endpoint must support `molecule_fingerprint`, `atom_concat`,
`bond_from_atom`, and `bond_from_bond`, all without pooling.

[chemistry_example.py](chemistry_example.py) registers the native Lightning host
with three independent `mlp2x_gelu` projectors into its 2,688-wide embedding
space. Stitch owns their forward scatter; no custom language-model forward is
needed. Native backend classes live together in `ModelSpec` entries, using
one registration callback per backend, as in genetics. The bundled entry selects
Lightning; another compatible model can be described without adding family
conditionals.

During alignment the decoder embeddings are frozen. SFT trains the decoder,
including its vocabulary embedding table, so molecule start/end tokens 30 and
31 learn directly. Tokens 32–34 are soft-token slots replaced by projector
outputs; their modality representations learn through the three projectors.
No separate marker-delta parameters are needed for this recipe.

| Projector       | Features    | Placeholder ID |
| --------------- | ----------- | -------------- |
| `chem_molecule` | `[M, 3200]` | 33             |
| `chem_atom`     | `[N, 1600]` | 32             |
| `chem_bond`     | `[E, 3200]` | 34             |

KERMT returns atoms in its featurizer's atom order and adjacent reverse directed
bond rows in sorted endpoint-pair order. Preparation concatenates the two
800-wide bond branches, then concatenates each adjacent reverse pair to form one
undirected 3,200-wide feature. It stores `[molecule; atoms; directed bonds]` in
a padded `[M + N + 2E, 3200]` cache matrix for M molecules; atom and
directed-bond rows occupy the first 1,600 columns. Total molecule, atom, and
directed-bond counts in the manifest restore the three native widths. Within
each branch, rows follow molecule content order. Repeated molecules reuse
encoding but retain every occurrence.

The processor renders an ordered molecule/atom/bond list at the molecule block.
One `<SPECIAL_30>...<SPECIAL_31>` pair encloses each entire molecule, including
the ordinary molecule/atom/bond labels and all their soft-token slots. Prepared blocks retain sorted low/high
`bond_endpoints`, matching each pair of directed feature rows. The renderer includes these endpoints as ordinary
text so the model can associate a bond vector with its atom vectors. Feature
order and placeholder order must agree within each branch.

The configs are [alignment](configs/alignment.yaml), [SFT](configs/sft.yaml),
and [GRPO](configs/grpo.yaml), using the indexed full-parameter production
settings documented in the [shared guide](../README.md). Both supervised stages
use one THD pack per microbatch. Per-record thinking settings are retained. The small collator calls Stitch's public feature collator
once per branch to retain its padding, shifted-label, flat-index, and slot-count
validation. Malformed shapes and incompatible geometry raise.

GRPO sends raw branch features to the policy and already-projected vectors to
vLLM. Its processor preserves the authored slot order and ordinary index text;
each transported vector owns compact storage. The vLLM plugin expands those
vectors into bare placeholder slots; the molecule boundaries remain in the
prompt text, so no per-slot start/end pairs are added. KERMT is never run inside
vLLM.

For your own examples, keep raw JSONL fields `sample_id`, `messages`, and
`answer`. IDs must be unique. Messages retain their roles and order, including
prior assistant turns. Content can be a plain text string or a list of text and
`molecule` blocks carrying `smiles`. Text-only turns and records are supported.
The separate `answer` supplies the next assistant response to train or score.
Prepared manifests and checksummed caches live under ignored `artifacts/data/`.
The header records the raw dataset hash and requested layers; returned feature
digests identify the values. It does not claim to verify the endpoint
checkpoint. Use verified artifact provenance for experiments. These eight
examples test integration, not chemistry performance.

Both supervised stages use the same shared THD packer as genetics. Chemistry
configures its `ChemistryCollator` directly; the packer retains each branch's
feature order and rebases its token indices without padding between samples.
`seq_lens` preserves document boundaries. `max_tokens` controls the 16,384-token
pack budget. Optional `target`, `reasoning_content`, and `chat_template_kwargs`
are preserved by preparation; two bundled records demonstrate short reasoning.
The same small dataset remains shared across stages.

GRPO freezes the three SFT projectors and trains the full decoder by default,
using the qualified two-node layout and optimization settings. The inherited
`configs/grpo-full.yaml` is an alias of that default. NeMo RL's built-in boxed
answer verifier runs with zero format bonus, so MC reward is final-letter
accuracy. Bounded qualification and dataset conversion live in the demos
repository; the updated examples have not completed a new end-to-end campaign.

Prepared data from the previous count-only renderer must be regenerated into a
new output directory: molecule blocks now require canonical bond endpoints.
Training and GRPO read the stored metadata and do not parse SMILES or run RDKit.
