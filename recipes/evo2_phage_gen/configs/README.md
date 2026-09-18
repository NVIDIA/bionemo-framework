# Evo2 Phage Generation Configurations

This directory contains the maintained inputs for the Microviridae workflow.

- `sft_microviridae_preprocess.yaml` and `sft_microviridae_dataset.yaml` reproduce the
  publication-era random split. The current PhiX case study instead uses
  `evo2_phage_prepare_sft_split`, which writes a cluster-held-out preprocessing config and
  training dataset under the chosen run directory.
- `grpo_phage_megatron.yaml` is a self-contained Evo2/NeMo-RL base configuration.
  `gdpo_phage_megatron.yaml` inherits it and supplies the case-study objectives and training
  settings. Its generation adapter shards the 768-sequence rollout across data-parallel replicas
  and calls Evo2's packed dynamic prefill/batched recurrent decode directly. The separate
  `policy.sequence_packing` switch remains off because it controls gradient-bearing policy and
  loss execution, which requires its own qualification rather than inheriting inference results.
- `arc_genome_design_filtering_local.yaml` configures downstream Arc screening.
  Both RL configs use this profile: individual PHROGs proteins for per-ORF AAI,
  nine named required functions with calibrated family coverage, and function-aware
  circular synteny over the same A/B/C/D/E/F/G/H/J slots (K and A\* excluded).
  The example's stage 00 downloads and SHA256-verifies the separate AAI database;
  no extra opt-in is needed for new runs.
- The `phage_safety_*.yaml` files describe the sequence-safety policy, data sources, and
  reference controls.

Generic NeMo-RL examples are not copied into this recipe. When adapting the RL configuration,
identify the NeMo-RL version selected in `pyproject.toml` or the installed environment, then
consult that version's `examples/configs` and configuration classes. Installed wheels may omit
examples, in which case use the matching upstream source checkout. Keep the Evo2 generation
adapter, whole-genome sequence length, selected SFT checkpoint, and objective/QC behavior from
this recipe while adapting infrastructure-specific fields.

Smoke runs and hardware-specific launches should use command-line overrides and a separate result
directory rather than adding permanent example configs here. Record the resolved settings and
observed tool/database versions with the run.

## Reward weights and GDPO groups

Both RL configs give every enabled scalar component weight 1. The zero fallbacks in
`RewardWeights` and `nemo_rl_env.py` leave optional components out of standalone scalar
scoring when they are not configured; the shipped configs explicitly enable their selected
components. GRPO trains on that weighted scalar mean. GDPO instead trains on the listed
`gdpo_objectives`: normalize each within identical prompt tokens, sum with coefficient 1,
then normalize the combined advantage. Scalar `weight_*` values do not scale or disable
GDPO objectives. Existing safety/EOD gating happens before normalization.

The default training batch has two groups of 384. Check `reward_prompt_group_count`,
`reward_prompt_group_size_min`, and `reward_prompt_group_size_max` on training metrics;
record IDs and generation shards do not define normalization groups.

## Length and termination

Biological length includes the prompt bases and generated bases, excluding control tokens and
EOD. Generation and context limits count tokens instead.

| Setting                                                               | Default          | Meaning                                                                      |
| --------------------------------------------------------------------- | ---------------- | ---------------------------------------------------------------------------- |
| `genome_length_min` / `genome_length_max`                             | 5,306 / 5,730 nt | Hard length-QC interval; also `genome_length_range` in the screening config. |
| `genome_length_reward_lower_full` / `genome_length_reward_upper_full` | 5,359 / 5,550 nt | Full-credit interval for the RL length objective.                            |
| `genome_length_reward_lower_zero` / `genome_length_reward_upper_zero` | 3,000 / 5,800 nt | Reward tapers linearly to zero outside the full-credit interval.             |
| `policy.generation.max_new_tokens`                                    | 6,000            | Maximum generated tokens per response.                                       |
| `policy.max_total_sequence_length` / `max_model_len`                  | 6,144            | Training/generation context budget, including the prompt.                    |

Both RL configs set `env.phage_qc.zero_reward_without_eod: true`. A response without a sampled
EOD receives zero for the entire scalar reward or GDPO objective vector, including safety
rewards. Its actions remain in the training loss (`grpo.overlong_filtering: false`); raw QC
measurements remain available for diagnosis. Set the reward gate to `false` for an ungated
experiment.

The length reward always uses four finite, ordered points (`lower_zero < lower_full <= upper_full < upper_zero`). The hard `genome_length_min` / `genome_length_max` interval controls
binary nucleotide acceptance and valid-only clustering; changing it does not reshape the reward.
The generic Python/FASTA scorer defaults to 2,000 / 4,000 / 6,000 / 8,000 nt for the four reward
points; the PhiX RL and calibration paths explicitly use the values in the table. The FASTA
scorer exposes `--genome-length-reward-lower-zero`, `--genome-length-reward-lower-full`,
`--genome-length-reward-upper-full`, and `--genome-length-reward-upper-zero` for other targets.

The hard interval admits demonstrated longer PhiX genomes; the reward prefers lengths below
the reported instability region above 5,550 nt. The 5,800-nt zero is a shaping parameter, not a
measured packaging limit. Literature citations are beside the YAML defaults.

## Biological checks and novelty objectives

With `external_qc.enable_smooth_reference_rewards`, protein-match integrity is the geometric
mean of E-value significance, baseline-adjusted identity, and native query/target coverage.
`synteny_identity_zero_credit` / `synteny_identity_full_credit` default to 0.05 / 0.90;
the tropism equivalents use 0.05 / 0.95. These identity settings are fractions.
These are direct-reference match targets, with reciprocal-coverage targets of 0.95
for synteny and 0.99 for tropism. For synteny, `synteny_reference_functions` also
admits the same curated families as required genes: each edge uses the stronger
of direct-reference integrity and normalized family coverage. A qualifying family
match earns full gene credit without 90% identity to PhiX or the consensus.
See the [shared function and architecture rules](required_genes.md#synteny-uses-the-same-function-definitions).
For the direct-reference match, any zero factor gives zero credit, with continuous
credit above zero and no additional
integrity cutoff or bonus. The [worked protein-score definitions](../examples/README.md#protein-evidence-synteny-and-diversity)
give the exact formula and its relationship to independent hard QC.

The PHROGs consensus annotation search uses `mmseqs_protein_database_sensitivity: 7.5`
and `mmseqs_threads: 16` in the maintained Arc template. The same search feeds online
required-function and synteny rewards and final screening. These consume all
admitted hits through a global one-to-one ORF/function assignment; the strongest
hit per ORF remains the displayed annotation. The
`protein_database_search` flag enables this search and annotation stage; it does not
filter genomes by hit count. Higher sensitivity recovers significant partial matches
that the prefilter can otherwise miss; the E-value admission and native-coverage/function rules still determine which evidence earns credit.
Its runtime depends on the called protein workload and CPU allocation. This setting is
separate from the small reference-panel search above and the individual-protein AAI search.

Hard QC determines screening eligibility. RL rewards also provide partial credit to guide
learning; a component's full-credit count is not its hard-QC pass count. Novelty measures
similarity, not viability: a known viable genome can legitimately fail a novelty criterion.

| Measurement        | Default behavior                                                                                                                                                                                                                                                                  |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Nucleotide QC      | ACGT only, GC 30–65%, homopolymers ≤10 bases. DUST supplies a separate online low-complexity objective.                                                                                                                                                                           |
| Required functions | Named A/B/C/D/E/F/G/H/J slots with explicit allowed PHROG families, including the viable alternate J. K and A\* are outside this term. See the [biological rationale, formula, and coverage limits](required_genes.md).                                                           |
| Synteny            | All nine curated functions in circular order, with distinct full-coverage ORFs and no extra qualifying copies. Smooth synteny also admits partial reference/family evidence; K/A\* are outside this profile.                                                                      |
| Spike/G match      | Hard QC requires ≥60% identity and ≥95% bidirectional coverage against PhiX174 G. The RL target is stricter: 95% identity and 99% coverage for full credit.                                                                                                                       |
| AAI novelty        | Mean identity of the lowest-E-value individual PHROGs protein hit per called ORF; optional hard cutoff ≤95%. RL scales novelty by `min(hit_ORFs / 10, 1)` to require supporting evidence. Annotation uses a separate consensus database.                                          |
| Gene-A origin      | Four-factor geometric mean of A integrity, baseline-adjusted motif, position, and strong-site uniqueness; not a final hard gate.                                                                                                                                                  |
| Genome diversity   | Online and final clustering require 99% identity and 95% coverage of both genomes, preserving supplied starts and strands. Online inverse-cluster-size credit pools all eligible prompts for the same design goal within each scoring batch and requires hard-length eligibility. |
| Safety             | Separate AMR, toxin, and lysogeny checks. Missing required evidence remains INDETERMINATE, not PASS.                                                                                                                                                                              |

The profile adapts [King et al.](https://doi.org/10.1126/science.aec2657) for architecture
preservation and RL, rather than reproducing every optional diversification filter. In particular,
architectural-distance filter 7 is diagnostic-only. The named-function gate requires
nine distinct matched ORFs, covering the paper's ≥7-hit condition without an additional
generic family-count or coverage quota. The RL full-credit targets are shaping choices,
not experimentally established viability requirements. See the
[score definitions](../examples/README.md#current-phix174-gdpo-score-definitions) for formulas,
eligibility rules, and GDPO versus scalar-GRPO aggregation.

Final screening applies required functions and synteny before diversification. The
launcher supplies `sequence_safety_manifest` and `sequence_safety_input_fasta` to
intersect safety PASS before the optional filter-7 architecture-removal gate and
the final AAI novelty gate. FAIL, INDETERMINATE and safety-input exclusions do not
reach novelty filtering. Standalone Arc runs without a safety manifest remain
safety-unqualified. The architecture **keep** gate stays in upstream homology QC;
it is separate from filter 7's novelty **remove** gate. Final clustering follows
all per-genome filters. The upstream `qc5`/`qc6` filenames are retained; their
numbers no longer specify execution order, and waterfall readers preserve the
order recorded in each run's count columns. Online RL still measures enabled
objectives without applying these final acceptance gates.

When final screening enables `checkv_filter`, only classifications in
`checkv_quality_range` survive. The maintained list includes Low-quality,
Medium-quality, High-quality, and Complete; Not-determined or missing classifications
do not pass. CheckV results are matched by exact FASTA identifier (the first header
token), so similarly named candidates cannot inherit one another's classification.
CheckV remains disabled in online RL scoring.

## Synteny and Arc's codon-landmark score

The active `synteny` objective measures protein/function content, circular order,
and excess homolog copies. Its scorer is
[`score_smooth_synteny`](../src/bionemo/evo2_phage_gen/protein_evidence.py), returning
`SmoothSyntenyScore`; the reward column is `reward_external_synteny`.
`summarize_function_synteny` produces the corresponding hard-QC measurements.
Profiles without a function map use `measure_reference_cluster_synteny` for hard
measurements from LoVis4u protein clusters. All of these functions concern synteny.

Arc's separate `genetic_architecture.py` scores **start/stop-codon landmarks**.
It marks occurrences of ATG/TAA/TAG/TGA and compares their positions with fixed
PhiX whole-genome and gene-module patterns over circular shifts. It does not
identify protein functions. This score supplies no current RL objective.

| Arc setting                                                | Role in the maintained workflow                                                                                                    |
| ---------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `genetic_architecture_filter`                              | Off during RL. Final screening enables the composite landmark-score keep range `[0, 10]`.                                          |
| `genetic_architecture_remove_filter`                       | Off during RL and target-profile screening. The separate filter-7 diagnostic removes whole-genome landmark scores in `[0.9, 1.1]`. |
| `genetic_architecture_visualization_and_synteny_filtering` | Enables an upstream stage that emits synteny, AAI, and required-gene measurements. It does not enable the landmark-score filters.  |

The `genetic_architecture_*` keys and external module filenames retain Arc's
names because the prepared upstream pipeline consumes them. The stage-50 launcher
sets its final-screening flags explicitly; the base template's `false` values
are not a description of every execution mode. The codon-landmark score's
scientific usefulness and continued role in final screening remain separate
review questions from the active synteny reward.
