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
  nine required gene-copy slots (K optional), control-calibrated C/E/B coverage,
  and one allowed missing reference locus in the independent synteny gate.
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
Reciprocal-coverage full-credit targets remain 0.95 for synteny and 0.99 for tropism.
Any zero factor gives zero credit, with continuous credit above zero and no additional
integrity cutoff or bonus. The [worked protein-score definitions](../examples/README.md#protein-evidence-architecture-and-diversity)
give the exact formula and its relationship to independent hard QC.

The PHROGs consensus annotation search uses `mmseqs_protein_database_sensitivity: 7.5`
and `mmseqs_threads: 16` in the maintained Arc template. The same search feeds online
protein-family and required-function rewards and final screening. Higher sensitivity
recovers significant partial matches that the prefilter can otherwise miss; the E-value
admission and native-coverage/function rules still determine which evidence earns credit.
Its runtime depends on the called protein workload and CPU allocation. This setting is
separate from the small reference-panel search above and the individual-protein AAI search.

Hard QC determines screening eligibility. RL rewards also provide partial credit to guide
learning; a component's full-credit count is not its hard-QC pass count. Novelty measures
similarity, not viability: a known viable genome can legitimately fail a novelty criterion.

| Measurement             | Default behavior                                                                                                                                                                                                                                                    |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Nucleotide QC           | ACGT only, GC 30–65%, homopolymers ≤10 bases. DUST supplies a separate online low-complexity objective.                                                                                                                                                             |
| Protein-family evidence | At least seven distinct PHROG targets with ≥75% coverage of both the called ORF and target. Multiple fragments of the same family do not increase the family count.                                                                                                 |
| Required functions      | Nine slots assigned to distinct ORFs/targets; K is optional. `required_gene_family_coverage` permits observed C/B truncations and E extensions; other families use 75% bidirectional coverage. These are PhiX function checks, not a universal essential-gene list. |
| Synteny                 | Circular reference order and copy checks, allowing one missing callable reference locus. The required-function check independently tests function completeness.                                                                                                     |
| Spike/G match           | Hard QC requires ≥60% identity and ≥95% bidirectional coverage against PhiX174 G. The RL target is stricter: 95% identity and 99% coverage for full credit.                                                                                                         |
| AAI novelty             | Mean identity of the lowest-E-value individual PHROGs protein hit per called ORF; optional hard cutoff ≤95%. RL scales novelty by `min(hit_ORFs / 10, 1)` to require supporting evidence. Annotation uses a separate consensus database.                            |
| Gene-A origin           | Four-factor geometric mean of A integrity, baseline-adjusted motif, position, and strong-site uniqueness; not a final hard gate.                                                                                                                                    |
| Genome diversity        | Online and final clustering require 99% identity and 95% coverage of both circular genomes. Online inverse-cluster-size credit is computed within prompt groups and requires hard-length eligibility.                                                               |
| Safety                  | Separate AMR, toxin, and lysogeny checks. Missing required evidence remains INDETERMINATE, not PASS.                                                                                                                                                                |

The profile adapts [King et al.](https://doi.org/10.1126/science.aec2657) for architecture
preservation and RL, rather than reproducing every optional diversification filter. In particular,
architectural-distance filter 7 is diagnostic-only. The RL full-credit targets are shaping choices,
not experimentally established viability requirements. See the
[score definitions](../examples/README.md#current-phix174-gdpo-score-definitions) for formulas,
eligibility rules, and GDPO versus scalar-GRPO aggregation.
