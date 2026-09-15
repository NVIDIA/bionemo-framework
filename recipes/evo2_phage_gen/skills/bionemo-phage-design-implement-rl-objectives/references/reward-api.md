# Reward APIs and module responsibilities

| Module                                                                                                                                           | Responsibility                                                                                                                 |
| ------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------ |
| [`qc.py`](../../../src/bionemo/evo2_phage_gen/qc.py)                                                                                             | Sequence normalization, nucleotide measurements, NCBI DustMasker execution, and hard nucleotide acceptance.                    |
| [`reward.py`](../../../src/bionemo/evo2_phage_gen/reward.py)                                                                                     | Reward shaping, weights, aggregation, and composition of the scoring pipeline. Private helpers execute and read Arc artifacts. |
| [`protein_evidence.py`](../../../src/bionemo/evo2_phage_gen/protein_evidence.py)                                                                 | Alignment coverage, distinct-family evidence, reference-locus matching, circular order, and origin scoring.                    |
| [`arc_pipeline.py`](../../../src/bionemo/evo2_phage_gen/arc_pipeline.py), [`external_qc.py`](../../../src/bionemo/evo2_phage_gen/external_qc.py) | Prepare the pinned Arc source and check external-tool prerequisites.                                                           |
| [`nemo_rl_env.py`](../../../src/bionemo/evo2_phage_gen/nemo_rl_env.py)                                                                           | Convert rollout messages to scored sequences, build gated GDPO objectives, and report training metrics.                        |

## Public scoring entry points

`score_sequences` runs nucleotide measurement, optional Arc and cluster scoring, configured sequence-safety scanning, and scalar aggregation. `score_fasta` supplies its FASTA/CSV interface.

For a target-specific workflow that already has measurements, use `add_nucleotide_rewards` independently. It copies its input and invokes no external tools:

```python
from bionemo.evo2_phage_gen.qc import (
    NucleotideQCConfig,
    add_nucleotide_metrics,
    nucleotide_pass_mask,
)
from bionemo.evo2_phage_gen.reward import add_nucleotide_rewards

config = NucleotideQCConfig()
measured = add_nucleotide_metrics(sequences, config)
scored = add_nucleotide_rewards(measured, config)
accepted = nucleotide_pass_mask(measured, config)
```

`accepted` means nucleotide acceptance only. `reward_nucleotide_pass` uses exactly this predicate, including both homopolymer bounds. It does not replace protein, synteny, or safety screening.

| Public function in `reward.py` | Inputs and result                                                                                                                                                        |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `score_genome_length`          | Length in nucleotides and a configured four-point envelope; returns a score in [0, 1], independent of hard min/max.                                                      |
| `score_tropism_identity`       | Percent identity, explicit measured-hit status, and an identity threshold; no hit earns zero. This is the identity curve used when smooth reference scoring is disabled. |
| `score_aai_novelty`            | Mean protein identity in percent; full credit through 95%, then a taper with a 0.25 floor. This curve is a PhiX preference, not a universal viability rule.              |
| `score_aai_evidence`           | Number of measured proteins; linear credit up to ten. Multiply by novelty so absent evidence cannot earn novelty credit.                                                 |
| `score_synteny_counts`         | Matched reference loci, fixed reference count, excess copies, and order violations; returns (reward, reference coverage, copy balance, reference deficit).               |
| `aggregate_rewards`            | Scored columns and positive component weights; adds the bounded weighted scalar reward subject to the safety gate. GDPO instead uses its configured objective columns.   |

The protein-evidence, synteny, and origin functions are public in `protein_evidence.py`: `smooth_protein_match_integrity`, `summarize_smooth_reference_evidence`, `score_smooth_synteny`, `score_function_matches`, `summarize_function_synteny`, and `score_gene_a_origin`. The [worked PhiX reward definitions](../../../examples/README.md#current-phix174-gdpo-score-definitions) identify which functions the shipped profile selects. Keep CSV readers, subprocess arguments, and artifact reconciliation private; those helpers are not standalone biological scoring APIs.

The online reference search in `reward.py` uses MMseqs `easy-search --prefilter-mode 2 -e 1`:
it aligns the small reference-protein panel against every called candidate ORF, avoiding
heuristic prefilter misses before grading significance and coverage. This cost depends on
the number of reference proteins and candidate ORFs; benchmark it when adapting to larger
panels. Separate hard-QC searches keep their own acceptance rules.

Reference-search E-values depend on the current target pool's total protein residues.
Changing scoring-call size or ORF content can change weak-hit credit and E=1 hit inclusion;
record this context when comparing scores. Removing the significance factor from shaping
alone would not remove the search cutoff. The [worked definitions](../../../examples/README.md#protein-evidence-synteny-and-diversity)
document the measured scope and its effect on closely grouped candidates. Do not claim
unconditional batch-composition invariance for this scorer.

`smooth_protein_match_integrity` combines significance, baseline-adjusted protein identity,
and native query/target coverage with a four-term geometric mean. Its keyword settings are
`identity_zero_credit` (default 0.05), `identity_full_credit`,
`reference_coverage_full_credit`, `candidate_coverage_full_credit`, and the guarded
`significance_zero_evalue` / `significance_full_evalue` endpoints (defaults 1 and 1e-5).
Identity settings are fractions; the measured identity argument is in percent. There is no
additional integrity cutoff, minimum-credit bonus, or adjustable exponent. All endpoints must
be finite and ordered. The shared score feeds smooth synteny, tropism, and gene-A origin evidence;
retain their separate full-credit targets and final-QC rules. The 5% baseline is a shaping
choice, not a homology acceptance threshold. The [worked definitions](../../../examples/README.md#protein-evidence-synteny-and-diversity)
provide the exact factors and config values.

The PHROGs consensus annotation search is configured separately by
`mmseqs_protein_database_sensitivity` (7.5) and `mmseqs_threads` (16) in the
[maintained Arc template](../../../configs/arc_genome_design_filtering_local.yaml).
It feeds online protein-family/required-function evidence and final screening. Higher
sensitivity recovers significant partial hits missed by the prefilter; it does not relax
native-coverage thresholds or the requirement for distinct functions. Benchmark the
search stage and end-to-end scoring with representative protein workloads and CPU threads.
Do not apply its measured relative slowdown to the entire training step.

## Required functions

`summarize_required_gene_evidence(hits_df, sequences_df, required_families, ...)`
accepts a mapping such as `{"A": ["phrog:713"], "J": ["phrog:2354", "phrog:3780"]}`.
Each key names one required function; its explicit families are alternatives. Families
must be disjoint across functions. Native target IDs are normalized, but annotation
labels do not define matches. The Arc config field is `required_gene_families`.

The reward is maximum one-to-one assigned coverage credit divided by the number of
configured functions. Each edge earns `min(1,qcov/qmin,tcov/tmin)`; defaults and
family-specific overrides have the same normalization. Extra genes and copies cannot
replace missing functions or increase the denominator. Final required-function
acceptance requires all functions to have distinct ORFs meeting both coverage targets.
The private CSV reader checks producer availability and finite, consistent counts;
unavailable evidence earns zero and remains unavailable. Keep this separate from a
successful search with no matching genes.

Use the [biological profile](../../../configs/required_genes.md) for the PhiX function
map, alternate J, conditional essentiality evidence, calibrated coverage, and guidance
for new profiles. `required_genes_integrity_sum` is summed normalized coverage in
this term, not the smooth reference-protein geometric mean. Any profile change needs
replay on viable and disrupted controls before interpreting a new run against old scores.

## Synteny entry points

| Function in `protein_evidence.py`     | Result and role                                                                                                                   |
| ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `score_smooth_synteny`                | Returns `SmoothSyntenyScore`: the continuous synteny reward, content, circular-order, excess-copy components, and assignment.     |
| `summarize_function_synteny`          | Measures matched functions, order violations, and extra copies for hard synteny using curated families.                           |
| `measure_reference_cluster_synteny`   | Writes hard-synteny measurements from LoVis4u protein clusters for an unmapped profile.                                           |
| `reference_synteny_pass_mask`         | Applies the configured completeness, order, and copy rules to hard-synteny measurements.                                          |
| `summarize_smooth_reference_evidence` | Combines reference/family evidence for smooth synteny, tropism, and gene-A origin. Its broader name reflects these three outputs. |

In `reward.py`, `_add_synteny_count_rewards` reads hard measurements, derives the
count-based reward, and records the hard pass. `_add_smooth_reference_rewards`
then replaces the reward when smooth scoring is enabled; it preserves the hard
pass. These are private pipeline readers, not additional objectives.

Arc's separate `genetic_architecture.py` computes start/stop-codon landmark
similarity. It has no active RL objective. The maintained final-screening and
filter-7 diagnostic branches still use it; see the
[Arc configuration distinction](../../../configs/README.md#synteny-and-arcs-codon-landmark-score).
Do not confuse its flags with the upstream visualization/synteny measurement stage.

## Function-aware synteny

`score_function_matches(hits_df, required_families, ...)` returns per-ORF/function
`credit` and `full_length` columns plus measurement availability. Both completeness
and synteny consume this evidence. Search admission comes from the best hit per
ORF against all PHROG consensuses; no additional identity threshold is applied to
allowed-family coverage credit. The current command inherits MMseqs's 1e-3 E-value
cutoff; this is an admission gate, not a continuous significance multiplier in
family credit. This is distinct from the individual-member AAI search.

The Arc config's `synteny_reference_functions` maps canonical reference GFF locus
IDs to distinct names in `required_gene_families`. Pass that mapping and measured
`function_matches` to `summarize_smooth_reference_evidence`. Each mapped slot uses
`max(direct_reference_integrity, family_coverage_credit)`; only mapped loci enter
the synteny denominator. A fully covered allowed-family match earns full
slot credit without the direct route's 90% identity target. The PhiX profile maps
A/B/C/D/E/F/G/H/J and excludes K/A\*. Order and excess-copy penalties still apply.
Tropism and gene-A origin retain their original direct-reference evidence.

`summarize_function_synteny` supplies the final hard-synteny measurements
from full-coverage family matches and called-ORF coordinates. The current profile
allows no missing functions, order violations, or extra qualifying copies. Partial
matches remain graded RL evidence. An unmapped profile still uses reference-only
synteny and its LoVis cluster gate. Use the
[biological profile](../../../configs/required_genes.md#synteny-uses-the-same-function-definitions)
for the 72-nt short-J calling threshold, family-specific coverage calibration, and
rotation-tested natural-homolog scope. A family-recognition score does not establish
compatibility of an arbitrary gene swap in a new genomic background.

## Gene-A origin

`score_gene_a_origin` scans complete 28-nt sites in the assigned A ORF within the configured
offset window and frame. Recognition is the first 10 nt, binding the next 18 nt, and the
overlapping nicking core is positions 4–7. For each region, rescale its match fraction
with `max(0, (fraction - 0.25) / 0.75)`, then compute
`M = nicking² × recognition × binding`. Use the best eligible site's M and combine it
with A-protein integrity, position, and uniqueness as `(A × M × position × uniqueness)**0.25`,
where `uniqueness = 1/max(1,strong_site_count)`.
The position factor is 1 for a selected positive site inside the accepted window/frame,
otherwise 0; there is no distance taper within that window.

The strong-site count scans the circular genome using raw recognition ≥8/10, binding ≥14/18,
and an exact nicking core. These thresholds control the copy penalty only; partial motif
credit does not require passing them. Uniqueness participates in the geometric mean, so two
strong sites multiply the reward by `0.5**0.25 ≈ 0.841`. No A evidence or no above-baseline eligible site gives
zero. The 25% anchor is a uniform-DNA shaping baseline, not a significance test: maximizing
over candidate offsets can give shuffled windows positive credit. This is an online objective,
not a separate final acceptance gate or evidence of whole-genome viability.

## Optional adaptation utilities

GDPO accepts explicitly configured scored-column names with mean, product, or minimum reduction. A target-specific scorer can therefore add its own bounded columns without extending a generic plugin system. Biological objectives must retain safety and EOD gating. Built-in `external_qc.enable_orf` and `enable_coding_density` enable Arc filters; their `reward_external_orf` and `reward_external_coding_density` columns may be selected explicitly as GDPO objectives. They describe the combined Arc ORF filter outcome, not independent graded density curves.

`binary_cluster_deduplicated_pass_mask(scored, pass_mask)` selects one representative per existing cluster from an explicitly supplied acceptance mask. This utility defines no acceptance rules. Final-screening workflows should construct their mask from the actual configured filters and cluster the appropriate candidate pool.

[`host_evidence.py`](../../../src/bionemo/evo2_phage_gen/host_evidence.py) supports collecting auditable host metadata. Use `resolve_ncbi_host_evidence` to acquire and cache accession evidence, then `HostEvidenceTable` and `write_host_evidence_table` to persist the collection. For an existing collection, call `load_host_evidence_table` and `validate_host_evidence_artifacts(table, table_path=path)`; a row's `to_task1_host_evidence()` converts it to the `HostEvidence` consumed by design-scope and safety configuration. The resolver currently handles bacterial host-domain evidence; it does not establish strain-specific host range. These utilities are optional preparation APIs, not automatically invoked by RL.
