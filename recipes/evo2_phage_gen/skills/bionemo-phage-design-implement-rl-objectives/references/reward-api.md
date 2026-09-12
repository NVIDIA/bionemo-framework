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

`accepted` means nucleotide acceptance only. `reward_nucleotide_pass` uses exactly this predicate, including both homopolymer bounds. It does not replace protein, architecture, or safety screening.

| Public function in `reward.py` | Inputs and result                                                                                                                                                        |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `score_genome_length`          | Length in nucleotides and a configured four-point envelope; returns a score in [0, 1], independent of hard min/max.                                                      |
| `score_tropism_identity`       | Percent identity, explicit measured-hit status, and an identity threshold; no hit earns zero. This is the identity curve used when smooth reference scoring is disabled. |
| `score_aai_novelty`            | Mean protein identity in percent; full credit through 95%, then a taper with a 0.25 floor. This curve is a PhiX preference, not a universal viability rule.              |
| `score_aai_evidence`           | Number of measured proteins; linear credit up to ten. Multiply by novelty so absent evidence cannot earn novelty credit.                                                 |
| `score_synteny_counts`         | Matched reference loci, fixed reference count, excess copies, and order violations; returns (reward, reference coverage, copy balance, reference deficit).               |
| `aggregate_rewards`            | Scored columns and positive component weights; adds the bounded weighted scalar reward subject to the safety gate. GDPO instead uses its configured objective columns.   |

The smooth reference architecture and origin functions are public in `protein_evidence.py`: `smooth_protein_match_integrity`, `summarize_smooth_reference_evidence`, `score_smooth_reference_architecture`, and `score_gene_a_origin`. The [worked PhiX reward definitions](../../../examples/README.md#current-phix174-gdpo-score-definitions) identify which functions the shipped profile selects. Keep CSV readers, subprocess arguments, and artifact reconciliation private; those helpers are not standalone biological scoring APIs.

## Optional adaptation utilities

GDPO accepts explicitly configured scored-column names with mean, product, or minimum reduction. A target-specific scorer can therefore add its own bounded columns without extending a generic plugin system. Biological objectives must retain safety and EOD gating. Built-in `external_qc.enable_orf` and `enable_coding_density` enable Arc filters; their `reward_external_orf` and `reward_external_coding_density` columns may be selected explicitly as GDPO objectives. They describe the combined Arc ORF filter outcome, not independent graded density curves.

`binary_cluster_deduplicated_pass_mask(scored, pass_mask)` selects one representative per existing cluster from an explicitly supplied acceptance mask. This utility defines no acceptance rules. Final-screening workflows should construct their mask from the actual configured filters and cluster the appropriate candidate pool.

[`host_evidence.py`](../../../src/bionemo/evo2_phage_gen/host_evidence.py) supports collecting auditable host metadata. Use `resolve_ncbi_host_evidence` to acquire and cache accession evidence, then `HostEvidenceTable` and `write_host_evidence_table` to persist the collection. For an existing collection, call `load_host_evidence_table` and `validate_host_evidence_artifacts(table, table_path=path)`; a row's `to_task1_host_evidence()` converts it to the `HostEvidence` consumed by design-scope and safety configuration. The resolver currently handles bacterial host-domain evidence; it does not establish strain-specific host range. These utilities are optional preparation APIs, not automatically invoked by RL.
