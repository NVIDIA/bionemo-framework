# PhiX174 screening rationale and evidence

Supporting evidence for the [current recipe results](../README.md#frozen-rollout-results).
The publication cohort below is selected; it is not a raw-generation baseline.

## ARC selected-cohort waterfall (2026-10-19)

Applying our current v3 screen to ARC's 302 ordered records leaves **two designs**, both labeled
**nonviable** in Science Data S1. The records contain 296 exact-unique sequences; the waterfall
below counts each sequence once. This is attrition within an already selected publication cohort,
not ARC's generation pass rate or a controlled comparison of RL and SFT. In production rollouts,
raw-observation yields remain separate from final duplicate removal and 99%-identity clustering.

### Largest filter exclusions

These are **additional exclusions in the sequential filtering order**, ranked by count.
They are not independent failure rates; changing gate order can change the attribution.

| Filter                                      | Additional unique designs excluded |
| ------------------------------------------- | ---------------------------------: |
| Length                                      |                                 98 |
| Nine-core completeness                      |                                 90 |
| G-protein identity + coverage               |                                 49 |
| Circular-ORF AAI                            |                                 43 |
| ARC count OR structured K+X diversification |                                 13 |
| Core order/copies                           |                                  1 |
| Sequence safety                             |                                  0 |

### What drives core-completeness exclusions?

Among the **149 designs reaching the core-completeness gate**:

| Core assessment                                                          | Unique designs | Gate outcome |
| ------------------------------------------------------------------------ | -------------: | ------------ |
| At least one required family not detected                                |             61 | Excluded     |
| All nine families detected, but at least one fails completeness/coverage |             29 | Excluded     |
| All nine core functions complete                                         |             59 | Retained     |

The first two categories account for all **90 core-completeness exclusions**. These are
annotation-based assessments, not proof of biological absence. The subsequent order/copy gate
excludes one extra-core-copy design; no additional order violation was observed among these 59.

### Supplemental: Full sequential waterfall and attribution notes

| Sequential gate                                                         | Unique designs remaining | Additional exclusions | Viable-labeled designs remaining\* |
| ----------------------------------------------------------------------- | -----------------------: | --------------------: | ---------------------------------: |
| Ordered cohort, exact-unique sequences                                  |                      296 |                     — |                                 16 |
| Length 5,306–5,730 nt                                                   |                      198 |                    98 |                                 15 |
| G-protein identity ≥60%, with ≥95% query and target coverage            |                      149 |                    49 |                                 15 |
| All nine core functions meet completeness criteria                      |                       59 |                    90 |                                 15 |
| Core circular order and no extra full-coverage core copies              |                       58 |                     1 |                                 15 |
| Sequence-safety PASS                                                    |                       58 |                     0 |                                 15 |
| Circular-ORF AAI ≤95%                                                   |                       15 |                    43 |                                  1 |
| Original ARC count diversification OR structured K+X add-one-gene route |                        2 |                    13 |                                  0 |
| Final duplicate removal / 99%-identity clustering                       |                        2 |                     0 |                                  0 |

The unchanged nucleotide-character, GC, homopolymer, ORF and other configured preliminary checks
remove no additional candidates at their positions in this saved run. The independent original-ARC
checks, including its seven-protein-hit requirement, remove none of the 15 candidates entering the
final module before diversification. The added K+X route contributes **zero** ARC designs. The
standalone architecture-diversification cutoff is disabled.

**Which filters dominate?** In this execution order, length removes 98, followed by core
completeness removing 90. Of those core failures, 61 lack at least one detected required family;
29 have all nine families detected but fail at least one full-length/coverage criterion. These are
annotation-based failures, not proof that a gene is biologically absent. The later order/copy gate
rejects one extra-core-copy case, with no additional order violation among its 59 inputs. Safety
passes all 296 unique inputs independently. Measured-only AAI rejects 106 of 198 length-qualified
designs, but only 43 additional designs after the preceding gates. Later metrics were not measured
on every early rejection: their marginal counts cannot be compared using a common 296 denominator
or summed as disjoint losses.

The G-stage loss is also coverage-sensitive: identity alone accepts 195 of its 198 inputs;
requiring a qualifying hit with ≥95% coverage of both query and reference leaves 149. This
coverage requirement is our added constraint; ARC's published description specifies ≥60% identity.
It must not be described as 49 failures of ARC's identity threshold or of the seven-hit gate.

### Why our length window differs from ARC's

ARC's published **Design constraints** section specifies **4–6 kb**, and the released pipeline
template sets `genome_length_range: [4000, 6000]`. All 296 unique ordered sequences fit that range.
Our retained PhiX profile instead uses **5,306–5,730 nt** in both the hard nucleotide gate and final
screening. This is an additional project-specific constraint, not an equivalent implementation of
ARC's length criterion. See the [Science supplement, p. 4](https://www.science.org/doi/suppl/10.1126/science.aec2657/suppl_file/science.aec2657_sm.pdf)
and [released configuration](https://github.com/ArcInstitute/evo2/blob/53f195997257c56c00e5ef8d33a54f5baad143a6/phage_gen/pipelines/genome_design_filtering_pipeline_config_template.yaml).

- **Lower bound, 5,306 nt:** inherited from an earlier narrow PhiX profile whose upper bound was
  5,493 nt. The introducing configuration cites a local 155-sequence PhiX/Sinsheimervirus collection
  spanning 5,339–5,388 nt, but does not derive the exact 5,306 cutoff. It is 80 nt below the 5,386-nt
  reference. Treat it as a heuristic guard against truncation, not an experimentally established
  minimum viable genome length; its precise justification remains weak.
- **Upper bound, 5,730 nt:** the maintained configuration cites engineered PhiX174 insertion
  mutants recovered at that length. Russell and Müller reported these mutants, while noting
  instability and selection of deletions above approximately 5,550 nt. This supports a cautious
  screening choice, not a universal packaging ceiling for diversified genomes. See
  [Russell and Müller, 1984](https://pubmed.ncbi.nlm.nih.gov/6092714/).
- These hard bounds are separate from the graded length reward's four-point envelope. The observed
  lengths of recovered viable phages and the lengths of originally ordered designs are also
  different evidence and should not be substituted for each other.

Our window rejects **40 short designs** (4,117–5,275 nt) and **58 long designs** (5,737–5,998 nt).
Counting ordered records instead gives 41 short and 59 long, because some sequences were ordered
more than once. One excluded long design is 5,948 nt:
`batch_2_evo1_microviridae_ft_5k_temp07_topk4_prompt4bp_genome_3392`. Its publication label is
**“viable, acquired mutation(s) during rebooting.”** That supports success from this nominated
design lineage, not viability of the unchanged 5,948-nt input. No length threshold was changed
for this audit.

### Evidence for shorter genomes and ARC boot outcomes

Shorter-than-wild-type PhiX174 genomes have experimental support, but the endpoint matters:

- **Small deletions in phage genomes:** Russell and Müller (1984) cite Müller and Wells' earlier
  J–F intergenic-region work. Incardona and Müller (1985) directly studied DNA ejection by phage
  particles carrying a 27-base deletion in this region, corresponding to approximately
  **5,359 nt** relative to the 5,386-nt reference. A later H-protein study recovered viable
  progeny retaining a seven-amino-acid deletion after a compensatory mutation. These demonstrate
  tolerance of particular small deletions, not a whole-genome minimum.
  [Russell and Müller references](https://pmc.ncbi.nlm.nih.gov/articles/PMC254601/?page=1);
  [Incardona and Müller, 1985](https://pubmed.ncbi.nlm.nih.gov/3158743/);
  [H-protein length study, 2016](https://journals.asm.org/doi/full/10.1128/jvi.00914-16).
- **Much shorter packaged DNA:** Aoyama and Hayashi (1985) reported efficient production of
  infectious particles with DNA **78.5–101%** of wild-type length in an in-vitro packaging system.
  The lower endpoint is approximately **4,228 nt**, calculated from 5,386 nt. These were
  plasmid-derived templates with phage machinery supplied separately; this does not demonstrate
  a complete, independently propagating phage at 4.23 kb.
  [Aoyama and Hayashi, 1985](https://pubmed.ncbi.nlm.nih.gov/3161888/).

**All 16 ARC designs labeled viable had original input lengths above our 5,306-nt minimum.**
This includes nine labeled viable and seven labeled viable with mutations acquired during
rebooting. The shortest successful input was **5,349 nt (99.31% of wild-type length)**:
`batch_3_evo2_microviridae_ft_12k_temp09_topk4_prompt9bp_genome_36`. It is labeled **viable**,
without the acquired-mutations qualifier. These are original ordered-design lengths and
publication outcome labels, not lengths inferred for recovered mutant genomes.

| Original design length                  | Exact-unique designs | Failed DNA synthesis | Tested designs | Viable-labeled |
| --------------------------------------- | -------------------: | -------------------: | -------------: | -------------: |
| Below 5,306 nt                          |                   40 |                    2 |             38 |              0 |
| 5,306–5,385 nt (shorter than wild type) |                   22 |                    0 |             22 |              1 |

Counts use the deduplicated Science Data S1 outcome audit; synthesis failures are excluded from
the tested denominator. The 40 nominated short inputs span 4,117–5,275 nt, and there are no
ordered designs at 5,276–5,305 nt. Thus the cohort is consistent with a
conservative lower screen, but cannot establish length as the cause of failure or validate the
precise 5,306-nt boundary. That boundary is approximately **98.5% of wild-type length**, or an
80-nt allowance; the numerical relationship does not establish its original derivation.

### Experimental outcomes and interpretation

A separate diagnostic relaxed only the length window for the excluded 5,948-nt design and
measured all nine core functions as complete, all nine in expected circular order, and zero extra
core copies. Its exact sequence also has safety PASS in the fully reconciled original cohort.
Thus **all 16 viable-labeled designs pass core+safety when evaluated independently of the length
gate** (nine labeled viable as designed; seven with mutations acquired during rebooting). The
production waterfall still excludes the long design before core evaluation. This diagnostic does
not change the length cutoff or the production acceptance counts.

\*The viable-labeled column includes nine designs labeled viable and seven labeled viable with
mutations acquired during rebooting. The two final passers are
`batch_3_evo1_microviridae_ft_5k_temp09_topk4_prompt11bp_genome_66` and
`batch_3_evo1_microviridae_ft_5k_temp11_topk4_prompt4bp_genome_3`; both are labeled nonviable.
Fifteen viable-labeled designs survive core and safety; AAI novelty removes 14, and the final
diversification intersection removes the remaining one. These novelty constraints are not
viability predictors.

ARC applied diversification routes separately and manually selected candidates. Our combined
screen requires AAI plus a diversification route and extra core/safety constraints; v3 also adds
the independently evidenced K+X alternative. Consequently, these endpoints do not establish a
simple strict-subset relationship. A design failing our combined screen could still be eligible
for ARC's AAI-only selection route. Circular-AAI correction and original-inventory AAI diagnostics
remain explicitly distinguished in the [final-filter contract](../skills/bionemo-phage-design/references/arc-final-filter-contract.md).

### Evidence for core conservation among AAI-diverse designs

To ask whether core conservation helps retain bootable, AAI-diverse candidates, compare designs
passing **circular-ORF AAI ≤95%** with the same length and G-protein gates, **without requiring
either original ARC count diversification or the structured K+X alternative**. This is a
retrospective comparison within ARC's selected cohort, using exact-unique ordered sequences.

| Core+safety outcome among AAI passers                | Successfully synthesized/tested designs | Viable-labeled | Nonviable |
| ---------------------------------------------------- | --------------------------------------: | -------------: | --------: |
| Pass nine-core completeness, order/copies and safety |                                      15 |              1 |        14 |
| Fail core completeness                               |                                      48 |              0 |        48 |

One additional core-incomplete AAI passer failed DNA synthesis and is excluded from the
experimental denominators above. All designs in this comparison pass safety. Among these AAI
passers, the completeness gate accounts for all 49 computational exclusions; the subsequent
order/copy checks remove no additional designs. This evidence therefore supports the **more
stringent core gene set/completeness requirement**, but does not independently demonstrate an
incremental benefit from the order/copy checks or safety. Order/copy checks remain explicit
biological constraints rather than an experimentally isolated benefit in this comparison.

Core filtering retains the sole viable-labeled AAI passer while excluding 48 nonviable designs.
The observed viable-labeled fraction rises from **1/63 (1.6%) before core filtering to 1/15 (6.7%)
afterward**. The successful lineage is
`batch_3_evo2_microviridae_ft_12k_temp09_topk4_prompt9bp_genome_5`, labeled **viable with mutations
acquired during rebooting**. Neither group contains a design demonstrated to boot unchanged.

This is **suggestive support for retaining the stringent core filter**, not a validated estimate
of enrichment or a causal claim: there is only one positive outcome, the cohort was already
selected by ARC, and acquired mutations separate the nominated input from the successful phage.
Fourteen core-complete AAI passers were still nonviable. Larger prospective experimental testing
is needed to establish whether the filter improves boot rates for newly generated diverse designs.

These counts are derived by joining the saved length, G, required-gene, core-order/copy, circular-AAI
and safety outcomes to Science Data S1 labels by exact sequence hash. They do not substitute
the original ARC-inventory AAI diagnostic for our explicitly versioned circular-AAI gate.

Provenance: frozen `arc-selected302-matched-screen-20261014` target-profile count/metric tables,
Science Data S1 sequence-hash/phenotype join, and
`phix174-arc-diversification-audit-20261019` circular-AAI/v3 replay. The supplement SHA-256 is
`f7f4c0dee438c625f7fbb85d2ec9124d43681a41d0dd8e0264169de3b8d32c5a` and Data S1 workbook SHA-256 is
`7796ed7f2cc201d2cc6bd990f297f4326f4d8355b74cff0447584b0fbf4fbef7`.
