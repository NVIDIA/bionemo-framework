# PhiX174 case-study results

The final-rollout acceptance contract is [versioned ARC-derived filters, explicit syntenic-diversity routes, safety and nine-core conservation](arc-final-filter-contract.md). Historical adapted-core results do not measure the original ARC synteny/count endpoint.

These numbers are context for the documented PhiX174 experiments, not defaults for another target and not evidence of wet-lab viability.

## 2026-10-22 fresh cosine step-900 rollout (v3)

- Checkpoint `fea26289`, step 900; fresh seed 12, BF16 on one GB300, generation batch 64 and
  likelihood microbatch 8. Sampling: temperature 1, top-k 5, top-p 1; 5,000 origin-aligned
  prompts each of 16 and 24 nt, with a 6,000-token generation cap.
- 10,000 raw generations and finite SFT likelihoods; 9,996 authentic EOD and four caps.
  Exact linear-sequence deduplication retained 9,999; 23 invalid-character representatives
  were excluded before safety, and all 9,976 submitted representatives received PASS.
- Current v3 screening completed successfully on 2026-10-22. Core/safety/circular-AAI gates
  retained 5,103 before diversification; **355 raw observations (3.55%)** pass all per-genome
  gates: 351 original ARC-count passes and four K+X-only passes, with no overlap.
  None of the passers has an additional exact duplicate.
- Final 99%-identity / 95%-reciprocal-coverage clustering retained **16 representatives**:
  one cluster of 340 passers and 15 singletons. Accepted-score summaries use these 16 IDs
  with one vote per cluster. Filter 7 separately passes 27 designs; it is diagnostic-only.
- Yield is similar to the earlier changing-policy cosine pool (352/10,176 = 3.46%, 16 clusters),
  not a controlled checkpoint comparison or a claim that the same clusters recur.
  Computational acceptance is not experimental viability.

Retained local artifacts are under
`/scratch/local-jstjohn/phix174-fea26289-step900-seed12-b32-20261021/current-v3-v3-20261022/main10000/rollout/`.
The historical `b32` directory name does not describe the actual generation batch (64).
These run artifacts are not bundled with the recipe:

| Artifact                             | SHA-256                                                            |
| ------------------------------------ | ------------------------------------------------------------------ |
| `final-designs.json`                 | `43d5496f64c97821c9181400c57d6608877c9350e4ac4ab5798cbf0bcf2861e0` |
| `accepted_candidates.fasta`          | `fbf8f87028a745b8563bffdefefcbd4da001de50968a0fd0c96f2069eca255a6` |
| `post-qc-clustering/memberships.csv` | `8ebcce6d60da2dc33f62e6706ec7f436649cadd37ee8a946f4c27588c8f01bcc` |

The separate `1cfffb21` step-1000 rollout remains intentionally held: 1,639 seed-10 records
plus 1,152 seed-11 recovery records are preserved (2,791 total; 7,209 unfulfilled).
It is neither a completed 10k cohort nor an uninterrupted single-seed continuation.

## 2026-10-10 canonical A/B step-400 follow-up

- Frozen checkpoint: `1cfffb21`, step 400, selected best-so-far validation reward 16.714893341064453; not a claim about the training experiment's eventual best checkpoint.
- Training used 8xH100; this evaluation used native BF16 inference on one GB300, seed 9, temperature 1, top-k 5, top-p 1, and 5,000 origin-16 plus 5,000 origin-24 prompts.
- 10,000 raw designs and finite SFT likelihoods; 9,435 exact linear-sequence representatives. This deduplication preserves start and strand, not circular/reverse-complement canonicalization.
- Safety received 9,425 representatives: 9,424 PASS, 0 FAIL, 1 INDETERMINATE. Ten invalid-character representatives were excluded before safety.
- Biological/core/synteny and safety gates retained 5,831 before AAI; median natural-member AAI was 96.31%.
- Mean AAI ≤95% retained 33 target passers: 0.33% of raw, 0.350% of exact representatives, or 0.566% of the pre-AAI survivors. Passer median AAI was 94.41%. None had additional raw exact duplicates.
- Post-QC MMseqs clustering at 99% identity and 95% bilateral coverage retained 8 clusters: one of size 26 and seven singletons.
- The separate filter-7 branch retained 10 designs (0.10% of raw), touching all 8 target clusters. This is not a new clustering of the diagnostic subset; the large cluster's representative fails filter 7 although three members pass it.
- Raw termination was 9,977 authentic EOD and 23 caps. All target and diagnostic passers had authentic EOD.

The frozen final profile uses 5,306–5,730 nt, 30–65% GC, homopolymers ≤10 nt, CheckV,
architecture-KEEP, G/tropism, the nine A/B/C/D/E/F/G/H/J functions in circular order without
excess full-coverage core copies, safety PASS, and then natural-member AAI ≤95%. Nine-function
ordered conservation is not the legacy 11/11 exclusion. Filter 7 is a codon-landmark diagnostic,
not a K/X or direct A/B intactness audit; no such additional audit was performed. The example's
default final rollout size remains 1,000.

The retained result root is `phix174-h100-1cfffb21-gb300-final10000-20261009`.
Its `completion.json` pins `REPORT.md` SHA-256
`b8a7480718718bbcd1950f44cfd797bad59d15b2e7ba351ca2f525c88ab68f0a`,
`final-verification.json` SHA-256
`b17a3c2d347e2f22b9b843eb536dfd7374a546750fec5d6b5b88f3a440c607f5`, and
`main10000/rollout/final-designs.json` SHA-256
`7d210c690a21f12896662834725eac3b853f9b724c6324ad9942f825fe6aab58`.
All three hashes were rechecked for the README update on 2026-10-12. These frozen artifacts,
not the live trainer's latest metrics, source the counts above.

## 2026-10-14 ARC selected-cohort matched rescore

All records in the pinned public
[`all_generated_phages.fasta`](https://github.com/ArcInstitute/evo2/blob/53f195997257c56c00e5ef8d33a54f5baad143a6/phage_gen/data/all_generated_phages.fasta)
were rescored with the retained step-400 runtime, target/safety profile and reference assets on
CPU. The source contains 302 selected synthesis candidates and 296 exact unique sequences;
SHA-256 `532726d1ac343a481a8419539392fdc2c8a27c0be4faeca61acf38fac0af03d6`.
The selected-candidate sequences and IDs match Science Data S1, Sheet 1 (workbook IDs additionally
carry `_umiN`). These are not the unavailable unfiltered rollouts. The linked
[Zenodo release](https://zenodo.org/records/17101843) contains SFT training data, not a substitute
baseline generation cohort.

| Selected cohort | Records / exact unique | Pre-AAI records / unique | Target passes | Filter-7 diagnostic passes | Target clusters |
| --------------- | ---------------------: | -----------------------: | ------------: | -------------------------: | --------------: |
| Evo 1 SFT       |              244 / 238 |                  36 / 35 |             4 |                          4 |               4 |
| Evo 2 SFT       |                58 / 58 |                  23 / 23 |            11 |                          1 |              11 |
| Combined        |              302 / 296 |                  59 / 58 |            15 |                          5 |              15 |

All 296 representatives passed the computational safety screen. The unique-sequence waterfall
was 296 input → 198 nucleotide/length passes → 149 G/tropism passes → 59 required-function
passes → 58 ordered-core and safety passes → 15 AAI passes. Pre-AAI median natural-member AAI
was 98.4027%. Target yield is **15/302 = 4.97% of selected records**, or **15/296 = 5.07% of
unique sequences**. The separate filter-7 diagnostic yields **5/302 = 1.66%**, or 5/296 = 1.69%.
No target or diagnostic passer has an exact duplicate. All 15 target clusters are singletons
at 99% identity / 95% bilateral coverage; the diagnostic subset touches five of these clusters.
It was not separately reclustered. EOD/token metadata is unknown; likelihood was not rescored
and is not an acceptance gate.

Data S1's observed outcomes for all 302 records are 269 nonviable, 17 failed synthesis, nine
viable unchanged, and seven viable with mutations acquired during rebooting. Of the 15 target
passers, 14 are labeled nonviable and one belongs to the acquired-mutation group; all five
diagnostic passers are labeled nonviable. All nine unchanged viable inputs pass the pre-AAI
biological/safety gates but fail the novelty gate. This is a computational design screen,
not an assay of viability, and a mutated descendant is not the original nominated sequence.

### Supplement generation-pool provenance

The supplied [Science supplement](https://doi.org/10.1126/science.aec2657), Fig. S7C (p. 30),
reports the following synthesis-generation pools. Checkpoint and top-k labels come from Data S1.
The displayed initial counts sum to **715,763**: 593,010 Evo 1 plus 122,753 Evo 2. This arithmetic
sum is not a deduplicated raw corpus or a count of all exploratory generations.

| Model / SFT step | Temperature | Prompt nt | Top-k | Initial generations | Selected records / unique |
| ---------------- | ----------: | --------: | ----: | ------------------: | ------------------------: |
| Evo 1 / 5,000    |         0.1 |         4 |     2 |               8,949 |                     6 / 6 |
| Evo 1 / 5,000    |         0.7 |         4 |     4 |             462,311 |                 174 / 168 |
| Evo 1 / 5,000    |         0.9 |         6 |     4 |               1,000 |                     1 / 1 |
| Evo 1 / 5,000    |         1.1 |         4 |     4 |               1,000 |                     1 / 1 |
| Evo 1 / 5,000    |         1.1 |         7 |     4 |               1,000 |                     1 / 1 |
| Evo 1 / 5,000    |         0.9 |        11 |     4 |             118,750 |                   61 / 61 |
| Evo 2 / 6,000    |         0.5 |        99 |     4 |              10,000 |                   12 / 12 |
| Evo 2 / 12,000   |         1.1 |         8 |     4 |               1,000 |                     1 / 1 |
| Evo 2 / 12,000   |         0.9 |         9 |     4 |             111,753 |                   45 / 45 |

The 174 selections in the Evo 1 / temperature-0.7 condition comprise 162 batch-2 and 12 batch-3
records, including six exact duplicate pairs. Thus nine generation conditions map to ten
batch-by-condition strata. Supplement p. 4 and Fig. S7's caption describe separate optional
diversification filters and manual selection, not one uniform sequential filtering funnel.

The nominal 110k SFT parameter sweep (approximately 1,000 × 11 prompts × five temperatures ×
two SFT models) is distinct from these synthesis pools; overlap is unresolved. The previous
README attribution of 15/110,000 to the primary paper was not substantiated and is withdrawn.
Our newly measured 15 passers are a result of this selected-cohort rescore, not a recovered
paper-wide baseline. Neither 15/110,000 nor 15/715,763 is a measured full-cohort matched pass rate.
Do not compute an RL enrichment ratio against the 33/10,000 raw-generation result. Matching the
screen does not remove candidate-selection, model-lineage, checkpoint or sampling differences.

### Retained audit artifacts

Result root: `arc-selected302-matched-screen-20261014` under the recipe's local `results/`.
`selected-record-outcomes.csv` joins every original record, duplicate representative, publication
stratum and observed outcome to its new safety/target/diagnostic/cluster results.
`final-audit.json` verifies counts, terminal sequence/ID mapping, finite AAI, cluster membership
and 30 artifact hashes; SHA-256
`44c98d3b82176944579e1fbcfe4a3ba68638fcf57ffa3ce8ae18677501fc9604`.
`publication-provenance.json` SHA-256
`094d29c365be524ead0a0ff0a76c754f7c5cee1f1ad5def589ce819012b6da61` pins the source audit.

Source hashes: Science supplementary PDF
`f7f4c0dee438c625f7fbb85d2ec9124d43681a41d0dd8e0264169de3b8d32c5a`;
Data S1 ZIP `7378912dc0b5ef1c867a8faf28cbdbd6ed8e2e6fa907bbb5885b17ac2cf020e4`;
inner XLSX `7796ed7f2cc201d2cc6bd990f297f4326f4d8355b74cff0447584b0fbf4fbef7`.
The rescore reuses the completed run's image `5fbd1262…`, frozen source/scorer and read-only
assets; only five input/output paths differ in each screening configuration. Post-QC clustering
uses eight CPU threads instead of 16, with scientific thresholds unchanged. The retained
17-reference positive-control qualification is reused; no fresh control experiment is claimed.

## 2026-08-24 8xH100 rerun

- The final rollout generated and SFT-likelihood scored 1,000 designs; deduplication retained 1,000 representatives before hard QC.
- Pre-safety nucleotide QC submitted 991 representatives and excluded 9.
- Safety screening reported 989 PASS, 0 FAIL, and 2 INDETERMINATE.
- Target hard QC retained 513 safety-PASS representatives.
- Post-QC 99%-identity clustering retained 511 clusters and accepted representatives.

## Earlier completed end-to-end run

- Cluster-held-out SFT used 14,266 training, 100 validation, and 100 test genomes.
- Training ran to a 12,000-step ceiling and selected step 5,600 by validation loss (0.750670); held-out test loss was 0.798180.
- Sampling calibration selected temperature 1.0 and a 50:50 mixture of 16- and 24-nucleotide prompts.
- GDPO ran to 500 steps and selected step 430 from full-QC validation.
- A 1,000-design rollout produced 610 target-profile passes with filter 7 disabled and 22 diagnostic passes with filter 7 enabled.

## Earlier released-SFT shortcut

An earlier RL-only run reused the published Microviridae SFT checkpoint and selected step 190. Its 1,000-design rollout produced 358 target-profile passes and 5 filter-7-enabled passes. Step 190 is historical evidence, not a prescribed stopping point.

These earlier runs used different checkpoints and screening profiles and are not matched baselines
for the step-400 evaluation. Keep validation, final rollout, online reward rates, filter profiles,
checkpoint selections, and post-QC cluster counts distinct when comparing results.

## 2026-10-19 original-ARC and circular-AAI audit of the selected 302

The 15/302 historical result above used the adapted core profile. Requiring original ARC count diversification reduces that set to **2/302**, also two final 99%-identity / 95%-reciprocal-coverage clusters, with existing physical-ORF AAI retained. Additionally requiring original-inventory AAI reduces it to **1/302**. Both cleaned-AAI passers are Evo 1 records labeled nonviable in Data S1. All scientific gate settings match the recent cosine-pool comparison. These selected-record fractions are not raw generation yields.

The v2 final contract chooses physical circular-ORF AAI on annotation evidence: the original pipeline counts a 39-aa prefix peptide that duplicates the tail of the full origin-spanning A call. Both methods omit a separately modeled A\* product; it was not removed from the original reference inventory by the cleanup. Original and cleaned calls recover nine other reference proteins exactly and retain the existing maximal A ORF (522 aa versus annotated 513 aa). The two starts are in frame, 27 bases apart, and share the same stop; expression of the longer form has not been excluded. A extraction stays unchanged, with this ambiguity documented rather than corrected by reference-length trimming. Rotation controls preserve the cleaned peptide multiset while original inventories change. This supports the circular correction, not a claim of perfect gene annotation or score-based selection. Original-inventory AAI remains a diagnostic.

The recent 13-cluster result is specifically the strict v1 endpoint; the completed current-v2 replay below yields 14 clusters. Results and scripts: `phix174-arc-diversification-audit-20261019/selected302` and `biological-reference-control`.

## 2026-10-19 current circular-AAI replay of frozen cohorts

The current `arc-1-6-8-9-circular-aai-plus-nine-core-safety-v2` endpoint uses physical
circular-ORF AAI ≤95%, original ARC filters except architecture diversification,
and nine-core completeness/order/copies plus safety PASS. Original-inventory AAI
is diagnostic. Exact deduplication and 99% nucleotide identity / 95% reciprocal
coverage clustering are applied to the final accepted set.

| Frozen cohort                                | Raw draws / selected records | All per-genome gates | Final clusters |
| -------------------------------------------- | ---------------------------: | -------------------: | -------------: |
| No-product `1dbd6edf` checkpoint290          |                       10,000 |                    2 |              2 |
| Product `1cfffb21` checkpoint400             |                       10,000 |                    6 |              5 |
| Cosine `fea26289`, train789–801 + val790/800 |                       10,176 |                  350 |             14 |
| ARC ordered set                              |                          302 |                    2 |              2 |

The recent pool has 337 passers in one cluster plus 13 singleton clusters. Mean
circular AAI over its 14 representatives is 93.3192%; score summaries weight each
cluster once. These are fixed checkpoints versus a changing-policy pool, not a
controlled attribution to products or schedules. ARC's selected set is not a
raw-generation denominator; both passing ordered designs are labeled nonviable
in the publication metadata.

Current native 72–1800-nt ORF extraction exactly reproduces both nucleotide and
protein inventories in all four cohorts; AAI is recomputed from unchanged native
natural-member hits. Original ARC synteny/count annotation and filtering are
separate from nine-core conservation. All 4,586 recent qualified comparisons
completed; previously cached LoVis tables were reused only for byte-identical
candidate/reference inputs, with 17 fresh overlaps agreeing exactly. The step400
screen originally used 9,435 exact representatives; all 10,000 raw draws were
mapped back by exact sequence hash before pass-rate accounting.

The requested `1cfffb21` step1000 rollout was prepared/imported, but no generation
outputs existed on the 2026-10-19 filesystem check. Its latest receiver receipt
(2026-10-16 05:20 UTC) records GPU contention and an unarmed release. It is not a
completed 10k cohort. No new generation was performed for this comparison.

Source results: `phix174-arc-diversification-audit-20261019/current-v2-replay/`.
Report/figures: `phix174-arc-diversification-audit-20261019/publication/CURRENT-V2-RESULTS.md`.
Historical strict-v1 and adapted-profile artifacts remain unchanged.

## 2026-10-19 explicit structured-addition category (v3)

The default `arc-circular-aai-syntenic-diversity-plus-nine-core-safety-v3` keeps
all common gates and accepts original ARC count diversification OR independently
verified structured K+X addition. The count bounds are unchanged: 11/13 alone
never qualifies. Native ORF assignments must establish exactly one full eligible
K and one full eligible X outside K/core, with no accessory duplicates or extra
types. Each decision and its X family/ORF provenance are exported separately.

| Re-screened cohort                                  | Original ARC-count route | Add-one-gene only | Union observations | Final clusters |
| --------------------------------------------------- | -----------------------: | ----------------: | -----------------: | -------------: |
| No-product step290, 10,000 draws                    |                        2 |                 0 |                  2 |              2 |
| Product step400, 10,000 draws                       |                        6 |                 0 |                  6 |              5 |
| Recent cosine, 10,176 observations                  |                      350 |                 2 |                352 |             16 |
| ARC ordered set, 302 records / 296 unique sequences |                        2 |                 0 |                  2 |              2 |

There is no route overlap in these cohorts. The two additions are recent training
samples step789/index130 and step793/index422; both have K=1, X=1 in PHROG1670,
zero accessory duplicates and two types. Both are original ARC count 11/13 and
form separate new nucleotide clusters. Their original-inventory AAI is 93.11%
and 93.33%, so they are eligible through an AAI route with ARC count diversification
disabled. Added here means added to our AAI AND count-rule intersection, not
proof ARC would have rejected them.

The ARC selected set is reported as counts only, not a comparable generation
pass rate. Unselected outcomes are unavailable; neither the 302 ordered designs
nor a production-generation total supplies a controlled generation-yield
comparison. V3 is an explicit extension and is not claimed to be strictly harder
than ARC. V1/v2 remain reproducible and the earlier results above retain their
named definitions.

## 2026-10-19 ARC waterfall and length provenance

See the screening-evidence waterfall in `$BIONEMO_RECIPES/recipes/evo2_phage_gen/docs/phix174-screening-evidence.md` (section `arc-selected-cohort-waterfall-2026-10-19`). The 296 unique selected sequences follow 296 → 198 → 149 → 59 → 58 → 58 → 15 → 2. Length removes 40 below 5,306 and 58 above 5,730 nt; ARC published 4–6 kb. The lower bound is a heuristic without an established experimental minimum, while the upper bound cites the 1984 insertion-mutant study and is not a universal packaging ceiling. G filtering requires 95% reciprocal coverage in addition to identity. All 296 pass safety; both final passers are labeled nonviable. Viable-labeled counts fall from 15 after core/safety to one after AAI and zero after diversification; novelty rejection is not predicted nonviability.

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

A separate diagnostic relaxed only the length window for the excluded 5,948-nt design and
measured all nine core functions as complete, all nine in expected circular order, and zero extra
core copies. Its exact sequence also has safety PASS in the fully reconciled original cohort.
Thus **all 16 viable-labeled designs pass core+safety when evaluated independently of the length
gate** (nine labeled viable as designed; seven with mutations acquired during rebooting). The
production waterfall still excludes the long design before core evaluation. This diagnostic does
not change the length cutoff or the production acceptance counts.

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
