# Reproducible PhiX final acceptance

The v3 target retains the common ARC filters, uses corrected circular-ORF AAI,
and requires **original ARC count diversification OR a structured add-one-gene
category**. Architecture diversification (7) is off. Independent sequence-safety
PASS and nine-core conservation/order/copies are required for both routes. This
is an explicit extension of ARC diversification, not a claim that every passer
satisfies ARC's literal count gate. RL rewards alone cannot qualify a design.

The maintained final-rollout builder sets
`final_arc_contract: arc-circular-aai-syntenic-diversity-plus-nine-core-safety-v3`. The independent implementation
is `src/bionemo/evo2_phage_gen/arc_final_filters.py`. Its pinned upstream source SHA-256
is `ab1a5d388dcd2085ad79909ea15f3666740d76486f8e4b43d25fdf6fb5553091` (ARC commit
`53f195997257c56c00e5ef8d33a54f5baad143a6`). The online path leaves this setting absent.

| Gate                | Final definition                                                                                                                            |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| ARC nucleotide QC   | A/C/G/T; 4,000–6,000 nt; GC 30–65%; homopolymer ≤10                                                                                         |
| ARC protein support | At least 7 hit-bearing ORFs; not distinct consensus families                                                                                |
| ARC tropism         | A G-protein hit with identity ≥60%                                                                                                          |
| Circular-ORF AAI    | Mean lowest-E-value natural-protein hit per retained physical ORF, first tie retained; ≤95%. Original-inventory AAI is separately reported. |
| Syntenic diversity  | Original ARC count rule (10–12 for each count, excluding 11/11) OR structured add-one-gene evidence; report both decisions separately       |
| Added core gate     | A/B/C/D/E/F/G/H/J complete, ordered, with no excess full-coverage core copies                                                               |
| Added safety gate   | Exact reconciled PASS; FAIL, INDETERMINATE or missing evidence cannot pass                                                                  |
| Final set selection | Exact duplicates, then MMseqs 99% identity with 95% reciprocal coverage; one representative per cluster                                     |

The existing narrower length band, G coverage requirement and CheckV gate remain additional
conditions. The filter-7 branch is a separate diagnostic with architecture enabled, never the
headline endpoint. Both branches also apply the independent common ARC gates and the versioned syntenic-diversity decision.

ARC annotation uses individual natural PHROG proteins, sensitivity 4, and 90–1,800-nt ORFs,
including the original caller's prefix features. Its original **11-feature reference** is staged
separately, correcting only zero-based starts to valid one-based GFF coordinates. The existing
reference-normalization helper is called with `circular_genome_length=None` so it cannot silently
remove the eleventh feature. The original reference SHA-256 is
`16a63389bfce4e32121c7e91b1d2936a5461a4e61eb80a9bef1fc54d517cbb7a`.

The **v2 and v3 AAI acceptance gate uses the existing physical-ORF AAI ≤95%**. Original-inventory AAI
is measured and saved as `original-inventory-aai-diagnostic.csv`, including each threshold
decision, but does not reject designs in v2 or v3. The original ARC synteny/count annotation and
11-feature reference are unchanged. For strict implementation reproduction, select
`arc-1-6-8-9-plus-nine-core-safety-v1`, which requires both AAI gates. Summaries record the
contract, acceptance definition and diagnostic role; never merge v1, v2 and v3 yields. V2 (`arc-1-6-8-9-circular-aai-plus-nine-core-safety-v2`) retains original ARC count-only diversification.

## Structured add-one-gene category (v3)

After all common gates, independently replay the maintained accessory scorer on
physical called-ORF proteins, admitted PHROG hits and a fresh direct-reference
search for core reservations. Require available measurements, K credit = 1,
X credit = 1, duplicate accessory mass = 0 and distinct-type mass = 2, with
numerical tolerance 1e-8. There must be **exactly two eligible ORF assignments**:
one K and one full-credit X whose PHROG family is outside K and every reserved
core family. A second X, a duplicated K/X, a partial hit, core-reserved ORF,
missing evidence, or merely a high aggregate reward cannot activate the route.
Full credit uses the configured reciprocal-coverage threshold (0.75 for PhiX),
not 100% amino-acid identity. This category requires K retention; K loss alone
is not an add-one-gene event. Existing original ARC-count passers remain eligible.

Do not expand ARC's raw count bounds. In particular, **11/13 is not a whitelist**.
The original 11-feature ARC inventory and count decision are preserved even when
this independently evidenced branch admits a design that fails those counts.
Core order/copies, safety and AAI cannot be bypassed by either branch.

`syntenic-diversity-decisions.csv` contains all candidates reaching this decision,
with independent `syntenic_diversity_arc_count_pass`,
`syntenic_diversity_add_one_gene_pass`, union `syntenic_diversity_pass`, and category
`original_arc_count`, `add_one_gene`, `both`, or `fail`. Evidence columns identify
K/X ORFs, families, credits and accessory copy/type masses. Accepted metrics retain
these names; original ARC annotations remain separately prefixed with `arc_`.
`summary.json` reports original-route, added-route, overlap, add-only and union
counts. The waterfall records only the union stage, so overlapping alternatives
cannot be mistaken for sequential gates. Native evidence and its fingerprints
are retained under `accessory-evidence/`. Legacy v1/v2 do not invoke this route.

The correction is justified by physical circular coordinates and reference translation,
not by acceptance yield. ARC's 39-aa origin-prefix call duplicates the tail of an existing
spanning A call. Removing it makes peptide inventories stable across tested circular cuts.
Both methods recover the other nine reference products exactly, but share an upstream
A-start prediction (522 aa versus annotated 513 aa). Neither explicitly models the genuine
341-aa A\* alternative product; A\* is deliberately excluded from this comparison. The cleanup
removed the 39-aa tail, not an originally included A\* call. These shared limitations must
remain visible; neither score is claimed to reconstruct all expressed products exactly.

Circular cleanup requires physical-locus evidence, not peptide similarity alone: a
forward prefix call must map inside a longer boundary-spanning call in the same
frame and terminate at the same stop after adding one genome length. An identical
peptide tail at another locus or on another strand is retained. The same check
applies when staging the cleaned reference; it does not alter the original ARC
synteny/count reference. Reverse-strand start/stop reconciliation is not supplied
by this forward-prefix cleanup. Alternative starts remain annotation uncertainty;
retain the current maximal-ORF convention. On NC_001422.1, the caller starts at
3954 (522 aa), whereas RefSeq/UniProt annotate 3981 (513 aa). Both starts are in
frame, share the same stop, and differ by nine N-terminal residues without an
intervening stop. Annotation supports the shorter form but does not establish
that the longer form is never expressed. Per the agreed policy, leave A extraction
and scoring unchanged and report this start-site ambiguity; do not trim A to the
reference length or select a start by the resulting AAI. This is separate from
the A\* omission and circular-prefix duplicate cleanup.

The prior published-data comparison was close numerical reproduction, not exact annotation,
database or software parity. Original-inventory AAI remains a diagnostic sensitivity analysis.

Outputs under `arc/arc_reference_filters/` include per-gate CSVs, original natural-member search
hits, annotated GFFs, per-genome LoVis cluster memberships, accepted observations and `summary.json`.
The input/configuration fingerprint binds both natural-member AAI and tropism-protein
database files (including sidecars), not merely their configured paths. An in-place
database change must invalidate reuse; retain prior results and use a new output root.
Core and ARC synteny
fields have separate names. Missing cluster output or incomplete reference/candidate annotation
fails the run. Only after every per-genome gate does final nucleotide clustering select designs.
Always report passing observations and cluster representatives with their denominators separately;
compute representative-level score summaries on the selected IDs, never all cluster members.

Historical reports produced without this contract used the adapted core-conservation endpoint.
Their counts must not be relabeled as passing original ARC synteny. Computational acceptance is
not evidence of experimental viability.

The final rollout JSON includes `accepted_score_summary`: SFT likelihood statistics computed only on accepted cluster representatives, with one vote per cluster. Raw per-observation likelihoods remain available as provenance.

The concise target/diagnostic report embeds the completed `final_filter_contract` summary, including alternative-route counts; final rollout JSON retains it under its target-profile evidence. A missing, mismatched-contract or count-inconsistent final summary fails reporting. V3 counts must be nonnegative integers: overlap cannot exceed either route, add-only equals added minus overlap, and union equals original plus added minus overlap and the terminal FASTA count. The 302 ARC ordered designs are a selected re-screening cohort, not a generation-yield denominator. ARC can disable count diversification for another selection route; the two new recent K+X passers also meet original-inventory AAI (93.11% and 93.33%). Do not claim v3 is strictly harder than ARC or that these candidates were ineligible for ARC selection.

## Independent regression evidence

CPU tests under `tests/bionemo/evo2_phage_gen/` exercise the real final-filter entry
point against pinned upstream function excerpts, published Data S1 sequences/scores
and saved raw alignments. See `fixtures/README.md` for provenance and limits.
The archived 302-design search matched 299 novelty decisions and 291 hit counts;
three threshold disagreements are explicitly retained, not hidden by tolerance.
Saved-hit aggregation parity does not establish fresh ORF/search/database parity.
Keep scientific changes (physical ORF inventory, K+X acceptance) separate from
output-preserving API/performance changes, and report original versus extended
acceptance separately. New scientific rules require a matched re-screen in a new
output root before replacing published rollout counts.
