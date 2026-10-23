# ARC validation and comparison limits

Changes intended to preserve ARC output need parity tests against an independent
oracle. Scientific extensions need explicit contracts and adversarial acceptance
tests; agreement with our own reward is not validation.

| Boundary                  | Independent check                                                                                                                                 | What it does not establish                                                    |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| AAI aggregation           | Data S1 sequences/scores, archived raw natural-member hits, and pinned upstream hit-selection/GFF arithmetic; equal-E-value and zero-E-value ties | Fresh MMseqs search or authors' database-byte parity                          |
| Search API extensions     | Original versus patched legacy columns, row/ID order, ties and search arguments; native gap-aware coverage                                        | Binary-version or thread-count numerical equivalence                          |
| GFF/GenBank coordinates   | Actual positive/negative-strand CDS DNA and translation through paired coordinate conversion                                                      | Correct biological initiation sites                                           |
| Circular-prefix cleanup   | Real published genomes, independently enumerated forward ORFs and physical codon coordinates across cuts; distinct loci retained                  | Reverse-strand generalization, A\* annotation or expression evidence          |
| Final v1/v2/v3 acceptance | Real entry point with upstream filters and real core/safety/K+X decisions; literal accepted IDs, gate counts and artifacts                        | Full external ORF/search/LoVis execution—the external boundaries are fixtures |
| Cache/report integrity    | Changed database bytes/sidecars and evidence invalidate reuse; v3 overlap/union arithmetic reconciles with terminal output                        | Proof that a historical run used a database version it did not record         |

The pinned upstream fixture is extracted from ARC commit
`53f195997257c56c00e5ef8d33a54f5baad143a6`. Its source/function hashes and the
publication fixture provenance are documented in the
[fixture notes](../tests/bionemo/evo2_phage_gen/fixtures/README.md).
Tests exercise wrong-best-hit selection, gate bypasses, overlap double-counting,
coordinate half-fixes and missing circular cleanup as negative controls.
High RL reward cannot replace native hard-filter evidence.

## AAI is not a free parameter

AAI means the unweighted average of one lowest-E-value **natural-member** hit per
hit-bearing ORF, retaining the first tie. Profile-consensus identity, maximum
identity, one observation per family, length weighting and extra coverage gates
are different metrics.

Two published designs have tight numerical regressions (absolute tolerance
**1e-8 percentage points**, not a biological tolerance). The fixture also retains
all three known 95%-threshold disagreements from the archived 302-design probe.
That probe matched 299/302 novelty decisions and 291/302 hit counts; it is not
exact publication reproduction. Replaying its 359,400 saved hit rows across 318
workbook records reproduces the archived aggregation, not a new search.

Physical circular-ORF cleanup intentionally changes the inventory relative to
original ARC. V3's structured K+X route intentionally extends count diversification.
Neither is an output-preserving optimization. Report original-route, added-route,
overlap and union separately, and distinguish selected-cohort re-screening from
random-generation yield. Computational acceptance is not experimental viability.

## Running the checks

From the recipe environment with test dependencies installed:

```bash
python -m pytest tests/bionemo/evo2_phage_gen/test_arc*.py \
  tests/bionemo/evo2_phage_gen/test_protein_evidence.py \
  tests/bionemo/evo2_phage_gen/test_accessory_genes.py \
  tests/bionemo/evo2_phage_gen/test_rollout_evidence.py \
  tests/bionemo/evo2_phage_gen/test_external_qc.py \
  tests/bionemo/evo2_phage_gen/test_reference_controls.py \
  tests/bionemo/evo2_phage_gen/test_lovis4u_metrics.py
```

The new regression fixtures run without downloading ARC or a protein database.
Existing full preparation-chain tests additionally require the pinned ARC checkout
and reference assets under `data/external/arc_evo2`; otherwise they report skips.
Do not describe a skipped external test as passed or the fixture suite as a full
10k re-screen.

Acceptance-changing fixes require matched re-screening of comparison cohorts in
new output roots, preserving old results. A reporting-only fix can instead be
checked against saved decision rows and terminal FASTAs. A cache-key fix does not
by itself prove historical results wrong: inspect database provenance and whether
reuse occurred before deciding which stages require recomputation.
