---
name: bionemo-phage-design-implement-rl-objectives
description: Use when adding or changing Evo2 phage RL metrics, reward functions, filter logic, or validation criteria after an objective plan has been approved.
metadata:
  author: NVIDIA <bionemofeedback@nvidia.com>
---

# Implement Phage RL Objectives

Work inside the selected recipe checkout and follow its existing code and test patterns. Implement only the approved scientific behavior, keeping online rewards and final QC definitions aligned where they are intended to match.

Use the [reward API and module map](references/reward-api.md) for public scoring functions, measurement boundaries, and optional adaptation utilities.

Keep rewards bounded on `[0, 1]`, with the objective plan's meaningful baseline at zero and target at one. Handle short genomes, absent genes or ORFs, empty tool output, missing observations, invalid values, and tool failures explicitly. Empty or malformed genomes receive explicit zero credit across every applicable objective. A tool-safe failure of one acceptance gate, such as length, must not suppress independent ORF, protein, or synteny measurements: record the failed gate separately and combine them only for final acceptance. In mixed valid and invalid batches, isolate invalid rows so valid rows still score; initialize numeric reward columns with stable numeric dtypes, distinguish a successful no-hit from measurement failure, and preserve record mapping under row reordering. Check batch-context dependencies explicitly: the current reference search uses batch-dependent E-values, as documented in the [reward API](references/reward-api.md). Do not let an exception, default value, or skipped branch create positive credit.

Audit GDPO/GRPO per-objective normalization, not only raw reward scales: in a batch of 48, a 47-versus-1 near-constant split produces `|z| ≈ 6.8` whether the raw gap is `0.001` or `1.0`, versus about `2.1` for a well-spread column. The current estimator keeps this normalization unchanged, including small positive differences; do not automatically floor sigma or collapse saturated columns. Diagnose numerical noise separately from meaningful movement and change the estimator only under an approved objective plan. Scalar `weight_*` settings default to equal weights for the shipped enabled components, but do not reweight GDPO: every listed objective has coefficient 1 after within-prompt normalization. See the [configuration contract](../../configs/README.md#reward-weights-and-gdpo-groups).

Preserve formulas, controls, record mapping, and verified scoring concurrency when refactoring or optimizing a scorer; compare the reference and optimized paths on the same cases.

The recipe's genome-length reward uses four finite ordered points: lower zero, lower full credit, upper full credit, and upper zero. Keep these separate from `genome_length_min` / `genome_length_max`, which control hard nucleotide acceptance and valid-only clustering. There is no min/max reward fallback; use the [length configuration reference](../../configs/README.md#length-and-termination) for config fields and FASTA-scoring options.

For the shared smooth protein scorer, use the four-factor geometric-mean definition in the [reward API](references/reward-api.md); its identity baseline, significance, and coverage have distinct meanings. Its small reference-panel search bypasses MMseqs prefiltering to retain alignments that heuristic candidate detection can miss; benchmark cost before adapting it to larger panels. For protein-match objectives, retain E-value, identity, alignment length, query length, target length, and native query/target coverage. Do not infer coverage from alignment-column count, which can include gaps. Distinguish a homologous fragment, an intact gene, and a distinct function. Give credible partial matches graded credit without requiring a complete gene set first; keep final completeness criteria separate. Retain all search-admitted PHROG hits for global one-to-one ORF/function assignment, before collapsing to the strongest hit for displayed annotations. A stronger unrelated-family match must not erase partial required-function evidence. Alternative families for one function fill one slot using their maximum coverage credit; do not sum them or let one ORF fill multiple slots. PHROGs search supplies evidence without a generic family-count reward or acceptance quota. Test embedded or overlapping genes that an ORF caller may miss.

Use gene-deletion and truncation controls to check incremental recovery of gene content. Base-shuffled controls test accidental match credit; gene-order-shuffled controls preserve the protein sequences and test order separately from content. Neither defines a universal reward floor, and an order penalty is appropriate only when the design calls for conserved architecture.

Required-gene objectives use named functions with explicit, reviewed PHROG alternatives via `required_gene_families`; annotation labels cannot substitute for a function. Use the [required-function API](references/reward-api.md#required-functions) and [biological profile](../../configs/required_genes.md) for the mean-coverage formula, calibrated targets, and experimental scope.

For function-aware synteny, use the same family evidence and `synteny_reference_functions` mapping described in the [synteny API](references/reward-api.md#function-aware-synteny). Supported family matches bypass the direct-reference 90% identity target while retaining order and copy checks; tropism and origin keep their own criteria. Preserve short natural J alternatives, including their ORF-calling and coverage calibration, when adapting the PhiX profile. Broader natural-homolog and domain/profile evidence can justify other detection alternatives without a published successful swap; document that this establishes plausible function, not genome compatibility. Do not infer essentiality solely from a reference annotation or presence in viable controls.

Use the [synteny entry points](references/reward-api.md#synteny-entry-points) for content/order/copy scoring. Arc’s separate `genetic_architecture` codon-landmark score has no current RL objective; its similarly named visualization stage is still needed for protein measurements. Keep these roles distinct.

The PHROGs annotation search has its own [configured sensitivity and CPU allocation](references/reward-api.md), shared by online function evidence and final screening. Evaluate search sensitivity separately from coverage/assignment rules, and compare absolute stage cost with end-to-end timing before judging a runtime ratio.

For PhiX gene-A origin shaping, use the [origin formula](references/reward-api.md#gene-a-origin): the four-factor geometric mean of A-protein integrity, motif credit above the 25% uniform-DNA baseline, position/frame eligibility, and strong-site uniqueness. Partial motif credit does not require a strong site. Check partial-match progression and shuffled-window background as well as viable endpoints; the baseline is a shaping choice, not a significance threshold.

Reference conservation, divergence from a database, and within-batch diversity are separate goals. Include novelty pressure only when the approved objective plan calls for it, with a database, direction, and threshold suited to that task. For the PhiX reproduction's specific AAI definition and asset setup, use the [PhiX protein-scoring reference](../../examples/README.md#protein-evidence-synteny-and-diversity); its novelty cutoff is not a general viability requirement.

Pool online genome diversity across all prompts for the same design goal in each scoring batch. A duplicate produced from a different prefix still shares its cluster and inverse-size credit. Use separate scoring calls or RL environments for different design goals; do not split a shared goal by prompt text or GPU rank. This pool is separate from GDPO's normalization by identical prompt tokens. See the [diversity API](references/reward-api.md#batch-diversity) for eligibility, supplied sequence order, and batch scope. The PhiX defaults share coordinate 1; approximate clustering is not declared rotation-invariant.

For synteny objectives, use maximum-weight one-to-one matching against a fixed configured slot denominator (mapped functions for the PhiX profile, callable reference loci for a reference-only profile). Score circular order from the same graded edges, and penalize excess homolog mass so deletion or duplication cannot repair the score. Remove pseudocircular annotation artifacts from the denominator rather than making the reference fail its own gate.

For every new or modified function, run biological positive and negative controls—not only arithmetic unit tests. Use known or evidence-supported expected high- and low-scoring genomes or targeted counterfactuals, record their provenance and expected ordering, and exercise them through the exact installed scorer and combined portfolio.

For learned or stacked scorers, construct upstream features from deployment-matched out-of-fold predictions. Test ranking and calibration in the high-reward tail, scorer-training versus policy-candidate shift, and model/seed disagreement; treat unsupported candidates as missing or out of domain rather than extrapolating confidently.

Add focused tests that demonstrate:

- each enabled component runs and is logged for a known positive control;
- reference, random/baseline, desired, boundary, missing/empty/no-hit, too-short, missing-gene, and tool-failure behavior as applicable;
- the complete enabled portfolio runs on the reference, equivalent circular representations where relevant, targeted negatives, and mixed-validity batches;
- reordered mixed-validity batches preserve valid-row scores, zero invalid rows, and stable numeric dtypes;
- combined objectives do not hide a skipped or fixed-zero component and preserve intended ordering;
- online scoring and final QC agree where promised;
- the installed runtime accepts the expected names, shapes, dtypes, devices, and reductions; and
- a tiny deterministic RL smoke computes rewards, writes a checkpoint, and can resume.

Before declaring reward work complete, create or refresh `artifacts/RL_SCORE_DEFINITIONS.md` in the selected result root. Reconcile it against the implemented reward columns, formulas, resolved configuration, and focused boundary/failure tests. For every enabled objective, state the measured quantity and units, direction, exact formula and settings, zero-credit and full-credit regions or categorical states, both-side partial-credit behavior when applicable, missing/invalid/empty/no-hit/missing-gene/tool-failure behavior, biological rationale and citations, controls and telemetry, and relationship to final hard QC. This is an agent-produced implementation artifact, not a required stage of the fully scripted run. Use the **Current PhiX174 GDPO score definitions** section in `examples/README.md` as a worked format, not as target-independent scientific defaults.

Use the execution skill for a real installed-environment smoke test when local imports are not representative. Run the relevant focused tests after implementation and record the command, settings, results, and any scientific limitation in the stage summary and `RUNLOG.md`.
