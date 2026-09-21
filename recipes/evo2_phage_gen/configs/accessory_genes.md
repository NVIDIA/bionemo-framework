# Accessory-gene diversification

`accessory_gene_diversification` is an RL reward and a diagnostic, **not a final
acceptance filter**. It favors K loss or one or two supported accessory types
outside the K and core-gene families. It does not replace
required-gene completeness, `core_gene_ordered_conservation`, safety, or AAI.

The implementation is in
[`accessory_genes.py`](../src/bionemo/evo2_phage_gen/accessory_genes.py).
The shipped GRPO scalar weight is 1; GDPO lists it as a separate equally weighted
objective. The usual safety eligibility and configured EOD gating also apply.
The standalone environment default is off: enable
`external_qc.enable_accessory_gene_diversification` and set
`weight_accessory_gene_diversification`, or include its reward column in GDPO.
Profile settings live in the Arc configuration so replay uses the same inputs;
they do not enable an additional Arc filter.

## Biological scope and evidence

The nine required A/B/C/D/E/F/G/H/J functions retain their existing definitions.
K is optional for this design goal: K-deficient PhiX mutants can remain viable,
with reduced yield under the studied conditions. This is not evidence that K
loss is cost-free or universally beneficial
([Tessman et al.](https://pubmed.ncbi.nlm.nih.gov/6445011/)).

The PhiX/G4-like K selector is PHROG1713. Distant K-like families remain distinct
accessory types: PHROG8511 can earn X credit alongside PHROG1713, rather than
being collapsed into a duplicate merely because both are annotated K. Such a
combination might have a distinct effect; this score does not predict burst size.
In the retained native-annotation panel, G4 and three related K proteins match
1713; alpha3 and ten related K proteins match 8511. Native target coverage is 1.0
and query coverage is 0.855–1.0. These are homolog-detection controls, not evidence
of compatibility in PhiX. Representative records are
[G4 NC_001420.2](https://www.ncbi.nlm.nih.gov/nuccore/NC_001420.2) and
[alpha3 NC_001330.1](https://www.ncbi.nlm.nih.gov/nuccore/NC_001330.1).
A\* is not a separate accessory type; A-related fragments are reserved as core evidence.

Use all search-admitted PHROGs hits, with native query and target coverage.
The standard PHROGs search retains its existing E-value admission and sensitivity
7.5. For an accessory match, normalized completeness is
`q = min(1, qcov / 0.75, tcov / 0.75)`. Thus 30% reciprocal coverage gives 0.4
credit, not 0.3. Percent identity is not multiplied into this coverage credit.
Full credit means reaching these coverage targets, not proving an intact function.

Before choosing accessory hits:

- Reserve every ORF with positive evidence for an allowed core family or direct
  core-reference match. This includes partial assignments, alternative assignments,
  and unselected core copies. The direct route uses the same match formula and
  configured thresholds as ordered conservation.
- Reserve by ORF ID, not by genomic overlap: genuine overlapping genes remain
  separate candidates.
- Preserve best-family K evidence before those exclusions. A K-like ORF reserved as core
  cannot create apparent K loss or earn X credit.
- Each ORF takes its lowest-E-value PHROG hit, with a deterministic family-ID tie
  break. One ORF cannot count as several accessory types; weaker alternate labels
  cannot be selected just to disguise repeated copies. A weak secondary K hit
  does not turn a stronger distinct-family match into a K duplicate.

A different family with a similar biological role can be an extra if it is
outside the approved core families and has no positive direct core-reference
evidence. It does not fill the missing core slot. Unmatched/spurious ORFs receive
no accessory credit. Unknown-function PHROGs establish repertoire evidence, not
biological benefit.

## K and X score surface

`k` is the strongest K completeness credit, including reserved K evidence.
`x` is the strongest eligible completeness credit for a family outside K and the
nine core genes. K variants and copies or partial matches of core genes cannot
contribute X credit. A distinct family can qualify even if its annotated role is K-like.
It is not a sum: splitting one useful addition into several fragments cannot
manufacture a complete X.

There is no sequence-divergence bonus within K. A full PHROG1713 match remains K
regardless of substitutions relative to PhiX or its source organism. With no X,
it scores 0.50 before copy penalties. Replacing K with an eligible PHROG8511
protein instead supplies X; keeping both families supplies K+X. Both earn 1.0
when their coverage credits are full and there are no excess copies.

With no qualifying X:

```text
base = 0.75 - 0.25*k
```

With qualifying X, let `d = min(k, 1-k)`:

```text
base = (1-x)*(0.75 - 0.25*d) + x*(1.00 - 0.40*d)
```

| K credit |  X = 0 | X = 0.10 | X = 0.50 | X = 1.00 |
| -------- | -----: | -------: | -------: | -------: |
| 0.00     | 0.7500 |   0.7750 |   0.8750 |   1.0000 |
| 0.10     | 0.7250 |   0.7485 |   0.8425 |   0.9600 |
| 0.50     | 0.6250 |   0.6425 |   0.7125 |   0.8000 |
| 0.90     | 0.5250 |   0.7485 |   0.8425 |   0.9600 |
| 1.00     | 0.5000 |   0.7750 |   0.8750 |   1.0000 |

With no X, removing K helps linearly. Once X qualifies, the K endpoints are
symmetric and every movement toward either endpoint helps. Increasing X always
helps. The first qualifying X intentionally causes a nonnegative state jump on
the K-retaining side; there is no hidden X plateau or flat K middle. These values
encode the design preference, not measured probabilities of viability.

Zero K means no admitted evidence after a successful search, not proven biological
absence. Missing or invalid artifacts fail measurement; an empty called-protein
cohort receives zero reward and unavailable status rather than deletion credit.

## Copy and repertoire budget

For each eligible type `t`, use its strongest ORF credit `u_t = max(q_i)`:

```text
N = sum(u_t)                  # distinct full-gene-equivalent accessory mass
D = sum(q_i) - N              # extra-copy mass, including partial copies
reward = base / (1 + max(0, N - 2) + D)
```

K counts as an accessory type. K+X and X+Y are within budget. With full matches,
K+X+Y or K+X+X halves the base score; three identical reference-K copies score
0.5/3. A partial third type starts reducing reward once distinct mass exceeds
two; three weak partial types are not treated as three complete genes. Raw
unmatched ORF counts do not enter the budget.

AAI still includes all hit-bearing called proteins, including K and X, and keeps
its existing `min(hit-bearing ORFs / 10, 1)` evidence factor. A borrowed natural K
can retain a 100% best natural-database match. That identity does not determine
whether its family supplies K or X.

## Outputs and migration

`reward_external_accessory_gene_diversification` is the new reward column.
`qc6_accessory_genes_metrics.csv` records K/X credit, distinct mass, duplicate
mass, base score, final score and measurement availability.
`qc6_accessory_genes_assignments.csv` retains selected and excluded ORFs with
their families, coverage and eligibility. RL logs only the compact
K/X/mass diagnostics for this configured objective.

The former `synteny` objective is now `core_gene_ordered_conservation`; its
formula and final gate are unchanged. Migrate `weight_synteny`, `enable_synteny`,
and `reward_external_synteny` to the corresponding new names. Reference/function
settings now use the `core_gene_` prefix. Historical W&B names remain historical;
do not silently join them with the new accessory objective.

This is **accessory diversification v2**. It removes v1's within-K divergence
bonus: historical full-credit K-only designs need rescoring under this definition.
The K/X surface and copy penalties are unchanged, including 0.75 for K loss alone.

This is not a reproduction of Arc's historical
cluster-row filter. That filter did not check order, used annotation-dependent
10–12 counts, and excluded (11,11). Our old shaped comparator assigned full credit
to (10,10), (10,11), (10,12), (11,12), (12,12), and 0.5 to (11,11). Those counts do
not map directly to the nine core functions plus K, particularly after removing
redundant circular-prefix annotations. The current final filters are unchanged
by this new reward; accessory novelty is not an additional acceptance gate.
