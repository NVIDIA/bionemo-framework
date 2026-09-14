# Required genes in the PhiX profile

The [Arc configuration](arc_genome_design_filtering_local.yaml) defines nine named
functions for a PhiX-like productive-infection cycle. Each function has an explicit
list of allowed PHROG families in `required_gene_families`. These are detection
targets for biologically motivated functions, not a claim that homology proves
function or that every viable microvirus must retain the same genes.

## Biological rationale and detection targets

The family assignments below were checked against the recipe's pinned PHROGs
consensus database and PhiX174 reference. Gene names and experimental roles govern
selection; shared annotation text does not establish interchangeability.

| Function                              | Allowed PHROGs | Experimental basis and scope                                                                                                                                                                                                                                                                                                                            |
| ------------------------------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| A: replication initiation             | 713            | Gene-A mutant work connects A to replicative-form DNA replication and distinguishes additional A-region phenotypes. [Funk & Snover, 1976](https://pmc.ncbi.nlm.nih.gov/articles/PMC515532/)                                                                                                                                                             |
| B: internal scaffolding               | 1473           | Mutant/complementation experiments connect B to coat conformational changes and assembly. [Novak & Fane, 2004](https://doi.org/10.1016/j.jmb.2003.09.050)                                                                                                                                                                                               |
| C: replication/packaging coordination | 1465           | C-mutant infections accumulate empty procapsids and show altered replication intermediates. The database label “terminase” does not capture this role. [Fujisawa & Hayashi, 1977](https://pmc.ncbi.nlm.nih.gov/articles/PMC353851/)                                                                                                                     |
| D: external scaffolding               | 1386           | D participates in organizing assembly intermediates into procapsids; structural and in-vitro assembly work identifies its contacts with coat proteins. [Procapsid assembly experiments](https://pmc.ncbi.nlm.nih.gov/articles/PMC5165185/)                                                                                                              |
| E: host lysis                         | 1472           | E inhibits host MraY. Its PHROGs “endolysin” annotation is not a mechanistic description. The endpoint here includes progeny release, not only intracellular replication. [Zheng et al., 2008](https://pubmed.ncbi.nlm.nih.gov/18791230/)                                                                                                               |
| F: major capsid                       | 514            | Coat-protein experiments and procapsid structures establish F's structural and assembly role. [Procapsid assembly experiments](https://pmc.ncbi.nlm.nih.gov/articles/PMC5165185/)                                                                                                                                                                       |
| G: major spike                        | 1483           | Cloned G complements G amber mutants in otherwise nonpermissive hosts; it does not rescue the tested A/E mutants. [Humayun & Chambers, 1978](https://pmc.ncbi.nlm.nih.gov/articles/PMC411339/)                                                                                                                                                          |
| H: DNA delivery pilot                 | 1471           | Structural, genetic, and biochemical evidence identifies an H tube needed for DNA delivery and infectivity. [Sun et al., 2014](https://pubmed.ncbi.nlm.nih.gov/24336205/)                                                                                                                                                                               |
| J: DNA binding/packaging              | 2354 or 3780   | In-vitro work distinguishes a requirement for packaging infectious phage from DNA synthesis. [In-vitro J study](https://pmc.ncbi.nlm.nih.gov/articles/PMC254803/) The second family detects the 25-aa alternate J in the viable Evo-Φ36 G1 control from [King et al.](https://doi.org/10.1126/science.aec2657), so either family fills the same J slot. |

This profile favors the usual PhiX architecture. It is deliberately narrower than
all possible viable architectures: selected mutants can bypass B's usual scaffolding
requirement through compensatory changes. That exception does not justify treating
B as optional in every background. [Chen et al., 2007](https://doi.org/10.1016/j.jmb.2007.07.064)

K is outside the required set. K-deficient mutants produced viable progeny with
reduced burst size, so omitting this requirement is compatible with a viability
endpoint but does not imply equal fitness. [Tessman et al., 1980](https://pubmed.ncbi.nlm.nih.gov/6445011/)
A\* is also outside this term. Progeny were recovered without detectable A\* in a
mutant study using amber-suppressor hosts; this is conditional experimental evidence,
not a universal dispensability claim. ORFipy's failure to emit A\* separately is an
independent measurement limitation. [Colasanti & Denhardt, 1987](https://pubmed.ncbi.nlm.nih.gov/2960819/)

## Scoring and acceptance

[`summarize_required_gene_evidence`](../src/bionemo/evo2_phage_gen/protein_evidence.py)
consumes admitted PHROGs hits with native query and target coverage. For each allowed
ORF-to-family hit, its function credit is:

```text
credit = min(1, query_coverage / query_min, target_coverage / target_min)
reward = maximum one-to-one assigned credit sum / number of required functions
```

One ORF can fill one function; one function can use one ORF. Multiple allowed
families are alternatives for that function, not additional requirements. A family
cannot be declared under two functions. Extra genes or copies neither increase the
denominator nor replace missing functions. An unrelated gene with the same
annotation label earns no required-function credit. With 12 configured functions,
the denominator is 12; adding three unrelated genes to the nine-function profile
leaves the denominator at nine.

Hard acceptance requires a distinct qualifying ORF for every function, with both
coverage thresholds met. The scorer independently finds the largest all-qualifying
assignment, so a fractional-score tie cannot accidentally lose a possible hard
pass. Missing functions receive zero credit individually and do not erase progress
on other functions. Missing or invalid measurement artifacts receive zero reward
and unavailable status. An empty profile supplies no reward or acceptance.

The historical column name `required_genes_integrity_sum` means **summed normalized
coverage credit** here. It is not the geometric-mean protein-match score used by
smooth synteny, tropism, and origin. E-value and identity affect upstream hit
admission; neither is an extra multiplier in this required-function reward.

The generic query/target thresholds are 0.75/0.75. The configured exceptions retain
observed viable C/B truncations and E extensions: C 0.70/0.47, B 0.75/0.68, and
E 0.58/0.75. These values were calibrated on the viable-design panel, not derived
from gene-essentiality experiments. Native coverage counts residues, whereas
alignment-column counts can include gaps. Passing coverage does not establish
catalytic activity, correct regulation, compatibility, or whole-genome viability.

## Adapting the profile

Choose named functions from deletion, complementation, or other functional evidence
for the intended host and endpoint. Record assay conditions and distinguish normal
fitness from merely producing viable progeny. Build the detection set from diverse natural homologs, not only substitutions found
in one design study. A family need not have a published successful PhiX swap to be
considered: conserved domains, profile-to-profile homology, and its role in naturally
occurring relatives can support the function assignment. PHROGs supplies member
sequences, alignments, and HMM profiles for this purpose. [Terzian et al., 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8341000/)

Keep function recognition separate from whole-genome compatibility. The G4-like J
in Evo-Φ36 illustrates why: it is a natural J homolog, while prior direct transfers
into PhiX failed in other backgrounds. [King et al., preprint](https://www.biorxiv.org/content/10.1101/2025.09.12.675911v1.full)
The current explicit alternatives are a starting profile, not an exhaustive catalogue
of possible replacements. Broadening it requires checking function evidence and
false matches, not restricting it to experimentally demonstrated swaps. Related-phage
cross-complementation can depend on both gene and direction. [Borrias et al., 1976](https://pmc.ncbi.nlm.nih.gov/articles/PMC353451/)

Check the installed search, ORF caller, scorer, and hard gate on known viable genomes
and targeted missing/truncated-gene controls. Recheck database versions and family
IDs together. Positive controls used to select targets or tune coverage are
calibration evidence; retain independent measured failures and viable variants to
test generalization. A broader architecture with a demonstrated bypass needs an
explicitly revised profile rather than an annotation-label fallback.
