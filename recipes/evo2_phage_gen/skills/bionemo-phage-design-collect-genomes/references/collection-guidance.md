# Collection guidance

Define eligibility before retrieval and keep full genomes in the project result directory. Preserve source accession/version, retrieval query and date, license, host evidence, verified termini class/topology and its evidence, and exclusion reason. For collapsed DTRs, keep the deposited repeat unit, recovered DTR/origin, and inferred packaged genome distinct.

Validate FASTA structure, nucleotide alphabet, nonempty sequences, record counts, lengths, and ambiguous bases. Keep source duplicates visible until definitive biological deduplication and cluster splitting.

Report source, parsed, eligible, and distinct biological-sequence counts, plus similarity-cluster support, total bases/tokens, target coverage, and feasible held-out sets. Use these observations to decide whether more collection, transfer learning, or a changed model plan is needed.

## Start-site variability

For biologically rotation-compatible genomes, align related complete assemblies to the target with circular and strand handling. Map each deposited coordinate 1 to a homologous reference position; report the start-position/orientation distribution by relatedness and independent sequence cluster, retaining uncertain mappings. Exact short-prefix counts are a useful initial check but miss divergent homologous starts and cannot establish the full distribution.

Inspect submission methods, annotation pipelines, and primary records for conventions such as placing a conserved gene first, reusing reference coordinates, or standardizing strand. Distinguish documented conventions from an inferred concentration of starts; neither alone establishes physical termini or a replication origin. Count accession duplicates and circular equivalents separately so repeated deposits do not exaggerate start-site diversity or support.

Compare the source assemblies with the selected model's processed SFT sequences and actual training split. Record whether preprocessing preserves, standardizes, or augments starts, and whether equivalent sequences are merely grouped to prevent split leakage. A model seeing a locus internally does not establish that it learned to start and terminate a whole genome there. Canonical starts dominating related records can make arbitrary circular prompting hard to generalize even when rotation is biologically valid.

Pass this evidence to sampling calibration. Favor represented starts when support is concentrated; treat alternate rotations as a separate hypothesis requiring paired evidence from the selected checkpoint before deployment. The [PhiX prompt-origin note](../../../examples/README.md#prompt-origin) gives the current corpus counts, their limitations, and the origin-only default.
