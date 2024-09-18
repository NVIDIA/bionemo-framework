# The Protein Data Bank ([PDB](https://www.rcsb.org/))
The Protein Data Bank is a public repository for experimental and computational chemists to publish results of protein-protein, protein-ligand, and isolated compound structure results. The database boasts over 213,221 experimentally derived structures, and 1,068,577 Computed Structure Models (CSM). Various API and toolings have been created to directly download from the server in [batch processes](https://www.rcsb.org/docs/programmatic-access/file-download-services), and some advanced commands for extracting specific groups of PDB IDs.

Given its size, diversity in chemistry, and experimental validity, pruning techniques are often used to select what are seen as the optimal samples for the model problem at hand. Derived from the original PDB depositions, separate databases have been curated and maintained with supplemental information, or fully extracted for future academic use.

## PoseBusters Benchmark {cite:p}`buttenschoen2023posebusters`
The PoseBusters Benchmark set is a new set of 308 carefully-selected publicly-available crystal complexes from the PDB. It is a diverse set of recent high-quality protein-ligand complexes which contain drug-like molecules. It only contains complexes released since 2021 and therefore does not contain any complexes present in the PDBbind General Set v2020 used to train many of the methods.

## The Docking Benchmark 5.5 (DB5.5) {cite:p}`vreven2015updates`
Docking Benchmark 5.5 ([DB5.5](https://zlab.umassmed.edu/benchmark/)) {cite:p}`vreven2015updates` is an updated and integrated version of a widely utilized protein–protein docking and binding affinity benchmark. This benchmark comprises non-redundant, high-quality structures of protein–protein complexes along with the unbound structures of their components. The update includes the addition of 55 new protein–protein complexes, of which 35 have experimentally measured binding affinities, resulting in a total of 230 entries for the docking benchmark and 179 entries for the affinity benchmark.

The composition of DB5.5 includes 55 new cases added to the docking benchmark, representing a 31% increase over the previous version, and 35 of these cases include experimental affinities that contribute to a 24% increase in the affinity benchmark. The inclusion of cases with multiple binding modes further enriches the benchmark.

The benchmark construction process involves collecting new structures from the Protein Data Bank (PDB) using a semi-automatic pipeline. The inclusion criteria ensure high-quality complexes, and sequence-based pruning is applied to prevent cross-contamination between training and test sets. The DB5.5 dataset is considered a gold standard for protein interface prediction, providing a diverse set of examples, including enzyme-inhibitor and antibody-antigen interactions.

## Database of Interacting Protein Structures (DIPS) {cite:p}`townshend2019end`
Database of Interacting Protein Structures ([DIPS](https://github.com/drorlab/DIPS)) {cite:p}`townshend2019end` is constructed to address limitations in existing structural biology tasks, particularly in predicting protein interactions. The dataset is two orders of magnitude larger than previous datasets, aiming to explore whether performance can be enhanced by utilizing large repositories of tangentially related structural data.

DIPS is built by mining the Protein Data Bank (PDB) for pairs of interacting proteins, yielding a dataset of 42,826 binary complexes. To ensure data quality, complexes are selected based on specific criteria, including a buried surface area of ≥ 500 Å2, solved using X-ray crystallography or cryo-electron microscopy at better than 3.5 Å resolution, containing protein chains longer than 50 amino acids, and being the first model in a structure.

Sequence-based pruning is applied to prevent cross-contamination between the DIPS and Docking Benchmark 5 (DB5) datasets. Any complex with individual proteins having over 30% sequence identity with any protein in DB5 is excluded. This pruning process, along with sequence-level exclusion, results in a dataset over two orders of magnitude larger than DB5.

## Continuous Automated Model EvaluatiOn (CAMEO) {cite:p}`robin2023automated`

Continuous Automated Model EvaluatiOn ([CAMEO](https://cameo3d.org/)) {cite:p}`robin2023automated` is a community-led project for continuous quality assessment for biomolecular structure prediction, including monomeric and multimeric proteins, nucleic acids and small molecule ligands. It operates by issuing weekly challenges in predicting protein structures soon to be released publicly at Protein Data Bank. Structure prediction from participating servers are evaluated in metrics such as Local Distance Difference Test (lDDT). CAMEO is considered a more frequent evaluation compared to the gold standard - the biannual Critical Assessment of protein Structure Prediction (CASP).

## Derivative datasets in OpenFold training {cite:p}`Ahdritz2022.11.20.517210`

OpenFold is trained on protein structures released before 2021-09-16 in PDB dataset as the training set and CAMEO structures from 2021-09-17 to 2021-12-11 as the validation set in the initial training phase. In the fine-tuning phase, a self-distillation dataset composes of structure predictions from initial-training model on Unclust30 sequences further augments the training set and help stabilize the learning.
