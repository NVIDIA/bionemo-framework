# Model Benchmarks

## Protein Sequence Representation

Metrics and datasets are from [FLIP](https://www.biorxiv.org/content/10.1101/2021.11.09.467890v2.full).

| Metric         |                                |                                                                                                                                                                                                                                                                             | Dataset     |                                                                                                                                                                                                                                                                                                                     |
|----------------|--------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Type           | Name                           | Definition                                                                                                                                                                                                                                                                  | Split Name  | Split Definition                                                                                                                                                                                                                                                                                                      |
| Classification | Secondary Structure            | Predict one of three secondary structure classes (helix, sheet, coil) for each amino acid in a protein sequence.                                                                                                                                                            | Sampled     | Randomly split sequences into train/test with 95/5% probability.                                                                                                                                                                                                                                                    |
| Classification | Subcellular Localization (SCL) | For each protein, predict one of ten subcellular locations (cytoplasm, nucleus, cell membrane, mitochondrion, endoplasmic reticulum, lysosome/vacuole, golgi apparatus, peroxisome, extracellular, and plastid).                                                            | Mixed Soft  | The mixed soft split uses train, validation, and test splits as provided in the DeepLoc 1.0 publication.                                                                                                                                                                                                            |
| Classification | Conservation                   | Predict one of nine possible conservation classes (1 = most variable to 9 = highly conserved) for each amino acid in a protein sequence                                                                                                                                     | Sampled     | Randomly split sequences into train/test with 95/5% probability.                                                                                                                                                                                                                                                    |
| Regression     | Meltome                        | Predict melting degree, which is the temperature at which >50% of a protein is denatured.                                                                                                                                                                                   | Mixed Split | Protein sequences were clustered by seq identity with 80% of clusters used for training, 20% for testing. The mixed split uses sequences from clusters for training and the representative cluster sequence for testing. The objective is to minimize performance overestimation on large clusters in the test set. |
| Regression     | GB1 Binding Activity           | The impact of amino acid substitutions for one or more of four GB1 positions (V39, D40, G41, and V54) was measured in a binding assay. Values > 1 indicate more binding than wildtype, equal to 1 indicate equivalent binding, and < 1 indicate less binding than wildtype. | Two vs Rest | The training split includes wild type sequence and all single and double mutations. Everything else is put into the test set.

**Classification Metric Values**

ESM models listed below are tested as deployed in BioNeMo.

| Secondary Structure |              | Subcellular Localization (SCL) |              | Conservation        |              |
|---------------------|--------------|--------------------------------|--------------|---------------------|--------------|
| **Model**           | **Accuracy** | **Model**                      | **Accuracy** | **Model**           | **Accuracy** |
| One Hot             | 0.643        | One Hot                        | 0.386        | One Hot             | 0.202        |
| ESM1nv              | 0.773        | ESM1nv                         | 0.720        | ESM1nv              | 0.249        |
| ProtT5nv            | 0.793        | ProtBERT                       | 0.740        | ProtT5nv            | 0.256        |
| ProtBERT            | 0.818        | ProtT5nv                       | 0.764        | ProtBERT            | 0.326        |
| ProtT5              | 0.854        | ESM2 T33 650M UR50D            | 0.791        | ESM2 T33 650M UR50D | 0.329        |
| ESM2 T33 650M UR50D | 0.855        | ESM2 T36 3B UR50D              | 0.812        | ESM2 T36 3B UR50D   | 0.337        |
| ESM2 T36 3B UR50D   | 0.861        | ProtT5                         | 0.820        | ESM2 T48 15B UR50D  | 0.340        |
| ESM2 T48 15B UR50D  | 0.867        | ESM2 T48 15B UR50D             | 0.839        | ProtT5              | 0.343        |

**Regression Metric Values**

| Meltome             |        | GB1 Binding Activity |        |
|---------------------|--------|----------------------|--------|
| **Model**           |**MSE** | **Model**            | **MSE**|
| One Hot             | 128.21 | One Hot              | 2.56   |
| ESM1nv              | 82.85  | ProtT5               | 1.69   |
| ProtT5nv            | 77.39  | ESM2 T33 650M UR50D  | 1.67   |
| ProtBERT            | 58.87  | ESM2 T36 3B UR50D    | 1.64   |
| ESM2 T33 650M UR50D | 53.38  | ProtBERT             | 1.61   |
| ESM2 T36 3B UR50D   | 45.78  | ProtT5nv             | 1.60   |
| ProtT5              | 44.76  | ESM1nv               | 1.58   |
| ESM2 T48 15B UR50D  | 39.49  | ESM2 T48 15B UR50D   | 1.52   |

## SMILES Representation

**Metric Definitions and Dataset**

| Type                | Metric        | Metric Definition                                                                                                                        | Dataset                                                                                                                                                                                                                                                                                         |
|---------------------|---------------|------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Physchem Properties | Lipophilicity | MSE from best performing SVM and Random Forest model, as determined by hyperparameter optimization with 20-fold nested cross-validation. | MoleculeNet datasets: Lipophilicity: 4,200 molecules FreeSolv: 642 molecules ESOL: 1,128 molecules                                                                                                                                                                                              |
|                     | FreeSolv      |                                                                                                                                          |                                                                                                                                                                                                                                                                                                 |
|                     | ESOL          |                                                                                                                                          |                                                                                                                                                                                                                                                                                                 |
| Bioactivities       | Activity      |                                                                                                                                          | ExCAPE database filtered on a subset of protein targets (28 genes). The set of ligands for each target comprise one dataset, with the number of ligands ranging from 1,341 to 367,067 molecules (total = 1,203,479). A model is fit for each dataset and the resulting MSE values are averaged. |

**Metric Values**

| Type                | Metric        | SVM MSE | Random Forest MSE |
|---------------------|---------------|---------|-------------------|
| Physchem Properties | Lipophilicity | 0.491   | 0.811             |
|                     | FreeSolv      | 1.991   | 4.832             |
|                     | ESOL          | 0.474   | 0.862             |
| Bioactivities       | Activity      | 0.520   | 0.616             |


## SMILES Generation

**Metric Definitions and Dataset**

| Type          | Metric       | Metric Definition                                                                                                                                                                                                                        | Dataset                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
|---------------|--------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Sampling      | Validity     | Percentage of molecules generated which are valid SMILES, as determined by RDKit.                                                                                                                                                        | The dataset was 10k molecules randomly selected from ChEMBL that are not present in the training data for MoFlow or MegaMolBART and pass drug-likeness filters. For each of these seed molecules, sample 512 molecules from MoFlow with a temperature of 0.25. For MegaMolBART, sample 10 molecules with a radius of 1.0. For each seed molecule, calculate metric or properties as described on its samples. The metric value is the percentage of molecules which meet the metric definition. |
|               | Novelty      | Percentage of valid molecules that are not present in training data and don’t match the seed molecule.                                                                                                                                   |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
|               | Uniqueness   | Percentage of valid molecules that are unique.                                                                                                                                                                                           |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
|               | NUV          | Percentage of molecules generated which meet all sampling metrics (novelty, uniqueness, validity).                                                                                                                                       |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| Drug-Likeness | QED          | Quantitative estimate of drug-likeness.                                                                                                                                                                                                  |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
|               | SAS          | Synthetic accessibility score.                                                                                                                                                                                                           |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
|               | Pass Filters | Fraction of valid molecules which meet all of the following drug-likeness criteria: (1) SAS between 2.0 and 4.0, inclusive; (2) QED >= 0.65; (3) Maximum ring size <= 6; (4) Number of rings >= 2; (5) No rings with fewer than 5 atoms. |

**Metric Values**

| Type          | Metric       | MegaMolBART |                    | MoFlow |                    |
|---------------|--------------|-------------|--------------------|--------|--------------------|
|               |              | Mean        | Standard Deviation | Mean   | Standard Deviation |
| Sampling      | Validity     | 0.819       | 0.034              | 1.000  | 0.000              |
|               | Novelty      | 1.000       | 0.000              | 1.000  | 0.000              |
|               | Uniqueness   | 0.513       | 0.069              | 0.841  | 0.190              |
|               | NUV          | 0.395       | 0.037              | 0.841  | 0.190              |
| Drug-Likeness | QED          | 0.746       | 0.007              | 0.583  | 0.009              |
|               | SAS          | 2.654       | 0.204              | 4.150  | 0.254              |
|               | Pass Filters | 0.766       | 0.074              | 0.215  | 0.020              |
