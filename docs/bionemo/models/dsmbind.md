# DSMBind

## Description:
DSMBind [1,2] is an energy-based model that has been trained on protein-ligand complexes to predict binding affinities. The model produces comparative values that are useful for ranking protein-ligand binding affinities. This model is for research and development only. <br>

## Third-Party Community Consideration
This model is not owned or developed by NVIDIA. This model has been developed and built to a third-party’s requirements for this application and use case; see [link to the model card by Broad Institute of MIT and Harvard](https://github.com/wengong-jin/DSMBind).

## License
DSMBind is provided under the Apache License 2.0.

## References:
[1] Wengong Jin, Siranush Sarkizova, Xun Chen, Nir Hacohen, and Caroline Uhler. "Unsupervised protein-ligand binding energy prediction via neural euler's rotation equation." Advances in Neural Information Processing Systems 36 (2024). <br>

[2] Wengong Jin, Xun Chen, Amrita Vetticaden, Siranush Sarzikova, Raktima Raychowdhury, Caroline Uhler, and Nir Hacohen. "DSMBind: SE (3) denoising score matching for unsupervised binding energy prediction and nanobody design." bioRxiv (2023): 2023-12. <br>

## Model Architecture:
**Architecture Type:** Energy-Based Model (EBM)  <br>
**Network Architecture:** SE(3)-Invariant Neural Network <br>

## Input:
**Input Type(s):** Text (PDB, SDF) <br>
**Input Format(s):** Protein Data Bank (PDB) Structure files for proteins, Structural Data Files (SDF) for ligands <br>

## Output:
**Output Type(s):** Numerical scores (indicating binding affinities) <br>
**Output Format:** List of scalar values <br>
**Other Properties Related to Output:** Only the rank of the predicted values matters because the model produces comparative values instead of absolute binding energies.<br>

## Software Integration:
**Runtime Engine(s):**
* BioNeMo (1.7), NeMo <br>

**Supported Hardware Microarchitecture Compatibility:** <br>
* Ampere <br>
* Hopper <br>

**Preferred/Supported Operating System(s):** <br>
* Linux <br>

## Model Version(s):
dsmbind.pth, version: 1.7

# Training & Evaluation:

## Training Dataset:
**Link:** a subset from [PDB](https://www.rcsb.org/)  <br>
**Data Collection Method by dataset** <br>
* Human <br>
**Properties (Quantity, Dataset Descriptions, Sensor(s)):** Our DSMBind checkpoint was trained using a subset of PDB. This subset includes a total of 25,561 samples, each representing a unique protein-ligand complex. <br>

## Evaluation Dataset:
**Link:** [CASF-16](http://www.pdbbind.org.cn/casf.php)  <br>
**Data Collection Method by dataset** <br>
* Human <br>
**Labeling Method by dataset** <br>
* Hybrid: Human & Automated <br>
**Properties (Quantity, Dataset Descriptions, Sensor(s)):** CASF-16 is an open challenge for comparative assessment of scoring functions. This benchmark has 285 protein-ligand complexes with binding affinity labels. <br>

## Inference:
**Engine:** BioNeMo, NeMo <br>
**Test Hardware:** <br>
* Ampere <br>

## Evaluation Results
We use gaussian noise to perturbe the ligand coordinates during training. We evaluate our trained DSMBind model on the CASF-16 benchmark. We measure the Pearson correlation coefficient to assess the linear relationship between the predicted scalar values and actual binding affinities. The trained checkpoint can achieve a Pearson correlation coefficient of 0.64.

## Limitations
DSMBind produces comparative values which are useful to rank complexes. But it does not provide absolute measures that are directly comparable to experimental ground truth affinities.
