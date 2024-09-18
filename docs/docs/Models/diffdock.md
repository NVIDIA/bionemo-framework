# DiffDock
# Model Overview

## Description:

DiffDock is a diffusion generative model for drug discovery in molecular blind docking [1]. DiffDock consists of two models: the Score and Confidence models. <br>

The Score model is a 3-dimensional equivariant graph neural network that has three layers: embedding, interaction layer with 6 graph convolution layers, and output layer. In total, the Score model has 20M parameters. The Score model is used to generate a series of potential poses for protein-ligand binding by running the reverse diffusion process. The Confidence model has a similar architecture as the Score model but with 5 graph convolution layers in the interaction layer. In total, the Confidence model has 5M parameters. The Confidence model is used to rank the generated ligand poses from the Score model. These models are ready for commercial use.  <br>

## Third-Party Community Consideration
This donor model was not owned or developed by NVIDIA. This model has been developed and built to a third-party’s requirements for this application and use case; see [link to Non-NVIDIA Model Card](https://github.com/gcorso/DiffDock/tree/main).

## References:
[Provide list of reference(s), link(s) to the publication/paper/article, associated works, and lineage where relevant.]  <br>
[1] Corso, Gabriele, Hannes Stärk, Bowen Jing, Regina Barzilay, and Tommi Jaakkola. "Diffdock: Diffusion steps, twists, and turns for molecular docking." arXiv preprint arXiv:2210.01776 (2022).

## Model Architecture:
**Architecture Type:** Score-Based Diffusion Model (SBDM)  <br>
**Network Architecture:**  Graph Convolution Neural Network (GCNN) <br>

## Input:
**Input Type(s):**  Text (PDB, SDF, MOL2, SMILES) <br>
**Input Format(s):** Protein Data Bank (PDB) Structure files for proteins, Structural Data Files (SDF), Tripos molecule structure format (MOL2) Structure files and Simplified molecular-input line-entry system (SMILES) strings for ligands <br>
**Other Properties Related to Input:** Pre-Processing Needed <br>

## Output:
**Output Type(s):** Text (Ligand Molecules, Confidence Score) <br>
**Output Format:** Structural Data Files (SDF) <br>
**Output Parameters:** Confidence Score and the rank based on this score<br>

## Software Integration:
**Runtime Engine(s):**
* NeMo, BioNeMo <br>

**Supported Hardware Microarchitecture Compatibility:** <br>
* [Ampere] <br>
* [Hopper] <br>

**[Preferred/Supported] Operating System(s):** <br>
* [Linux] <br>
* [Windows] <br>

## Model Version(s):
diffdock_score.nemo, version: 1.5
diffdock_confidence.nemo, version: 1.5

**NOTE**: previous version of checkpoint files, i.e., version 1.1, can be converted to the current version by running the scripts:

```
python ${BIONEMO_HOME}/examples/molecule/diffdock/scripts/convert_nemo_chkpt_cugraph-equiv.py previous_checkpoint.nemo new_checkpoint.nemo --swap_mlp_weight_blocks
```
where `previous_checkpoint.nemo` is the previous checkpoint version 1.1 and
the script outputs the new checkpoint for version 1.5 to
`new_checkpoint.nemo` as specified in the command line.

## Change Log:
* version 1.5: A new version of tensor product convolution layer is implemented using the cugraph-equivariant package, which results in 1.2x speed up in training performance. Because of this new implementation, the checkpoint file should be reformatted. We provide a new checkpoint file for this release and a script that can convert the checkpoint file in the old format to the new format (see section [Model Version(s)](#model-versions) for details).

# Evaluation:
## Evaluation Dataset:
**Link:** [PoseBusters benchmark (PDB) set](https://zenodo.org/records/8278563))  <br>
**Data Collection Method by dataset** <br>
* [Human] <br>

**Labeling Method by dataset** <br>
* [Hybrid: Human & Automated] <br>

**Properties:** 428 protein-ligand complexes manually curated using the PDB database <br>

## Inference:
**Engine:** NeMo <br>
**Test Hardware:** <br>
* Ampere <br>

## Benchmarks

The pretrained DiffDock checkpoints for score and confidence models are available for download: `diffdock_score.nemo` and `diffdock_confidence.nemo`. These have been converted to the NeMo format from publicly available checkpoints. This section provides accuracy benchmarks for these models, as well as information on expected training speed performance. Currently, models trained from randomly initialized weights within the BioNeMo framework are not provided. The production of these models is ongoing work.

### Accuracy Benchmarks

The accuracy of DiffDock was measured over the 428 protein complexes from the PoseBusters benchmark {cite:p}`buttenschoen2023posebusters` available in [zenodo](https://zenodo.org/records/8278563). The metrics was computed from 20 independent runs with the [DiffDock GitHub](https://github.com/gcorso/DiffDock/commit/bc6b5151457ea5304ee69779d92de0fded599a2c) and the DiffDock in this BioNeMo Framework. Due to the inherent stochasticity of DiffDock during the molecular docking generation, the metrics are not expected to be identical. The values in parentheses are standard deviations from 20 runs.

| Dataset     | Number of Poses Sampled | Metric                                  | BioNeMo | GitHub |
|-------------|-------------------------|-----------------------------------------|---------|--------|
| PoseBusters |            10           | Percentage of Top-1 RMSD<2 Å (%) &uarr; | 34.4 (1.4)   | 32.3 (1.0)  |
| PoseBusters |            10           | Median of Top-1 RMSD (Å) &darr;         | 3.36 (0.14)    | 3.61 (0.15)   |
| PoseBusters |            10           | Percentage of Top-5 RMSD<2 Å (%) &uarr; | 41.9 (0.9)  | 39.9 (0.8)  |
| PoseBusters |            10           | Median of Top-5 RMSD (Å) &darr;         | 2.52 (0.07)    | 2.68 (0.08)   |


### Training Performance Benchmarks

Training speed was tested on DGX-A100 systems GPUs with 80GB of memory. Three iterations of performance improvement were made by NVIDIA engineers: 1) DiffDock integrated into BioNeMo FW featuring the size-aware batch sampling enhancement 2) NVIDIA Acceleration of Tensor Product operation in DiffDock 3) Integrated with cugraph-equivariant for fast tensor product graph convolution. [The public As-Received version of DiffDock](https://github.com/gcorso/DiffDock/tree/bc6b5151457ea5304ee69779d92de0fded599a2c) used a dataset with a similar size, while it was trained using 4 A6000 GPUs and converged after 850 epochs for 18 days. As detailed performance metrics are not available, therefore it is not shown in the table below.

|Time   |Label|Speed [it/s]|Epochs to Converge|Epochs/GPU Hour|GPU Hours|Dataset|Batch size|Number of A100 GPUs|
|-------|:-----:|------------|------------------|---------------|---------|-------|----------|:------:|
|2023-09|Size Aware Batch Sampling (first version in BioNeMo FW) |0.41|400|1.1296244|354.1|NV-PDBData|96|8|
|2023-11|Fast Tensor Product Kernel Integration|1.09|400|2.965159377|134.9|NV-PDBData|96|8|
|2024-04|Cugraph-equivariant Integration|1.32|400|3.613369467|110.7|NV-PDBData|96|8|
|2024-06|Training with webdataset|1.57|400|4.293949006|93.2|NV-PDBData|96|8|

## Limitations
DiffDock is currently restricted to static snapshot understanding of single ligand and protein interactions. For more involved systems included multi-ligands in a single protein pocket, multiple protein pockets without a ligand blocker, DiffDock inference may perform poorly due to the unaware implications of ligand-ligand interactions in solvent.

DiffDock is also restricted to rigid protein-ligand docking. For the systems have structural or conformation changes during docking, DiffDock inference may generate not ideal results.

Because ESM2 is used as a featurizer, some non-standard amino acids are ignored in the data preprocessing.


## License
DiffDock is provided under the {{model_license_slug}}.
