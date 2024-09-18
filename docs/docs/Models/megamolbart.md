# MegaMolBART
# Model Overview

## Description:
MegaMolBART molecular sequence, based upon known molecular sequences,is an autoencoder trained on small molecules in the form of SMILES that can be used for molecular representation tasks, molecule generation, and retrosynthesis. It was developed using the BioNeMo framework. MegaMolBART has eight layers, four attention heads, a hidden space dimension of 256, and contains 45M parameters. This model is ready for commercial/non-commercial use. <br>

## References:
Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov, and Luke Zettlemoyer. BART: denoising Sequence-to-Sequence pre-training for natural language generation, translation, and comprehension. October 2019. *arXiv*:1910.13461.

Ross Irwin, Spyridon Dimitriadis, Jiazhen He, and Esben Jannik Bjerrum. Chemformer: a pre-trained transformer for computational chemistry. *Mach. Learn.: Sci. Technol.*, 3(1):015022, January 2022. doi:10.1088/2632-2153/ac3ffb.

Teague Sterling and John J Irwin. ZINC 15–ligand discovery for everyone. *J. Chem. Inf. Model.*, 55(11):2324–2337, November 2015. doi:10.1021/acs.jcim.5b00559. <br>

## Model Architecture:
**Architecture Type:** Transformer, Sequence-to-sequence <br>
**Network Architecture:** BART <br>

Pre-norm layer normalization and GELU activation are used throughout.

## Input:
**Input Type(s):** SMILES Text (Molecular Sequence) <br>
**Input Format(s):** Comma Separated Values, Simplified Molecular-Input Line Entry System (SMILES)  <br>
**Input Parameters:** 1D <br>
**Other Properties Related to Input:** Maximum input length is 512 tokens. Pre-training dataset samples were randomly split into train, validation, test sets ( 99% / 0.5% / 0.5% ). <br>

## Output:
**Output Type(s):** [Text, Numerical.] <br>
**Output Format:** [SMILES] <br>
**Output Parameters:** [2D] <br>
**Other Properties Related to Output:** [Maximum output length is 512 tokens] <br>

## Software Integration:
**Runtime Engine(s):**
* BioNeMo, NeMo 1.2 <br>

**Supported Hardware Microarchitecture Compatibility:** <br>
* [Ampere] <br>
* [Hopper] <br>

**[Preferred/Supported] Operating System(s):** <br>
* [Linux] <br>

## Model Version(s):
Megamolbart.nemo, version 23.06  <br>

# Training & Evaluation:

## Training Dataset:

**Link:** [ZINC-15](https://zinc15.docking.org) <br>

**Data Collection Method by dataset** <br>
* [Human] <br>

**Labeling Method by dataset** <br>
* [Hybrid: Human & Automated] <br>

**Properties:** 1.54B molecules with molecular weight <= 500 Daltons, LogP <= 5, with reactivity levels rated as  “reactive” and purchasability “annotated.” The compounds were filtered to ensure a maximum length of 512 characters. The final dataset contained 1.54 Billion molecules. <br>

## Evaluation Dataset:
**Link:** [MoleculeNet - Lipophilicity, FreeSolv, ESOL](https://moleculenet.org/datasets-1) <br>

**Data Collection Method by dataset** <br>
* [Human] <br>
* [Automatic/Sensors] <br>

**Labeling Method by dataset** <br>
* [Hybrid: Human & Automated] <br>

**Properties:** Contains 4,200 experimentally measured octanol/water distribution coefficients (logD at pH 7.4)randomly split for evaluation;  642 experimental and calculated hydration free energy of small molecules in water randomly split for evaluation; and1,128 experimentally measured water solubility data (log solubility in mols per liter) for common organic small molecules randomly split for evaluation. <br>

## Inference:
**Engine:** NeMo <br>
**Test Hardware:** <br>
* Ampere <br>

MegaMolBART was trained with data parallelism on 64 V100 32 GB GPUs (4 nodes x 16 GPUs) using a micro batch size of 32. The Noam scheduler was used, with a peak learning rate value of 0.0005 and ~8000 warm up steps. FusedAdam optimization was used with parameters β1 = 0.9 and β2 = 0.999. Categorical cross-entropy loss was used to train the model. Dropout was set to 0.1 during training.

## Ethical Considerations:
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  When downloaded or used in accordance with our terms of service, developers should work with their team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.  For more detailed information on ethical considerations for this model, please see the Model Card++ Explainability, Bias, Safety & Security, and Privacy Subcards [Insert Link to Model Card++ here].  Please report security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).

## License:
MegaMolBART is provided under the {{model_license_slug}}.
