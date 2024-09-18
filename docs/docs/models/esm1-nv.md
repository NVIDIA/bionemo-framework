# ESM-1nv
# Model Overview

## Description:

ESM-1nv is a protein language model that provides numerical embeddings for each amino acid in a protein sequence. It is developed using the BioNeMo Framework. The model uses an architecture called Bidirectional Encoder Representations from Transformers (BERT) and is based on the ESM-1 model [1]. Pre-norm layer normalization and GELU activation are used throughout. The model has six layers, 12 attention heads, a hidden space dimension of 768, and contains 44M parameters. The embeddings from its encoder can be used as features for predictive models. This model is ready for commercial use. <br>


## References:

[1] Rives, Alexander, Joshua Meier, Tom Sercu, Siddharth Goyal, Zeming Lin, Jason Liu, Demi Guo et al. "Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences." Proceedings of the National Academy of Sciences 118, no. 15 (2021): e2016239118.

[2] "UniProt: the universal protein knowledgebase in 2021." Nucleic acids research 49, no. D1 (2021): D480-D489.

## Model Architecture:
**Architecture Type:** BERT <br>
**Network Architecture:** ESM-1 <br>

## Input: (Enter "None" As Needed)
**Input Type(s):** Text (Protein Sequences) <br>
**Input Parameters:** 1D <br>
**Other Properties Related to Input:** Protein sequence represented as a string of canonical amino acids, of maximum length 512. Longer sequences are automatically truncated to this length. <br>

## Output: (Enter "None" As Needed)
**Output Type(s):** Text (Protein Sequences) <br>
**Output Parameters:** 1D <br>
**Other Properties Related to Output:** Numeric vector with one float-point value corresponding to each amino acid in the input protein sequence. Maximum output length is 512 embeddings - one embedding vector per amino acid. <br>

## Software Integration:
**Runtime Engine(s):**
* BioNeMo, NeMo 1.2 <br>

**Supported Hardware Microarchitecture Compatibility:** <br>
* [Ampere] <br>
* [Hopper] <br>
* [Volta] <br>

**[Preferred/Supported] Operating System(s):** <br>
* [Linux] <br>

## Model Version(s):
esm1nv.nemo, version 1.0  <br>

# Training & Evaluation:

## Training Dataset:

**Link:**  [UniRef50](https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50) <br>

** Data Collection Method by dataset <br>
* [Human] <br>

**Properties (Quantity, Dataset Descriptions, Sensor(s)):** UniRef50 (release 05/2022) was used for training [2]. The reference sequence for each cluster was selected, with sequences longer than the maximum sequence length of 512 removed, resulting in approximately 46M protein sequences with maximum length of 512 amino acids. The sequences were randomly split with 4.35K sequences for validation loss calculation during training, 875K sequences in testing, and 45.1M sequences used exclusively in training . <br>

## Evaluation Dataset:
**Link:** [FLIP – secondary structure, conservation,subcellular localization, meltome, GB1 activity](http://data.bioembeddings.com/public/FLIP/fasta/)  <br>
** Data Collection Method by dataset <br>
* [Human] <br>
* [Automatic/Sensors] <br>

** Labeling Method by dataset <br>
* [Experimentally Measured] <br>
* [Hybrid: Human & Automated] <br>

**Properties (Quantity, Dataset Descriptions, Sensor(s)):**
The FLIP datasets evaluate the performance of the model on five specific downstream tasks for proteins. It provides pre-defined splits for fine-tuning a pretrained model using task-specific train and validation examples, and subsequently evaluating it on a task-specific test split.

The secondary structure FLIP dataset contains experimental secondary structures, with 9712 proteins for model finetuning, 1080 proteins for validation, and 648 proteins for testing.

The Conservation dataset contains conservation scores of the residues of protein sequences with 9392 proteins for training, 555 proteins for validation, and 519 proteins for testing.

The Subcellular localization dataset contains protein subcellular locations with 9503 proteins for training, 1678 proteins for validation, and 2768 proteins for testing.

The Meltome dataset contains experimental melting temperatures for proteins, with 22335 proteins for training, 2482 proteins for validation, and 3134 proteins for testing.

The GB1 activity dataset contains experimental binding affinities of GB1 protein variants with variation at four sites (V39, D40, G41 and V54) measured in a binding assay, with 6289 proteins for training, 699 proteins for validation, and 1745 proteins for testing. <br>

License Data:
**Dataset License(s):** [AFL-3](https://opensource.org/license/afl-3-0-php/) <br>

## Inference:
**Engine:** BioNeMo, NeMo <br>
**Test Hardware:** <br>
* [Ampere] <br>
* [Hopper] <br>
* [Volta]  <br>

## Ethical Considerations:
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  When downloaded or used in accordance with our terms of service, developers should work with their team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.  For more detailed information on ethical considerations for this model, please see the Model Card++ Explainability, Bias, Safety & Security, and Privacy Subcards [Insert Link to Model Card++ here].  Please report security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).
