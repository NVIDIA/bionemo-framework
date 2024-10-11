# ESM-2
# Model Overview

## Description:

ESM-2 is a protein language model that provides numerical embeddings for each amino acid in a protein sequence. It is developed using the BioNeMo Framework. The embeddings from its encoder can be used as features for predictive models. The ESM-2 3B model has 36 layers, 40 attention heads, a hidden space dimension of 2560, and contains 3B parameters. The 650M model has 33 layers, 20 attention heads, a hidden space dimension of 1280, and contains 650M parameters. These models are ready for commercial use. <br>

## Third-Party Community Consideration
This model is not owned or developed by NVIDIA. This model has been developed and built to a third-party’s requirements for this application and use case [1]; see link to [Non-NVIDIA Model Card for ESM-2 3B model](https://huggingface.co/facebook/esm2_t36_3B_UR50D) and [non-NVIDIA Model Card for ESM-2 650M model](https://huggingface.co/facebook/esm2_t36_650M_UR50D)


## References:
[1] Lin, Z., Akin, H., Rao, R., Hie, B., Zhu, Z., Lu, W., Smetanin, N., Verkuil, R., Kabeli, O., Shmueli, Y. and dos Santos Costa, A., 2023. Evolutionary-scale prediction of atomic-level protein structure with a language model. Science, 379(6637), pp.1123-1130.

[2] "UniProt: the universal protein knowledgebase in 2021." Nucleic acids research 49, no. D1 (2021): D480-D489.

[3] Devlin, J., Chang, M.W., Lee, K. and Toutanova, K., 2018. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.
<br>

## Model Architecture:
**Architecture Type:** BERT <br>
**Network Architecture:** ESM-2 <br>

## Input:
**Input Type(s):** Text (Protein Sequences) <br>
**Input Parameters:** 1D <br>
**Other Properties Related to Input:** Protein sequence represented as a string of canonical amino acids, of maximum length 1022. Longer sequences are automatically truncated to this length. <br>

## Output:
**Output Type(s):** Text (Protein Sequences) <br>
**Output Parameters:** 1D <br>
**Other Properties Related to Output:** Numeric vector with one float-point value corresponding to an embedding for each amino acid in the input protein sequence. Maximum output length is 1022 embeddings - one embedding vector per amino acid. <br>

## Software Integration:
**Runtime Engine(s):**
* BioNeMo, NeMo, Megatron, TransformerEngine <br>

**Supported Hardware Microarchitecture Compatibility:** <br>
* [Ampere] <br>
* [Hopper] <br>
* [Volta] <br>

**[Preferred/Supported] Operating System(s):** <br>
* [Linux] <br>

## Model Version(s):
esm2/650m:2.0, esm2/3b:2.0  <br>

# Training & Evaluation:

## Training Dataset:

**Link:**  [UniRef50](https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50) <br>
[UniRef90](https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref90) <br>

**Data Collection Method by dataset** <br>
* [Human] <br>

**Properties:** UniRef50 (release 04/2021) was used for training [2]. The representative sequence for each UniRef50 cluster was selected, resulting in 49,874,565 protein sequences. The sequences were randomly split with 249,372 sequences in validation and 49,625,193 in training. All training sequences that matched a validation sequence with 50% sequence identity were removed from the train set, resulting in 49,425,807 train sequences. A sampling dataset of UniRef90 sequences was created based on any UniRef90 representatives and cluster members that had complete sequences available from UniRef90 or UniRef100, and filtered to UniRef90 sequences for clusters that corresponded to the UniRef50 train set. The UniRef90 dataset was combined with the filtered UniRef50 training dataset to create the sampling fasta file. A mapping file was created to enable rapid replacement of UniRef50 sequences with a sequence sampled uniformly from the corresponding records in the sampling fasta file during each training update. The UniRef50 training fasta was sorted in the order of occurrence of records in column 1 of the mapping file. The UniRef90+UniRef50 sampling fasta file was sorted in the order of occurrence of records in column 2 of the mapping file. Protein sequences longer than 1024 amino acids were cropped to 1022 from sequence start [3]. <br>

## Inference:
**Engine:** BioNeMo, NeMo <br>
**Test Hardware:** <br>
* [Ampere] <br>
* [Hopper] <br>
* [Volta]  <br>

## License

ESM-2 is as provided under the Apache 2.0 license.
