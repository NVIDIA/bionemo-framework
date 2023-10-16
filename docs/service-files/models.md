<style>
.prev-next-area {
    font-size: x-large;
}
</style>
# Models

## Model Descriptions

### AlphaFold-2

#### Model Overview

AlphaFold2 is a deep learning model for protein structure prediction developed by the research group at DeepMind, an artificial intelligence (AI) research lab owned by Google {cite:p}`jumper2021alphafold`. AlphaFold2 builds on the success of its predecessor, AlphaFold, and represents a significant breakthrough in the field of protein structure prediction.

#### Intended Use

The intended use of AlphaFold2 is to predict the three-dimensional structure of a protein from its amino acid sequence. This is an important problem in structural biology, as the 3D structure of a protein is closely related to its function. By predicting the structure of a protein, researchers can gain insights into its function and potentially develop new drugs and treatments for diseases.

The accuracy and speed of AlphaFold2 make it well-suited for a range of applications in structural biology and drug discovery. For example, it could be used to predict the structures of proteins that have not been experimentally characterized or to screen large libraries of potential drug molecules for those that are most likely to interact with a target protein.

#### Model Architecture

AlphaFold2 is an attention-base model consisting of a Template Pair stack (2 blocks), an Extra MSA stack (4 blocks), an Evoformer stack (48 blocks), and the Structure Module (eight blocks) {cite:p}`jumper2021alphafold`.

Both the Extra MSA stack and Evoformer stack consist of two tracks: MSA representation and Pair representation, with self-attention layers for each of the tracks and cross-attention that allow exchanging information between the two tracks. MSA representation is processed with gated row-wise and column-wise self-attention layers, while the pair representation with triangular self-attention layers.

If optional templates are provided as input, the Template Pair stack utilizes triangular self-attention to process them and produces additional input for the Evoformer stack (pair representation). The final portion of the network, the Structure Module, consists of a single track processed with Invariant Point Attention layers.

#### Limitations

Input sequence length is limited to 3000 amino acids.

#### Training

Training information is not provided.

#### Suggested Reading

Learn more about AlphaFold-2 [here](https://www.deepmind.com/publications/applying-and-improving-alphafold-at-casp14) and about BioNeMo service [here](https://www.nvidia.com/en-us/gpu-cloud/bionemo/).

#### License

AlphaFold2 is provided under the Apache License.

### DiffDock

#### Model Overview

DiffDock is a state-of-the-art generative model for blind molecular docking pose estimation {cite:p}`corso2023diffdock`. It requires protein and molecule 3D structures as input and does not require any information about a binding pocket. During its diffusion process, the position of the molecule relative to the protein, its orientation, and the torsion angles are allowed to change. By running the learned reverse diffusion process, it transforms a distribution of noisy prior molecule poses to the one learned by the model. As a result, it outputs many sampled poses and ranks them via its confidence model.

#### Intended Use

DiffDock is designed to generate multiple docking poses for a given protein and molecule 3D structure. It accepts protein structure in .pdb format and molecule structure in .sdf or .mol2 formats.

#### Model Architecture

DiffDock contains two models:

* Score model: a Denoising Diffusion Probabilistic Model with an equivariant graph convolutional network {cite:p}`thomas2018tensorfield`. The model has 6-graph convolutional layers, and each node has two layer MLPs for weights, with a total of 20.24M parameters. The convolution is based on equivariant tensor algebra, where the maximum order of angular momentum is 2 for edge embedding and is capped at 1 for node attributes. The score model operates on the coarse-grained representation of the protein and uses language model embeddings from ESM2 {cite:p}`geiger2022e3nn`. For the molecule, it excludes the H atoms from the graph. Its output is two equivariant vectors, one for molecule translation and one for rotation, and as many invariant scalars as there are freely rotatable bonds in the molecule.

* Confidence model: Same architecture as the score model, with five layers and 4.77M parameters. Differently from the score model, this model operates on the all atom representation of the protein and outputs a single scalar.

#### Limitations

Because DiffDock uses language model embeddings from the EMS2 language model, it carries the same limitation of a maximum input sequence length of 1024 amino acids.

#### Training

##### Dataset and Processing

DiffDock uses the molecular complexes in PDBBind extracted from the Protein Data Bank (PDB) {cite:p}`liu2017score,berman2003pdb`. The train/validation/test split is based on time-split, as proposed earlier, with 17k complexes from 2018 or earlier for training/validation and 363 test structures from 2019 with no ligand overlap with the training complexes {cite:p}`stark2022equibind`.

#### Infrastructure and Configuration

The checkpoint provided is the one obtained from the [public repository of the authors](https://github.com/gcorso/DiffDock). The authors trained the score model on four 48GB RTX A6000 GPUs for 850 epochs, the confidence model is trained on a single 48GB GPU for 75 epochs. For further details, refer to the DiffDock publication {cite:p}`corso2023diffdock`.

#### Suggested Reading

Learn more about DiffDock [here](https://arxiv.org/abs/2210.01776) and about BioNeMo service [here](https://www.nvidia.com/en-us/gpu-cloud/bionemo/).

#### License

DiffDock is provided under the [MIT License](https://github.com/gcorso/DiffDock/blob/main/LICENSE).

### ESM-1nv

#### Model Overview

ESM-1nv is a model that has been trained on protein sequences. The embeddings from its encoder can be used as features for predictive models.

#### Intended Use

Compute embeddings from input protein sequences. Embeddings are created for each amino acid in the protein sequence. Embeddings can then be used for downstream tasks such as prediction of secondary structure, subcellular localization, or others, as detailed by the FLIP benchmark tasks {cite:p}`dallago2021flip`.

#### Model Architecture

ESM-1nv was developed using the BioNeMo framework. The model uses an architecture called Bidirectional Encoder Representations from Transformers (BERT) and is based on the ESM-1 model {cite:p}`rives2021esm,devlin2018bert`. Pre-norm layer normalization and GELU activation are used throughout. The model has six layers, 12 attention heads, a hidden space dimension of 768, and contains 44M parameters.

#### Limitations

Input sequence length is limited to 512 amino acids.

#### Training

##### Dataset and Processing

UniRef50 (release 05/2022) was used for training {cite:p}`uniprot2021`. The reference sequence for each cluster was selected, resulting in approximately 52M protein sequences. The sequences were randomly split with 5K sequences in validation, 1M sequences in test, and the remaining in train. Truncation of protein sequences longer than 1024 amino acids and data masking was performed as described previously {cite:p}`devlin2018bert`. The input tokens were randomly masked at a rate of 15% with the masked tokens being predicted. During training by minimizing a cross-entropy categorical loss in order to predict the masked tokens {cite:p}`devlin2018bert`.

##### Infrastructure and Configuration

ESM-1nv was trained with data parallelism on 176 A100 GPUs for 420 epochs (approximately 349500 iterations) using a micro batch size of 370 sequences per GPU. Cosine annealing was used, with a minimum learning rate of 2.0e-05, 500 warmup steps, and 50000 constant steps. Fused Adam optimization was used with parameters β1 = 0.9, β2 = 0.98, and weight decay = 0.01. Dropout was set to 0.1 during training. The model training was then continued on 144 A100 GPUs for an additional 600 epochs, resulting in a total of 957610 iterations. The weights of the last 47 checkpoints were averaged to produce the final model.

#### Suggested Reading

Learn more about ESM-1nv [here](https://developer.nvidia.com/blog/predict-protein-structures-and-properties-with-biomolecular-large-language-models-2/) and about BioNeMo service [here](https://www.nvidia.com/en-us/gpu-cloud/bionemo/).

#### License

ESM-1nv is as provided under the {{model_license_slug}}..

### ESM-2

#### Model Overview

ESM2 is a state of the art language model for protein sequences. It outputs a numerical representation of input protein sequences that is suitable for downstream tasks. In particular, its output is used for protein folding in the ESMFold model. ESM2 comes in various sizes: 650M parameters, 3B parameters, and 15B parameters. ESM2 was developed and trained by META {cite:p}`lin2023esmfold`.

#### Intended Use

Compute embeddings from input protein sequences. Embeddings are created for each amino acid in the protein sequence. Embeddings can then be used by downstream tasks like protein folding (as in ESMFold model) or to predict properties such as secondary structure, subcellular localization, or others, as detailed by the FLIP benchmark tasks {cite:p}`devlin2018bert`.

#### Model Architecture

BERT with one modification to support arbitrary sequence length, namely rotational embeddings.

#### Limitations

Input sequence length for ESM2 models can be up to 1024; however, there are some inconsistencies, and an error can still present.

#### Training

The ESM-2 language models are trained with the masked language modeling objective, which trains the model to predict the identity of randomly selected amino acids in a protein sequence by observing their context in the rest of the sequence. During training, sequences are sampled with even weighting across ∼43 million UniRef50 training clusters from ∼138 million UniRef90 sequences so that over the course of training, the model sees about 65 million unique sequences {cite:p}`uniprot2021`.

##### Infrastructure and Configuration

FAIR trained each model over 512 NVIDIA V100 GPUs. ESM2 650M took eight days to train. The 3B parameter LM took 30 days. The 15B model took 60 days. All language models were trained for 500K updates except the 15B language model, which was stopped after 270K updates.

#### Suggested Reading

Learn more about ESM-2 [here](https://esmatlas.com/about) and about BioNeMo service [here](https://www.nvidia.com/en-us/gpu-cloud/bionemo/).

#### License

ESM2 is provided under the [MIT License](https://github.com/facebookresearch/esm/blob/main/LICENSE).

### ESMFold

#### Model Overview

ESMFold is a protein structure prediction deep learning model developed by Facebook AI Research (FAIR) {cite:p}`lin2023esmfold`. The model was inspired by Alphafold, but does not require multiple sequence alignment (MSA) as an input, leading to significantly faster inference times for protein structure prediction that is nearly as accurate as alignment-based methods.

#### Intended Use

Predicts the three-dimensional structure of a protein from only the sequence of amino acids (does not require multiple-sequence alignment).

#### Model Architecture

ESMFold is an attention-based model consisting of a pre-trained embedding model (ESM-2 3B parameter model, which has 36 blocks and is frozen during training), a folding trunk (48 blocks), and the structure module (eight blocks) {cite:p}`ahdritz2022openfold,jumper2021alphafold`. The forward pass of ESMFold first processes an amino acid sequence through the feedforward layers of the language embedding model, and the model’s internal states are passed to the folding head (trunk + structure module). After three steps of recycling, the model outputs an atomic-level structure and predicted confidences.

#### Limitations

Because ESMFold uses language model embeddings from the EMS2 language model, it carries the same limitation of a maximum input sequence length of 1024 amino acids.

#### Training

##### Dataset and Processing

ESMFold was trained on all Protein Data Bank (PDB) chains until 2020-05-01 using the filtering criteria and clustering (MMSeqs2 {cite:p}`steinegger2018mmseqs2`) procedures from AlphaFold {cite:p}`jumper2021alphafold`. The PDB chains were supplemented with 13,477,259 predicted structures from AlphaFold2 {cite:p}`ahdritz2022openfold` and filtered to structures with mean pLDDT > 70. Predicted structures were sampled 75% of the time, and real structures 25% of the time during training.

##### Infrastructure and Configuration
The model was trained on 128 NVIDIA V100 GPUs for over 10 days. The model was trained for 125K steps on protein crops of size 256, then fine-tuned with an additional structural violation loss for 25K steps on crop sizes of 384.

#### Suggested Reading

Learn more about ESMFold [here](https://esmatlas.com/about) and about BioNeMo service [here](https://www.nvidia.com/en-us/gpu-cloud/bionemo/).

#### License

ESMFold is provided under the MIT License.

### MegaMolBART

#### Model Overview

MegaMolBART is a model that understands chemistry and can be used for a variety of cheminformatics applications in drug discovery. The embeddings from its encoder can be used as features for predictive models. Alternatively, the encoder and decoder can be used together to generate novel molecules by sampling the model's embedding space.

#### Intended Use

For each molecule in SMILES format, the model can do one of the following:

* Produce an embedding from the encoder that is suitable for representation learning.

* Sample randomly based on a seed molecule to generate new molecules in SMILES format.

#### Model Architecture

MegaMolBART was developed using the BioNeMo framework. The model is a seq2seq transformer called a Bidirectional and Auto-Regressive Transformer (BART) {cite:p}`lewis2019bart`. It is based on a model developed by AstraZeneca called Chemformer {cite:p}`irwin2022chemformer`. MegaMolBART has eight layers, four attention heads, a hidden space dimension of 256, and contains 45M parameters. The maximum sequence length supported by MegaMolBART is 512 tokens. Pre-norm layer normalization and GELU activation are used throughout.

#### Limitations

MegaMolBART supports SMILES of length 512 tokens for input and generation.

#### Training

##### Dataset

The ZINC-15 database was used for training {cite:p}`sterling2015zinc`. Approximately 1.54 Billion molecules (SMILES strings) were selected from [tranches](https://zinc15.docking.org/tranches/home/) meeting the following constraints: molecular weight <= 500 Daltons, LogP <= 5, reactivity level was "reactive," and purchasability was "annotated." The compounds were filtered to ensure a maximum length of 512 characters. Train, validation, and test splits were randomly split as 99% / 0.5% / 0.5%.

##### Infrastructure and Configuration

MegaMolBART was trained with data parallelism on 64 V100 32 GB GPUs (4 nodes x 16 GPUs) for 8 epochs (approximately 160k iterations or ~80 wall-clock hours) using a batch size of 32 molecules per GPU (micro batch). The Noam scheduler was used, with a peak learning rate value of 0.0005 and ~8000 warmup steps. FusedAdam optimization was used with parameters β1 = 0.9 and β2 = 0.999. Categorical cross-entropy loss was used to train the model. Dropout was set to 0.1 during training.

#### Suggested Reading

Learn more about MegaMolBART [here](https://resources.nvidia.com/en-us-drug-discovery/gtcfall21-a31106) and about BioNeMo service [here](https://www.nvidia.com/en-us/gpu-cloud/bionemo/).

#### License

MegaMolBART is provided under [the Apache License](https://gitlab-partners.nvidia.com/clara-discovery/bionemo/-/blob/v0.3.0_ea1/LICENSE/license.txt)


### MoFlow

#### Model Overview

MoFlow is a model for molecule generation that leverages Normalizing Flows. Normalizing Flows is a class of generative neural networks that directly models the probability density of the data. They consist of a sequence of invertible transformations that convert the input data that follow some hard-to-model distribution into a latent code that follows a normal distribution which can then be easily used for sampling.

#### Intended Use

For a molecule in SMILES format (seed molecule), the model can generate a random sample of new molecules in SMILES format by sampling from the latent space around the point corresponding to the given seed.

#### Model Architecture

The model consists of two types of Normalizing Flows:

* Glow {cite:p}`kingma2018glow` for processing edges (bonds) and Graph Conditional Flow {cite:p}`zang2020moflow` for processing nodes (atoms) in the context of edges. The Glow part of the model consists of 10 normalizing flows, each using ActNorm and Affine Coupling layers. Each coupling layer has two convolutional blocks with 512 channels in the Conv2D layer, followed by batch normalization and ReLU activation. Squeeze operation were used {cite:p}`dinh2016nvp` with a factor of 20.

* For Graph Conditional Flow, 38 normalizing flows were used. Each flow consists of ActNorm and Graph Affine Coupling layers. Each coupling layer consists of a Relational Graph Convolutional {cite:p}`schlichtkrull2018gcn` with a hidden dimension of 256 and two linear layers with 512 and 64 neurons, respectively. Both graph convolutional and linear layers are followed by batch normalization and ReLU activations.

#### Limitations

MoFlow supports SMILES corresponding to molecules with up to 38 heavy atoms for both input and generated samples.

#### Training

##### Dataset and Processing

The ZINC 250k dataset was used for training {cite:p}`gomez-bombarelli2018chemvae`. This dataset consists of approximately 250k small molecules with up to 38 heavy atoms selected from the ZINC database {cite:p}`sterling2015zinc`. The original train-validation split was used with 90% of the data used for training and 10% for validation.

##### Infrastructure and Configuration

MoFlow was trained with data parallelism on 8 A100 80 GB GPUs (single node) for 1200 epochs (approximately 160 wall-clock minutes) using a batch size of 512 molecules per GPU (micro batch). FusedAdam optimization was used with parameters β1 = 0.9 and β2 = 0.99, and learning rate = 0.0005. Negative log-likelihood loss was used to train the model.

#### Suggested Reading

Learn more about MoFlow [here](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/DrugDiscovery/MoFlow/README.md) and about BioNeMo service [here](https://www.nvidia.com/en-us/gpu-cloud/bionemo/).

#### License

MoFlow is provided under the MIT and Apache Licenses.


### OpenFold

#### Model Overview

OpenFold is a DL model designed to predict the three-dimensional structure of a protein from its amino acid sequence. It is a trainable re-implementation of AlphaFold2 and matches AlphaFold2 accuracy. The model was trained using OpenProtein-Set, the largest public database of protein multiple sequence alignments. The neural network takes two types of inputs: Multiple Sequence Alignment (MSA) and templates, obtained by searching sequence and structure databases, respectively.

#### Intended Use

The purpose of OpenFold is to predict the three-dimensional structure of a protein from its amino acid sequence.

#### Model Architecture

OpenFold is an attention-base model consisting of a Template Pair stack (2 blocks), an Extra MSA stack (4 blocks), an Evoformer stack (48 blocks), and the Structure Module (8 blocks) {cite:p}`ahdritz2022openfold,jumper2021alphafold`. Both the Extra MSA stack and Evoformer stack consist of two tracks: MSA representation and Pair representation, with self-attention layers for each of the tracks and cross-attention that allow exchanging information between the two tracks. MSA representation is processed with gated row-wise and column-wise self-attention layers, while the pair representation with triangular self-attention layers.

If optional templates are provided as input, the Template Pair stack utilizes triangular self-attention to process them and produces additional input for the Evoformer stack (pair representation). The final portion of the network, the Structure Module, consists of a single track processed with Invariant Point Attention layers.


#### Limitations

Input sequence length is limited to 3000 amino acids.

#### Training

##### Dataset and Processing

OpenFold was trained using OpenProteinSet {cite:p}`ahdritz2022openfold,openproteinset`. This dataset consists of MSAs and templates for 640k protein structures (132k unique chains) from the PDB database {cite:p}`berman2003pdb` and 4.85M UniClust30 {cite:p}`mirdita2017uniclust` clusters. MSAs were generated with HHBlits {cite:p}`remmert2011hhblits` and JackHMMER {cite:p}`johnson2010jackhmmer` against MGnify {cite:p}`mitchell2020mgnify`, BFD {cite:p}`jumper2021alphafold`, UniRef90 {cite:p}`suzek2015unirefclust`, and UniClust30. Templates were retrieved from PDB70 with HHSearch {cite:p}`steinegger2019hhsuite` for all PDB samples and 270k UniClust30 samples.

Dataset was split into training and validation sets based on the structure deposition date, with CAMEO targets from the last three months of 2021 used for validation and remaining structures released before and the entire UniClust30 subset used for training and fine-tuning.

##### Infrastructure and Configuration

OpenFold was trained with data parallelism on 128 A100 80GB GPUs (16 nodes with eight GPUs each) with micro batch size of 1. The training procedure consists of two phases: pre-training and fine-tuning, with 80k and 12k training steps, respectively. For both phases, MSA resampling and clustering were used.

The pre-training phase was performed on the samples prepared from the PDB database randomly cropped to 256 amino-acids. MSA depth was limited to 128 and 1024 sequences for Evoformer and Extra MSA stacks, respectively, and up to four templates were used per sample. Adam optimizer was used with a base learning rate of 0.001. The learning rate was linearly increased over the first 128k samples (warm-up) and then decreased by a factor of 0.95 after 6.4M samples.

Fine-tuning was performed on the entire training set (PDB and UniClust30 samples) using crops of 384 amino-acids, 512 sequences in Evoforer, and 5120 in the Extra MSA

#### Suggested Reading

Learn more about OpenFold [here](https://openfold.io/) and about BioNeMo service [here](https://www.nvidia.com/en-us/gpu-cloud/bionemo/).

#### License

OpenFold is provided under the [Apache License](https://github.com/aqlaboratory/openfold/blob/main/LICENSE).

### ProtGPT2

#### Model Overview

ProtGPT-2 is a language model trained on the protein sequence space that generates de novo protein sequences following the principles of natural ones {cite:p}`ferruz2022protgpt2`. ProtGPT2 generates protein sequences with amino acid and disorder propensities on par with natural ones while being evolutionarily distant from the current protein space.

#### Intended Use

ProtGPT2 enables the screening for proteins with similarities to natural proteins in order to improve, fine-tune, or alter a specific biochemical function of a natural protein. Large-scale screening of ProtGPT2-designed protein libraries might identify proteins with folds not captured in structural databases and functions that have no related counterpart in the natural space.

#### Model Architecture

The model uses a GPT-2 Large architecture, with 36 layers, a hidden size of 1280, and 20 attention heads. It contains a total of 774M parameters.

#### Limitations

1.  It generates de novo proteins in a zero-shot fashion with a prompt of "<|endoftext|>", a special token that refers to the start and end of a sequence in FASTA format.

    Example:

    "<|endoftext >MGEAMGLTQPAVSRAVARLEERVGIRIFNRTARAITLTDEGRRFYEAVAPLLAGIEMHGY\nRVNVEGVAQLLELYARDILAEGRLVQLLPEWAD<|endoftext|>"

    The zero-shot generation results for other prompts are unexplored. The authors have introduced another model for controlled generations. They also suggested fine tuning ProtGPT-2 on the input sequences of interest to generate sequences that would resemble the properties of the input sequences. The performance or results are not published.

2.  The generation results require postprocessing to keep only sequences with a high chance of being valid. Perplexity measures and some other basic checks are recommended (and implemented in the API) for post-processing that would impact the generation speed.

#### Training

##### Dataset and Processing

ProtGPT2 was trained on the UniRef50 database (version 2021_04), containing 48 million protein sequences {cite:p}`uniprot2021`. The pre-training was done on the raw sequences without FASTA headers. The data were tokenized using BPE and randomly split into train and validation datasets (90%/10%).

##### Infrastructure and Configuration

Model weights were reinitialized prior to training. The model was optimized using Adam (β1 = 0.9, β2 = 0.999) with a learning rate of 1e-03. For the main model, 65,536 tokens were used per batch (128 GPUs × 512 tokens). A micro batch size of 8 was used, resulting in a global batch size of 1024. The model trained on 128 NVIDIA A100s in 4 days. Parallelism of the model was handled with DeepSpeed69.

#### Suggested Reading

Learn more about ProtGPT-2 [here](https://www.uni-bayreuth.de/press-releases/AI-protein-design) and about BioNeMo service [here](https://www.nvidia.com/en-us/gpu-cloud/bionemo/).

#### License

ProtGPT2 is provided under the [MIT License](https://zenodo.org/record/6796843#.ZCBoouzMK74).

## Model Benchmarks

### Embedding

#### Protein Sequences

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

| Secondary Structure |          | Subcellular Localization (SCL) |          | Conservation        |          |
|---------------------|----------|--------------------------------|----------|---------------------|----------|
| Model               | Accuracy | Model                          | Accuracy | Model               | Accuracy |
| One Hot             | 0.643    | One Hot                        | 0.386    | One Hot             | 0.202    |
| ESM1nv              | 0.773    | ESM1nv                         | 0.720    | ESM1nv              | 0.249    |
| ProtT5nv            | 0.793    | ProtBERT                       | 0.740    | ProtT5nv            | 0.256    |
| ProtBERT            | 0.818    | ProtT5nv                       | 0.764    | ProtBERT            | 0.326    |
| ProtT5              | 0.854    | ESM2 T33 650M UR50D            | 0.791    | ESM2 T33 650M UR50D | 0.329    |
| ESM2 T33 650M UR50D | 0.855    | ESM2 T36 3B UR50D              | 0.812    | ESM2 T36 3B UR50D   | 0.337    |
| ESM2 T36 3B UR50D   | 0.861    | ProtT5                         | 0.820    | ESM2 T48 15B UR50D  | 0.340    |
| ESM2 T48 15B UR50D  | 0.867    | ESM2 T48 15B UR50D             | 0.839    | ProtT5              | 0.343    |

**Regression Metric Values**

| Meltome             |        | GB1 Binding Activity |      |
|---------------------|--------|----------------------|------|
| **Model**           |**MSE** | **Model**            | **MSE**|
| One Hot             | 128.21 | One Hot              | 2.56 |
| ESM1nv              | 82.85  | ProtT5               | 1.69 |
| ProtT5nv            | 77.39  | ESM2 T33 650M UR50D  | 1.67 |
| ProtBERT            | 58.87  | ESM2 T36 3B UR50D    | 1.64 |
| ESM2 T33 650M UR50D | 53.38  | ProtBERT             | 1.61 |
| ESM2 T36 3B UR50D   | 45.78  | ProtT5nv             | 1.60 |
| ProtT5              | 44.76  | ESM1nv               | 1.58 |
| ESM2 T48 15B UR50D  | 39.49  | ESM2 T48 15B UR50D   | 1.52 |


#### SMILES

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



### Generation

#### Protein Sequences

**Metric Definitions and Dataset**

| Dataset            | Dataset Definition                                                                                                                                                                                                                                         |   |   | Type                        |
|--------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---|---|-----------------------------|
| U50                | 100.000 sequences sampled from UniRef50 as done in https://doi.org/10.1016/j.csbj.2022.11.014 . Data collected from http://data.bioembeddings.com/public/design/                                                                                           |   |   | Public/Reference            |
| Permuted sequences | The U50 sequences with random shuffling within each sequence, for example: ABCD -> CDAB. This retains the original distribution of amino acid content, while changing the order of the sequence. Data collected from http://data.bioembeddings.com/public/design/ |   |   | Public/Reference            |
| ProtGPT2           | 100.000 sequences generated from the original ProtGPT2 using PGP ( https://github.com/hefeda/PGP ) as done in https://doi.org/10.1016/j.csbj.2022.11.014 . Data collected from http://data.bioembeddings.com/public/design/                                |   |   | Public/Reference Generation |
| ProtGPT2_bionemo   | 100.000 sequences produced by the bionemo hosted ProtGPT2 using the same parameters as in PGP


| Metric                                                   | Metric Definition                                                                                                                                                                         |   |   | Method                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |   |   |
|----------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---|---|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---|---|
| Embeddings of generated sequences predicted as "natural" | A linear classifier is used to predict whether generated sequences are "natural" or "random". The expectation is that the sequences will be predicted "natural" in the majority of cases. |   |   | Sequences are passed through a representation layer, in this case (the original) ProtT5. A subsequent linear model is trained on representations from "natural" sequences (U50, labeled as "1") and "random" sequences (random, labeled as "0"). The classifier is applied to the same datasets (U50 and random) as validation (expectation is that U50 > 95% = 1; random > 95% = 0), and to the representations of generated sequence sets (ProtGPT2 and PRotGPT2_bionemo) to attest whether they are more like random or natural sequence embeddings. |   |   |


**Metric Values**

| Metric                                                                                     | Class   | Randomly Shuffled U50 Sequences (%) | Unmodified U50 (%) | ProtGPT2 Generated Sequences (%) | ProtGPT2 - BioNeMo Generated Sequences (%) |
|--------------------------------------------------------------------------------------------|---------|-------------------------------------|--------------------|----------------------------------|--------------------------------------------|
| Embeddings of generated sequences predicted as "natural" or “Random” by ProtT5 based model | Natural | 1.553124                            | 97.228778          | 99.024                           | 95.468                                     |
|                                                                                            | Random  | 98.446876                           | 2.771222           | 0.976                            | 4.532                                      |



#### SMILES

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

| Type          | Metric       | MegMolBART |                    | MoFlow |                    |
|---------------|--------------|------------|--------------------|--------|--------------------|
|               |              | Mean       | Standard Deviation | Mean   | Standard Deviation |
| Sampling      | Validity     | 0.819      | 0.034              | 1.000  | 0.000              |
|               | Novelty      | 1.000      | 0.000              | 1.000  | 0.000              |
|               | Uniqueness   | 0.513      | 0.069              | 0.841  | 0.190              |
|               | NUV          | 0.395      | 0.037              | 0.841  | 0.190              |
| Drug-Likeness | QED          | 0.746      | 0.007              | 0.583  | 0.009              |
|               | SAS          | 2.654      | 0.204              | 4.150  | 0.254              |
|               | Pass Filters | 0.766      | 0.074              | 0.215  | 0.020              |


### Protein Structure Prediction

| Type              | Metric                | Metric Definition                                                                                                                                                                                                                                                                                                                                                                                                 |   |   | Dataset                                                                                                                                                                                                                       |   |   |
|-------------------|-----------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---|---|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---|---|
| Structure         | lDDT Score            | Local Distance Difference Test score measures the fraction of distances which are preserved based on a reference structure within a certain threshold (0.5, 1, 2, and 4 Å). The higher the better.                                                                                                                                                                                                                |   |   | Protein sequences from [CAMEO DB](https://www.cameo3d.org/modeling/3-months/difficulty/all/?to_date=2022-12-31): last three months up to 2022-12-31. This dataset with this link includes 172 sequences. Removing the protein 7vyx[A], which has length >1024, leaves 171 cases [Spreadsheet of all sequences](https://docs.google.com/spreadsheets/d/18bsGhxYhPZv292Z7cbXS3y-9VucNr0yTrY-o12M-bQ4/edit?usp=share_link) |   |   |
|                   | CAD Score             | Contact area difference (CAD) calculates the residue–residue contact area using Vironoi volumes. The higher the better.                                                                                                                                                                                                                                                                                           |   |   |                                                                                                                                                                                                                               |   |   |
|                   | STRIDE Score          | STRIDE is a software to identify the secondary structure types (alpha-fold, beta-turns, rand-coils …) and assign a code to each residue for the types it is in. We launch STRIDE for both predicted structure and experimental structure, and then calculate the fraction of residues in the predicted structure that has the correct secondary structure code as in the experimental one. The higher the better. |   |   |                                                                                                                                                                                                                               |   |   |
| Steric & Rotamers | MolProbity Score      | MolProbity score combines clash score, Ramachandran outliers, and sidechain rotamer outliers into a single score. Lower indicates better quality.                                                                                                                                                                                                                                                                 |   |   |                                                                                                                                                                                                                               |   |   |
|                   | Clash Score           | Clash score is the number of serious steric overlaps (> 0.4 Å) per 1000 atoms. 100th percentile is the best among structures; The lower the better.                                                                                                                                                                                                                                                               |   |   |                                                                                                                                                                                                                               |   |   |
|                   | Rotamer Outliers      | Number of rotamer outliers in structure. The lower the better.                                                                                                                                                                                                                                                                                                                                                    |   |   |                                                                                                                                                                                                                               |   |   |
|                   | Ramachandran Outliers | Number of ramachandran outliers (phi, psi angles in backbone) in structure. The lower the better.                                                                                                                                                                                                                                                                                                                 |   |   |                                                                                                                                                                                                                               |   |   |
|                   | Bond Outliers         | Number of bond length outliers in structure. The lower the better.                                                                                                                                                                                                                                                                                                                                                |   |   |                                                                                                                                                                                                                               |   |   |
|                   | Angle Outliers        | Number of angle outliers in structure. The lower the better.


### Docking

**Dataset**

The test set consists of 363 protein-ligand complexes, and the files for each complex are located in a separate directory within the PDBBind_processed_testset.tgz file. Each directory contains one protein file in PDB format (.pdb) and two ligand files in SDF and MOL2 format (.sdf and .mol2) that contain the same information. The availability of two file formats allows testing compatibility with different software tools. The PDBBind_processed_testset.tgz file is obtained from the DiffDock publication located in its GitHub repo at: [GitHub](https://github.com/gcorso/DiffDock#dataset).

**Metric Definitions**

| Metric     | Description                                                                                                                                                                     |
|------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Top-1 RMSD | Heavy-atom RMSD between the predicted and the crystal ligand atoms after aligning the protein structures.                                                                       |
| Top-5 RMSD | Heavy-atom RMSD between the predicted and the crystal ligand atoms after aligning the protein structures, where the top five poses are considered.                                 |
| %<2        | Percentage of predictions that have an RMSD less than 2A. A chemically relevant metric since much prior work considers poses with an RMSD less than 2A as "good" or successful. |
| Med        | Median value of the RMSD distribution.

**Metric Values**

|                                              | 10 poses   |      |            |      | Failures | 40 poses   |      |            |      |
|----------------------------------------------|------------|------|------------|------|----------|------------|------|------------|------|
|                                              | Top-1 RMSD |      | Top-5 RMSD |      |          | Top-1 RMSD |      | Top-5 RMSD |      |
|                                              | %<2        | Med  | %<2        | Med  |          | %<2        | Med  | %<2        | Med  |
| [Published](https://arxiv.org/pdf/2210.01776.pdf) | 35         | 3.6  | 40.7       | 2.65 | N/A      | 38.2       | 3.3  | 44.7       | 2.4  |
| DiffDock in BioNeMo                             | 32.51      | 3.81 | 39.67      | 2.68 |          | 38.02      | 3.20 | 44.35      | 2.40 |
