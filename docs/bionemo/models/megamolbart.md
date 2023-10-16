# MegaMolBART

## Model Overview

MegaMolBART is a model that understands chemistry and can be used for a variety of cheminformatics applications in drug discovery. The embeddings from its encoder can be used as features for predictive models. Alternatively, the encoder and decoder can be used together to generate novel molecules by sampling the model's embedding space.

## Intended Use

For each molecule in SMILES format, the model can produce an embedding from the encoder that is suitable for representation learning. It can also sample based on a seed molecule to generate new molecules in SMILES format.

## Model Architecture

MegaMolBART was developed using the BioNeMo framework. The model is a seq2seq transformer called a Bidirectional and Auto-Regressive Transformer (BART) {cite:p}`lewis2019bart`. It is based on a model developed by AstraZeneca called Chemformer {cite:p}`irwin2022chemformer`. MegaMolBART has eight layers, four attention heads, a hidden space dimension of 256, and contains 45M parameters. The maximum sequence length supported by MegaMolBART is 512 tokens. Pre-norm layer normalization and GELU activation are used throughout.

## Limitations

MegaMolBART supports SMILES of length 512 tokens for input and generation.

## Training

### Dataset

The ZINC-15 database was used for training {cite:p}`sterling2015zinc`. Approximately 1.54 Billion molecules (SMILES strings) were selected from [tranches](https://zinc15.docking.org/tranches/home/) meeting the following constraints: molecular weight <= 500 Daltons, LogP <= 5, reactivity level was "reactive," and purchasability was "annotated." The compounds were filtered to ensure a maximum length of 512 characters. Train, validation, and test splits were randomly split as 99% / 0.5% / 0.5%.

### Infrastructure and Configuration

MegaMolBART was trained with data parallelism on 64 V100 32 GB GPUs (4 nodes x 16 GPUs) for 8 epochs (approximately 160k iterations or ~80 wall-clock hours) using a batch size of 32 molecules per GPU (micro batch). The Noam scheduler was used, with a peak learning rate value of 0.0005 and ~8000 warmup steps. FusedAdam optimization was used with parameters β1 = 0.9 and β2 = 0.999. Categorical cross-entropy loss was used to train the model. Dropout was set to 0.1 during training.

## Suggested Reading

Learn more about MegaMolBART [here](https://resources.nvidia.com/en-us-drug-discovery/gtcfall21-a31106).

## License

MegaMolBART is provided under the {{model_license_slug}}.