# ProtT5nv

## Model Overview

ProtT5nv is a model that has been trained on protein sequences. Its encoder output can be used for predictive models, while sequence translation tasks can utilize the entire encoder-decoder architecture.

## Intended Use 

For each protein sequence, the model can produce an embedding from the encoder that is suitable for representation learning. For sequence translation tasks, both the encoder and decoder are utilized.

## Model Architecture

ProtT5nv was developed using the BioNeMo framework starting from a model pre-trained on NLP data. The model uses an architecture called T5 and is based on the original ProtT5 model {cite:p}`elnaggar2020prottrans,raffel2021t5`. The model has 12 layers, 12 attention heads, a hidden space dimension of 768, and contains 192M parameters. The maximum sequence length supported by ProtT5 is 512 tokens. Pre-norm layer normalization and GELU activation are used throughout.

## Limitations

ProtT5nv has a maximum sequence length of 512 for both the encoder and the decoder. Proteins whose amino acid sequence is longer than this are truncated at 512 amino acids.

## Training

### Dataset and Processing

UniRef50 (release 05/2022) was used for training {cite:p}`uniprot2021`. The reference sequence for each cluster was selected, with sequences longer than the maximum sequence length of 512 removed, resulting in approximately 46M protein sequences. The sequences were randomly split with 4.35K sequences in validation, 875K sequences in test, and the remaining in train. Data masking was performed as described previously {cite:p}`raffel2021t5`.

### Infrastructure and Configuration

Using the T5 model trained on NLP data, the model was then further pre-trained with protein sequences using data parallelism on 224 V100 GPUs for 58 epochs (approximately 970189 iterations) using a micro batch size of 12 molecules per GPU. The total training time was approximately 120 wall-clock hours. Inverse square root annealing was used, with a minimum learning rate of 0.0 and ~10000 warmup steps. Fused Adam optimization was used with parameters β1=0.9 β2=0.999 and weight decay=0.01. Categorical cross-entropy loss was used to train the model. Dropout was set to 0.1 during training.

## Suggested Reading

Learn more about ProtT5 [here](https://syncedreview.com/2020/07/15/prottrans-delivers-sota-pretrained-models-for-proteins/).

## License

ProtT5nv is provided under the {{model_license_slug}}.