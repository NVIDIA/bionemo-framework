# ESM-1nv

## Model Overview

ESM-1nv is a model that has been trained on protein sequences. The embeddings from its encoder can be used as features for predictive models.

## Intended Use

Compute embeddings from input protein sequences. Embeddings are created for each amino acid in the protein sequence. Embeddings can then be used for downstream tasks such as prediction of secondary structure, subcellular localization, or others, as detailed by the FLIP benchmark tasks {cite:p}`dallago2021flip`.

## Model Architecture

ESM-1nv was developed using the BioNeMo framework. The model uses an architecture called Bidirectional Encoder Representations from Transformers (BERT) and is based on the ESM-1 model {cite:p}`rives2021esm,devlin2018bert`. Pre-norm layer normalization and GELU activation are used throughout. The model has six layers, 12 attention heads, a hidden space dimension of 768, and contains 44M parameters.

## Limitations

Input sequence length is limited to 512 amino acids.

## Training

### Dataset and Processing

UniRef50 (release 05/2022) was used for training {cite:p}`uniprot2021`. The reference sequence for each cluster was selected, resulting in approximately 52M protein sequences. The sequences were randomly split with 5K sequences in validation, 1M sequences in test, and the remaining in train. Truncation of protein sequences longer than 1024 amino acids and data masking was performed as described previously {cite:p}`devlin2018bert`. The input tokens were randomly masked at a rate of 15% with the masked tokens being predicted. During training by minimizing a cross-entropy categorical loss in order to predict the masked tokens {cite:p}`devlin2018bert`.

### Infrastructure and Configuration

ESM-1nv was trained with data parallelism on 176 A100 GPUs for 420 epochs (approximately 349500 iterations) using a micro batch size of 370 sequences per GPU. Cosine annealing was used, with a minimum learning rate of 2.0e-05, 500 warmup steps, and 50000 constant steps. Fused Adam optimization was used with parameters β1 = 0.9, β2 = 0.98, and weight decay = 0.01. Dropout was set to 0.1 during training. The model training was then continued on 144 A100 GPUs for an additional 600 epochs, resulting in a total of 957610 iterations. The weights of the last 47 checkpoints were averaged to produce the final model.

## Suggested Reading

Learn more about ESM-1nv [here](https://developer.nvidia.com/blog/predict-protein-structures-and-properties-with-biomolecular-large-language-models-2/).

## License

ESM-1nv is as provided under the {{model_license_slug}}.