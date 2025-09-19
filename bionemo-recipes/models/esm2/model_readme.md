---
library_name: transformers
license: mit
widget:
  - text: MQIFVKTLTGKTITLEVEPS<mask>TIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG
---

## ESM-2 (TransformerEngine-optimized)

This version of the ESM-2 model is optimized with NVIDIA's
[TransformerEngine](https://github.com/NVIDIA/TransformerEngine) library. It is based on the
[original ESM-2 model](https://huggingface.co/facebook/esm2_t48_15B_UR50D) from Facebook Research,
and (within numerical precision) has identical weights and outputs.

ESM-2 is a state-of-the-art protein model trained on a masked language modelling objective. It is
suitable for fine-tuning on a wide range of tasks that take protein sequences as input. For detailed
information on the model architecture and training data, please refer to the [accompanying
paper](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v2). You may also be interested in
some demo notebooks
([PyTorch](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/protein_language_modeling.ipynb),
[TensorFlow](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/protein_language_modeling-tf.ipynb))
which demonstrate how to fine-tune ESM-2 models on your tasks of interest.

Several ESM-2 checkpoints are available in the Hub with varying sizes. Larger sizes generally have
somewhat better accuracy, but require much more memory and time to train:

| Checkpoint name                                                          | Num layers | Num parameters |
| ------------------------------------------------------------------------ | ---------- | -------------- |
| [esm2_t48_15B_UR50D](https://huggingface.co/nvidia/esm2_t48_15B_UR50D)   | 48         | 15B            |
| [esm2_t36_3B_UR50D](https://huggingface.co/nvidia/esm2_t36_3B_UR50D)     | 36         | 3B             |
| [esm2_t33_650M_UR50D](https://huggingface.co/nvidia/esm2_t33_650M_UR50D) | 33         | 650M           |
| [esm2_t30_150M_UR50D](https://huggingface.co/nvidia/esm2_t30_150M_UR50D) | 30         | 150M           |
| [esm2_t12_35M_UR50D](https://huggingface.co/nvidia/esm2_t12_35M_UR50D)   | 12         | 35M            |
| [esm2_t6_8M_UR50D](https://huggingface.co/nvidia/esm2_t6_8M_UR50D)       | 6          | 8M             |
