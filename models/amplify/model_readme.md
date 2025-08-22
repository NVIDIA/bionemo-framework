---
library_name: transformers
license: mit
datasets:
  - chandar-lab/UR100P
language:
  - en
tags:
  - biology
---

> [!NOTE]
> This model has been optimized using NVIDIA's [TransformerEngine](https://github.com/NVIDIA/TransformerEngine)
> library. Slight numerical differences may be observed between the original model and the optimized
> model. For instructions on how to install TransformerEngine, please refer to the
> [official documentation](https://github.com/NVIDIA/TransformerEngine?tab=readme-ov-file#installation).
>
> The original xformers-based models are available at [chandar-lab/AMPLIFY](https://huggingface.co/chandar-lab/AMPLIFY_350M).

## AMPLIFY

AMPLIFY is an efficient, state-of-the-art protein language model pre-trained using masked language modeling on UniRef100, OAS, and SCOP ([UR100P](https://huggingface.co/datasets/chandar-lab/UR100P)). AMPLIFY can generate residue and protein embeddings, suggest mutations, differentiate disordered proteins from non-protein sequences, and much more. AMPLIFY is available in two sizes, 120M and 350M parameters, with the `_base` models not extended beyond 512 residues (Stage 1). The model architecture and pre-training procedure are detailed below. For more details, please refer to the [accompanying paper](https://www.biorxiv.org/content/10.1101/2024.09.23.614603v1).

- [`AMPLIFY_350M`](https://huggingface.co/nvidia/AMPLIFY_350M)
- [`AMPLIFY_350M_base`](https://huggingface.co/chandar-lab/AMPLIFY_350M_base)
- [`AMPLIFY_120M`](https://huggingface.co/nvidia/AMPLIFY_120M)
- [`AMPLIFY_120M_base`](https://huggingface.co/chandar-lab/AMPLIFY_120M_base)

### Model Description

|                                | AMPLIFY 120M | AMPLIFY 350M |
| :----------------------------- | -----------: | -----------: |
| `hidden-size`                  |          640 |          960 |
| `num-hidden-layers`            |           24 |           32 |
| `num-attention-heads`          |           10 |           15 |
| `intermediate-size`            |         2560 |         3840 |
| `max-position-embeddings`      |         2048 |         2048 |
| `vocab-size`                   |           27 |           27 |
| `rope-theta`                   |        10000 |        10000 |
| `dropout-prob`                 |            0 |            0 |
| `embedding-init-range`         |         0.02 |         0.02 |
| `norm-eps`                     |      1.0e-05 |      1.0e-05 |
| `hidden-act`                   |       swiglu |       swiglu |
| `pre-activation-layer-norm`    |         true |         true |
| `layer-norm-after-embedding`   |        false |        false |
| `layer-norm-before-last-layer` |         true |         true |
| `rms-norm`                     |         true |         true |
| `ffn-bias`                     |        false |        false |
| `attn-bias`                    |        false |        false |

### Training Description

|                     |     Stage 1 |                      Stage 2 |
| :------------------ | ----------: | ---------------------------: |
| `dataset`           |      UR100P |                       UR100P |
| `max-steps`         |     1000000 | 25000 (120M) or 50000 (350M) |
| `max-length`        |         512 |                         2048 |
| `optimizer`         |       adamw |                        adamw |
| `lr`                |       0.001 |                       0.0001 |
| `betas`             | (0.9, 0.95) |                  (0.9, 0.95) |
| `eps`               |     1.0e-08 |                      1.0e-08 |
| `weight-decay`      |        0.01 |                         0.01 |
| `scheduler`         | cosinedecay |                         none |
| `warmup-steps`      |       1,000 |                         none |
| `final-step`        |     900,000 |                         none |
| `warmup-steps`      |       1,000 |                         none |
| `gradient-clipping` |         1.0 |                          1.0 |
| `tf32`              |        true |                         true |
| `mixed-precision`   |        bf16 |                         bf16 |
| `padding`           |  max-length |                   max-length |
| `random-truncate`   |        true |                         true |
| `mask-probability`  |        0.15 |                         0.15 |
| `total-batch-size`  |        4096 |                         4096 |
| `deepspeed`         |        true |                         true |
| `zero-stage`        |           3 |                            3 |

## Get Started

```python
from transformers import AutoModel
from transformers import AutoTokenizer
from datasets import load_dataset

# Load AMPLIFY and tokenizer
model = AutoModel.from_pretrained("nvidia/AMPLIFY_350M", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("nvidia/AMPLIFY_350M", trust_remote_code=True)

# Move the model to GPU (required due to Flash Attention)
model = model.to("cuda")

# Load the UniProt validation set
dataset = load_dataset("chandar-lab/UR100P", data_dir="UniProt", split="test")

for sample in dataset:
    # Protein
    print("Sample: ", sample["name"], sample["sequence"])

    # Tokenize the protein
    input = tokenizer.encode(sample["sequence"], return_tensors="pt")
    print("Input: ", input)

    # Move to the GPU and make a prediction
    input = input.to("cuda")
    output = model(input)
    print("Output: ", output)

    break
```

## Citations

If you find the models useful in your research, we ask that you cite the paper:

```bibtex
@article{Fournier2024.09.23.614603,
	title        = {Protein Language Models: Is Scaling Necessary?},
	author       = {Fournier, Quentin and Vernon, Robert M. and van der Sloot, Almer and Schulz, Benjamin and Chandar, Sarath and Langmead, Christopher James},
	year         = {2024},
	journal      = {bioRxiv},
	publisher    = {Cold Spring Harbor Laboratory},
	doi          = {10.1101/2024.09.23.614603},
	url          = {https://www.biorxiv.org/content/early/2024/09/23/2024.09.23.614603},
	elocation-id = {2024.09.23.614603},
	eprint       = {https://www.biorxiv.org/content/early/2024/09/23/2024.09.23.614603.full.pdf}
}
```
