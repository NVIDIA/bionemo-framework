# PEFT Fine-tuning with TransformerEngine-accelerated ESM-2

This folder demonstrates how to fine-tune a TransformerEngine-accelerated ESM-2 model using PEFT.

Note: This recipe is a work in progress, and currently only demonstrates basic support for LoRA fine-tuning and
TransformerEngine layers. See `bionemo-recipes/models/esm2/tests/test_peft.py` for additional information and known
limitations.

## Commands to Launch LoRA Fine-tuning

To run single-process training on one GPU, run:

```bash
python train_lora.py
```

To run multi-process training locally on 2+ GPUs, run (e.g. 2 GPUs):

```bash
torchrun --nproc_per_node=2 train_lora_ddp.py
```
