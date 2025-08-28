# ESM-2 Pre-training with Accelerate

```bash
NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 WANDB_PROJECT=<WANDB_PROJECT> \
accelerate launch --config_file  accelerate_config/default.yaml \
    --num_processes 2 train.py \
    --config-name=L0_sanity \
    ...
```
