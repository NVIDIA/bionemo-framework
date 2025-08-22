# AMPLIFY 350M Pre-training with Accelerate

Running FP8 training locally on 2x5090s:

```bash
NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 WANDB_PROJECT=<WANDB_PROJECT> \
accelerate launch --config_file  accelerate_config/fp8_config.yaml \
    --num_processes 2 train.py \
    --config-name=L1_350M_partial_conv \
    trainer.per_device_train_batch_size=40 \
    data_size='parquet' \
    trainer.run_name="L1-350M-partial-conv-set-mxfp8" \
    stop_after_n_steps=1000
```

Running BF16 training on 2x5090 GPUs with Accelerate:

```bash
NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 WANDB_PROJECT=<WANDB_PROJECT> \
accelerate launch --config_file  accelerate_config/bf16_config.yaml \
    --num_processes 2 train.py \
    --config-name=L1_350M_partial_conv \
    trainer.per_device_train_batch_size=40 \
    data_size='parquet' \
    trainer.run_name="L1-350M-partial-conv-set-mxfp8" \
    stop_after_n_steps=1000
```
