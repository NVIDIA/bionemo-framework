#!/bin/bash
# ============================================================================
# Run all 6 memory profiler benchmarks (8B + 70B x BF16/MXFP8/MXFP8+qinit)
#
# Usage (inside Docker container on a B300 8-GPU node):
#   bash /workspace/bionemo/bionemo-recipes/recipes/llama3_native_te/benchmarks/run_all_memory_benchmarks.sh
#
# Runs from /tmp to avoid NFS permission issues.
# Snapshots saved to /tmp/memory_snapshots/ — copy to /workspace/memory_snapshots/ before exiting container.
# ============================================================================
set -euxo pipefail

export PYTHONPATH=/workspace/transformer_engine:${PYTHONPATH:-}
RECIPE=/workspace/bionemo/bionemo-recipes/recipes/llama3_native_te
TRAIN=$RECIPE/train_fsdp2.py
TRAIN_CP=$RECIPE/train_fsdp2_cp.py
CFG=$RECIPE/hydra_config
SNAP=/tmp/memory_snapshots

mkdir -p $SNAP

echo "=== 1/6: 8B BF16 ==="
torchrun --nproc_per_node=8 $TRAIN --config-name L2_lingua_7b --config-path $CFG \
  dataset.micro_batch_size=4 dataset.use_stateful_dataloader=true \
  grad_acc_steps=1 num_train_steps=100 \
  checkpoint.save_every_n_steps=999999 checkpoint.resume_from_checkpoint=false \
  logger.frequency=10 \
  memory_profiler.enabled=true memory_profiler.snapshot_dir=$SNAP/8b_bf16 \
  wandb.name=8b_bf16_mbs4 wandb.id=8b_bf16_mbs4 wandb.project=lingua-7b \
  hydra.run.dir=/tmp/hydra_outputs/8b_bf16

echo "=== 2/6: 8B MXFP8 no qinit ==="
torchrun --nproc_per_node=8 $TRAIN --config-name L2_lingua_7b_mxfp8 --config-path $CFG \
  fp8_layers=null \
  dataset.micro_batch_size=4 dataset.use_stateful_dataloader=true \
  grad_acc_steps=1 num_train_steps=100 \
  checkpoint.save_every_n_steps=999999 checkpoint.resume_from_checkpoint=false \
  logger.frequency=10 \
  memory_profiler.enabled=true memory_profiler.snapshot_dir=$SNAP/8b_mxfp8 \
  wandb.name=8b_mxfp8_mbs4 wandb.id=8b_mxfp8_mbs4 wandb.project=lingua-7b \
  hydra.run.dir=/tmp/hydra_outputs/8b_mxfp8

echo "=== 3/6: 8B MXFP8 qinit ==="
torchrun --nproc_per_node=8 $TRAIN --config-name L2_lingua_7b_mxfp8_qinit --config-path $CFG \
  fp8_config.quantized_model_init_kwargs.preserve_high_precision_init_val=false \
  dataset.micro_batch_size=4 dataset.use_stateful_dataloader=true \
  grad_acc_steps=1 num_train_steps=100 \
  checkpoint.save_every_n_steps=999999 checkpoint.resume_from_checkpoint=false \
  logger.frequency=10 \
  memory_profiler.enabled=true memory_profiler.snapshot_dir=$SNAP/8b_mxfp8_qinit \
  wandb.name=8b_mxfp8_qinit_mbs4 wandb.id=8b_mxfp8_qinit_mbs4 wandb.project=lingua-7b \
  hydra.run.dir=/tmp/hydra_outputs/8b_mxfp8_qinit

echo "=== 4/6: 70B BF16 ==="
torchrun --nproc_per_node=8 $TRAIN_CP --config-name L2_lingua_70b_computelab --config-path $CFG \
  num_train_steps=100 \
  checkpoint.save_every_n_steps=999999 checkpoint.resume_from_checkpoint=false \
  logger.frequency=10 \
  memory_profiler.enabled=true memory_profiler.snapshot_dir=$SNAP/70b_bf16 \
  wandb.name=70b_bf16_mbs1 wandb.id=70b_bf16_mbs1 wandb.project=lingua-70b \
  hydra.run.dir=/tmp/hydra_outputs/70b_bf16

echo "=== 5/6: 70B MXFP8 no qinit ==="
torchrun --nproc_per_node=8 $TRAIN_CP --config-name L2_lingua_70b_computelab --config-path $CFG \
  fp8_config.enabled=true \
  fp8_config.fp8_recipe=transformer_engine.common.recipe.MXFP8BlockScaling \
  fp8_config.fp8_format=E4M3 \
  fp8_layers=null \
  num_train_steps=100 \
  checkpoint.save_every_n_steps=999999 checkpoint.resume_from_checkpoint=false \
  logger.frequency=10 \
  memory_profiler.enabled=true memory_profiler.snapshot_dir=$SNAP/70b_mxfp8 \
  wandb.name=70b_mxfp8_mbs1 wandb.id=70b_mxfp8_mbs1 wandb.project=lingua-70b \
  hydra.run.dir=/tmp/hydra_outputs/70b_mxfp8

echo "=== 6/6: 70B MXFP8 qinit ==="
torchrun --nproc_per_node=8 $TRAIN_CP --config-name L2_lingua_70b_computelab --config-path $CFG \
  fp8_config.enabled=true \
  fp8_config.fp8_recipe=transformer_engine.common.recipe.MXFP8BlockScaling \
  fp8_config.fp8_format=E4M3 \
  fp8_config.quantized_model_init_kwargs.enabled=true \
  fp8_config.quantized_model_init_kwargs.preserve_high_precision_init_val=false \
  num_train_steps=100 \
  checkpoint.save_every_n_steps=999999 checkpoint.resume_from_checkpoint=false \
  logger.frequency=10 \
  memory_profiler.enabled=true memory_profiler.snapshot_dir=$SNAP/70b_mxfp8_qinit \
  wandb.name=70b_mxfp8_qinit_mbs1 wandb.id=70b_mxfp8_qinit_mbs1 wandb.project=lingua-70b \
  hydra.run.dir=/tmp/hydra_outputs/70b_mxfp8_qinit

echo "=== ALL 6 DONE ==="
echo "Snapshots in: $SNAP/"
ls -la $SNAP/*/memory_snapshot.pickle 2>/dev/null || echo "Check individual snapshot dirs"
echo "Copy to host: cp -r /tmp/memory_snapshots/ /workspace/memory_snapshots/"
