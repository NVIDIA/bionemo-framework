## Data creation for pretraining
First download cellxgene into a nice metadata preserving heirarchy. Files will also be split into small parts so that we can more granularly break into train/val/test.
```bash
cd /workspace/bionemo
pip install cellxgene-census==1.13.0  # Make sure that the license terms etc are acceptable
python examples/singlecell/geneformer/download_pretraining_data.py --num-workers 32
```

Next split up the data into train/test/validation
```bash
python examples/singlecell/geneformer/scripts/train_val_test_split_from_metadata.py data/cellxgene_2023-12-15/input_data/dataset_metadata.csv data/cellxgene_2023-12-15/input_data/ --move
```

Optionally clean up the now empty directories, all data has been moved into `data/cellxgene_2023-12-15/input_data/[train, test, val]`
```bash
cd data/cellxgene_2023-12-15/input_data/
rm -rf assay__*
cd -
```

Now create the memmap files for this dataset
```bash
for STEP in train test val; do python bionemo/data/singlecell/sc_memmap.py --use-mp --num-workers 32 --data-path data/cellxgene_2023-12-15/input_data/$STEP --save-path data/cellxgene_2023-12-15/processed_data/$STEP; done
```

Now do some training:
```bash
python  examples/singlecell/geneformer/pretrain.py trainer.precision=bf16-mixed exp_manager.exp_dir=./results/test_new_dset exp_manager.create_wandb_logger=False exp_manager.wandb_logger_kwargs.name=workstation_test_ensg_loaders exp_manager.wandb_logger_kwargs.project=scFM_v8 exp_manager.resume_if_exists=False ++exp_manager.wandb_logger_kwargs.offline=False trainer.num_nodes=1 trainer.devices=2 trainer.max_steps=500000 trainer.accumulate_grad_batches=1 trainer.val_check_interval=100  model.micro_batch_size=32 model.optim.weight_decay=0.1 model.optim.lr=0.001  ++model.optim.betas.1=0.999 ++model.optim.sched.warmup_steps=100  ++model.optim.sched.constant_steps=100 ++model.optim.sched.min_lr=0.00002 ++model.optim.sched.max_steps=500000 ++model.hidden_dropout=0.02 ++model.attention_dropout=0.02 ++model.fp32_residual_connection=True ++model.layernorm_epsilon=1e-12  ++model.activation=relu  ++seed_everything=False do_training=False
python  examples/singlecell/geneformer/pretrain.py trainer.precision=bf16-mixed exp_manager.exp_dir=./results/test_new_dset exp_manager.create_wandb_logger=True exp_manager.wandb_logger_kwargs.name=workstation_test_ensg_loaders exp_manager.wandb_logger_kwargs.project=scFM_v8 exp_manager.resume_if_exists=False ++exp_manager.wandb_logger_kwargs.offline=False trainer.num_nodes=1 trainer.devices=2 trainer.max_steps=500000 trainer.accumulate_grad_batches=1 trainer.val_check_interval=100  model.micro_batch_size=32 model.optim.weight_decay=0.1 model.optim.lr=0.001  ++model.optim.betas.1=0.999 ++model.optim.sched.warmup_steps=100  ++model.optim.sched.constant_steps=100 ++model.optim.sched.min_lr=0.00002 ++model.optim.sched.max_steps=500000 ++model.hidden_dropout=0.02 ++model.attention_dropout=0.02 ++model.fp32_residual_connection=True ++model.layernorm_epsilon=1e-12  ++model.activation=relu  ++seed_everything=False do_training=True
```
