# Hyperparameter Usage and Tuning

This section discusses recommended practices for choosing and tuning hyperparameters for BioNeMo models.

## General Information

Configuration files and command-line arguments can be used to define hyperparameters. Sets of configuration parameters are based on YAML files and constructed using [Hydra](https://hydra.cc/docs/intro/). Refer to the [Command Line Configuration section](./bionemo-fw-for-model-training-fw.md#command-line-configuration) for more information.

## Hyperparameter Tuning Tips

There is a lot of information on how to tune hyperparameters (refer to comprehensive [guide](https://github.com/google-research/tuning_playbook)). Here we provide a few tips that are specific to BioNeMo-based models.

1. Start small initially: dataset size, model parameters, number of epochs, and number of GPUs to tune hyperparameters.
2. Scale up experiment size gradually with best performing hyperparameters, for example, model size increases from 10M, 100M, 1B, 5B, 15B.
3. Use [Weights & Biases](https://wandb.ai/) to track experiment results. Group experiments by project per set of hyperparameters, and use meaningful names for experiments which include the parameters that have been varied. Stop experiments early if they are performing poorly.

## Recommended Hyperparameter Search Method

Below, hyperparameters and recommendations for their adjustment are provided. The proposed values are generally applicable to large language models built upon NeMo Megatron, such as BioNeMo models.

### Precision

* Configure with: `trainer.precision=bf16-mixed` if available, otherwise use`trainer.precision=16-mixed`.
* Switch to `trainer.precision=32` if training is unstable with bf16 or 16-bit.

### Gradient Clipping

* Configure with: `trainer.gradient_clip_val=1.0`
* Recommended alternative values: 0.5, 0.1
* Reduce value if training is unstable.

### Optimizer and Weight Decay

* Configure with: `model.optim.name=fused_adam`, `model.optim.weight_decay=0.01`
* Recommended alternative values: `model.optim.weight_decay=0.0`
* Increase weight decay value to mitigate over-fitting and stabilize training. Values that become too large may degrade performance.

### Learning Rate

* Configure with: `model.optim.lr=1e-4`
* Recommended alternative values: 2e-4, 5e-5
* Instability in training or validation loss may indicate that the learning rate is too high. Slow convergence and poor performance of converged model may indicate that LR is too low.

### Batch Size

* Configure with: `model.micro_batch_size=N` (per GPU batch size)
* Recommended value: use `N` resulting in 85-90% GPU memory utilization
* Keep `model.global_batch_size=null` to compute global batch size at run-time.
* Further increase the effective global batch size by using gradient accumulation (for example, `trainer.accumulate_grad_batches=2`).

### Model Parallelism

* For large models (that is > 1B parameters) use model tensor parallelism `model.tensor_model_parallel_size=N`
* For larger models (that is > 5B parameters) add also model pipeline parallelism `model.pipeline_model_parallel_size=N`
* The various parallelism options are independent and can be combined as needed.

### Dropout

* Configure with: `model.hidden_dropout=0.1`, `model.attention_dropout=0.1`
* Increase value to mitigate over-fitting. Values too large may degrade performance.

## General Training Tips

### Gradient Norm

* To mitigate spikes in gradient norm, reduce the gradient clipping value
* Lower the learning rate, although this can also reduce performance and make training slow
* Increase the number of warmup steps
* Replace layer normalization with configuration `model.normalization=normformer`
* Increase global batch size (for example, using gradient accumulation)
* Skip updates with large norm (for example, top 0.005% batches, leads to smoother loss)
* For debugging: try `trainer.precision=32` to differentiate problems in numerical calculation vs. data batch

### Data Cleaning

* Ensure data has been deduplicated to reduce memorization
* Filter out irrelevant / bad quality data (for example invalid SMILES strings)

### Model Architecture

* Pre-norm gives better performance but is less stable than post-ln. Normformer will be the most stable. Configure with `model.transformer_block_type` with options ['pre_ln', 'post_ln', 'normformer'].
* Model activaction of SwiGLU provides better performance at ~2% slowdown in training speed. Configure with `model.activation=swiglu`.
* Remove dropouts, configure with `model.hidden_dropout=0.0`, `model.attention_dropout=0.0`
* Remove bias term from linear layers (increase stability and speed, almost no performance cost). Configure with `model.bias=false`.

### Optimization

* Use 1-2k warmup steps with configuration `model.optim.sched.warmup_ratio=0.01` and `trainer.max_steps=100000`, or alternatively define warmup steps directly with configuration `model.optim.sched.warmup_steps=1000`.
* Batch size ramp-up can be done via consecutive training with increased batch size, where previous model is used to initialize the weights of the next model. This can be done with configuration `restore_from_path=<PATH TO .nemo FILE>`.
