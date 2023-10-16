# Training Parallelism 

This section describes the various parallelism options available in BioNeMo models and provides best practices for scaling the number of parameters when training large models. An in-depth discussion of training models at scale is beyond the scope of this guide, but the reader is referred to a number of recent references {cite:p}`shoeybi2019,narayanan2021,kaplan2020`. 

## Supported Parallelism Features

The following parallelism options are supported by the current BioNeMo models:

* Data Parallelism - dividing the global batch between multiple GPUs and multiple nodes.
* Model Parallelism
    * Tensor Model Parallelism - dividing the model weights matrices between multiple GPUs and multiple nodes.
    * Pipeline Model Parallelism - dividing the model layers between multiple GPUs and multiple nodes. 

:::{note}
Pipeline model parallelism is available but not currently supported for BioNeMo models, thus `pipeline_model_parallel_size` should be set at 1.
:::

The global batch size is computed as follows:

```python
global_batch_size = \
( micro_batch_size * devices (GPUs) * num_nodes * accumulate_grad_batches ) /
( tensor_model_parallel_size * pipeline_model_parallel_size )
```

and the total number of devices must be an integer multiple of `tensor_model_parallel_size * pipeline_model_parallel_size`. 

These variables can be set in the YAML configuration file as follows:

```yaml
trainer:
  devices: 2 # number of GPUs
  num_nodes: 1
  accumulate_grad_batches: 1 # gradient accumulation steps
model:
  # model parallelism
  micro_batch_size: 1
  global_batch_size: null # compute automatically
  tensor_model_parallel_size: 1
  pipeline_model_parallel_size: 1
```

## Model Parallelism Guidelines

Model parallelism increases training time due to the increased communication required between GPUs. Before using model parallelism, ensure that all data parallelism options are exhausted. The order of preference for parallelism is: data parallelism, tensor model parallelism, and then pipeline model parallelism.
Using multiple nodes also adds communication overhead and should be used only when required by data or model parallelism requirements.

The following guidelines describe how to scale a large language model using model parallelism. The key model architecture parameters are: `hidden_size`, `ffn_hidden_size`, and `num_layers`. 

1. Increase the global batch size until 85-90% of GPU memory is used. Data parallelism may be utilized if needed.
2. Then scale model size by increasing `hidden_size` and `ffn_hidden_size`, while decreasing `micro_batch_size` as needed to control memory size.
3. Once the model is too large for the GPU even with `micro_batch_size=1` to fit in a single GPU memory, increase `tensor_model_parallel_size`.
4. Once the desired `hidden_size` and `ffn_hidden_size` have been reached, increase `num_layers` until model is too large to fit in memory with `micro_batch_size=1`. 

