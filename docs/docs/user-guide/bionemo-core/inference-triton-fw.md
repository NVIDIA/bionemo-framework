# Inference with Nvidia Triton Server

This section will use the pre-trained BioNeMo checkpoints to demonstrate [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server).
BioNeMo uses [PyTriton](https://github.com/triton-inference-server/pytriton), which provides utilities that simplify Triton configuration and local deployment in Python environments.

Sections:
- [Start Inference Server](#start-inference-server)
- [Perform Inference as a Client](#perform-inference-as-a-client)
- [Detailed Example of PyTriton Inference](#detailed-example-of-pytriton-inference)

**Before diving in, ensure you have all [necessary prerequisites](./pre-reqs.md).**

## Start Inference Server

Components for performing inference are part of the BioNeMo source code. This example demonstrates the use of these components.

In one terminal, you will start the PyTriton inference server as:
```bash
python -m bionemo.triton.inference_wrapper --config-path ${CONFIG_PATH}
```

Where `${CONFIG_PATH}` is an absolute or relative path to the directory containing the model's configuration files. By default, the configuration file to be used is named `infer.yaml`. If you want to override this, supply the filename via the `--config-name` parameter.

See [the inference server's documentation](../../bionemo/triton/README.md) for more details and examples.


## Perform Inference as a Client

BioNeMo comes with a set of example scripts for inference with PyTriton.

You may perform inference using either gRPC or HTTP, connecting to the [`tritonserver`](https://github.com/triton-inference-server) instance that the [`bionemo.triton.inference_wrapper`](../../bionemo/triton/inference_wrapper.py) program starts.

For convenince, we provide gRPC based clients that allow you to obtain inference results from the hosted models. One is [`bionemo.triton.client_encode`](../../bionemo/triton/client_encode.py), which povides access to obtaining the embeddings, hidden states, or sample inference for a model. The other is [`bionemo.triton.client_decode`](../../bionemo/triton/client_decode.py), which povides access to the decode inference for the model.

NOTE: Only MegaMolBART and MolMIM implements decoding and sample inference. All models (MegaMolBART, MolMIM, esm1nv, esm2nv, prott5nv) provide access for calculating embeddings and hidden states.

See [the inference client documentation](../../bionemo/triton/README.md) for more details and examples.


## Detailed Example of PyTriton Inference

For a detailed example of performing BioNeMo model inference with PyTriton, refer to [this detailed tutorial](deep-dive-esm1-pytriton-inference.md).
