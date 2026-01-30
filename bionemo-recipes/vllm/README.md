# vLLM inference for BioNeMo Models

# Phase 1: Build docker image and validate vLLM

To build the image:

```bash
docker build -t vllm .
```

Set `HF_TOKEN` in your environment to avoid getting rate limited.

To launch a container:

```bash
docker run -it --gpus all --network host --ipc=host -e HF_TOKEN --rm -v ${PWD}:/workspace/bionemo vllm /bin/bash
```

To generate samples using vLLM:

```python
python vllm_sample.py
```

### Notes on dockerfile.

The options were either to install TransformerEngine in the vLLM image, or to install vLLM in the pytorch image. The first method was much less messy, as vLLM depends on specific pytorch versions and installs its own, overriding the custom install in the nvidia image. In order to do so, several missing cuda dev libraries had to be installed in the vLLM.
