# Llama-3 vLLM Inference

This recipe demonstrates serving a round-tripped
[Llama-3 TE checkpoint](../../../models/llama3/) via
[vLLM](https://github.com/vllm-project/vllm) (>= 0.14).

The workflow is:

1. Convert an HF checkpoint to TE format, then back to HF
   (`export_llama3.py`).
2. Serve the round-tripped checkpoint with vLLM.

See [tests/test_vllm.py](tests/test_vllm.py) for golden-value validation
confirming the round-tripped model matches the original.

## Installing vLLM in the container

There are two ways to get vLLM installed in the Docker image.

**Option 1: Build-time installation via Dockerfile build arg**

Pass `--build-arg INSTALL_VLLM=true` and `--build-arg TORCH_CUDA_ARCH_LIST=<arch>` when
building the image. `TORCH_CUDA_ARCH_LIST` is required when `INSTALL_VLLM=true` (the
Dockerfile will error if it is not set):

```bash
docker build -t llama3-vllm \
  --build-arg INSTALL_VLLM=true \
  --build-arg TORCH_CUDA_ARCH_LIST="9.0" .
```

**Option 2: Post-build installation via `install_vllm.sh`**

Build the base image normally, then run `install_vllm.sh` inside the container. The script
auto-detects the GPU architecture, or you can pass an explicit arch argument:

```bash
docker build -t llama3 .
docker run --rm -it --gpus all llama3 bash -c "./install_vllm.sh"
# or with an explicit architecture:
docker run --rm -it --gpus all llama3 bash -c "./install_vllm.sh 9.0"
```

## Benchmarking

The recipe includes benchmark scripts for comparing HuggingFace native and vLLM
inference:

```bash
python benchmark_hf.py
python benchmark_vllm.py
```
