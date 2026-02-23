# FROM nvcr.io/nvidia/vllm:26.01-py3
FROM gitlab-master.nvidia.com:5005/dl/dgx/vllm:main-py3.43005406-devel
# using this because we need vllm >= 0.14 to work with Transformers v5. no released nvidia version with this yet.

# The vLLM image has CUDA 13.1 runtime and nvcc, but missing dev headers (cusparse.h, nvtx, etc.)
# Install cuda-keyring to add NVIDIA's apt repo, then install the dev headers for transformer_engine
RUN apt-get update && apt-get install -y --no-install-recommends wget && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm cuda-keyring_1.1-1_all.deb && \
    apt-get update && apt-get install -y --no-install-recommends \
    cuda-nvtx-13-1 \
    cuda-cupti-dev-13-1 \
    cuda-nvml-dev-13-1 \
    libcusparse-dev-13-1 \
    libcusolver-dev-13-1 \
    libcufft-dev-13-1 \
    libnvjitlink-dev-13-1 \
    libnvjpeg-dev-13-1 \
    libcublasmp0-dev-cuda-13 \
    libcudnn9-cuda-13 \
    && rm -rf /var/lib/apt/lists/*

# Install remaining dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=/requirements.txt \
    pip install -r /requirements.txt

# Install transformer_engine from source (force build for CUDA 13.1, not pre-built cu12 wheel)
RUN pip install --no-build-isolation transformer_engine[pytorch]

RUN pip install transformers[torch]==5.0.0


WORKDIR /workspace/bionemo
COPY . .
