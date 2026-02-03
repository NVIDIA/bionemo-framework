# syntax=docker/dockerfile:1.4
FROM nvcr.io/nvidia/pytorch:25.12-py3

# Install sccache for faster builds
RUN --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update \
    && apt-get install -y sccache \
    && rm -rf /var/lib/apt/lists/*

# Uninstall pre-installed Transformer Engine and install from source
RUN pip uninstall -y transformer-engine && \
    NVTE_USE_CCACHE=1 NVTE_CCACHE_BIN=sccache NVTE_FRAMEWORK=pytorch NVTE_BUILD_DEBUG=1 \
    pip install -v --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@main

# Install BioNeMo requirements
WORKDIR /workspace/bionemo
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=/requirements.txt \
    PIP_CONSTRAINT= pip install -r /requirements.txt

COPY . .
