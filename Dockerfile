# Use the specified base image
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:24.02-py3
FROM ${BASE_IMAGE}

# Set the working directory
WORKDIR /workspace/

ARG MAX_JOBS=4
ENV MAX_JOBS=${MAX_JOBS}

# See NeMo readme for the latest tested versions of these libraries
RUN git clone https://github.com/NVIDIA/apex.git && \
  cd apex && \
  git checkout 810ffae374a2b9cb4b5c5e28eaeca7d7998fca0c && \
  pip install . -v --no-build-isolation --disable-pip-version-check --no-cache-dir --config-settings "--build-option=--cpp_ext --cuda_ext --fast_layer_norm --distributed_adam --deprecated_fused_adam --group_norm"

# Transformer Engine pre-1.7.0. 1.7 standardizes the meaning of bits in the attention mask to match
#  Use the version NeMo claims works in the readme (bfe21c3d68b0a9951e5716fb520045db53419c5e)
RUN git clone https://github.com/NVIDIA/TransformerEngine.git && \
  cd TransformerEngine && \
  git fetch origin 7d576ed25266a17a7b651f2c12e8498f67e0baea && \
  git checkout FETCH_HEAD && \
  git submodule init && git submodule update && \
  NVTE_FRAMEWORK=pytorch NVTE_WITH_USERBUFFERS=1 MPI_HOME=/usr/local/mpi pip install .

# This is the latest commit of megatron-lm on 2024/06/14
#   feel free to try updating.
RUN git clone https://github.com/NVIDIA/Megatron-LM.git && \
  cd Megatron-LM && \
  git checkout c7a1f82d761577e6ca0338d3521eac82f2aa0904 && \
  pip install .

# Install NeMo dependencies including apt packages and causal-conv1d
RUN apt-get update && apt-get install -y libsndfile1 ffmpeg && rm -rf /var/lib/apt/lists/*
# Check the nemo dependency for causal conv1d and make sure this checkout
#  tag matches. If not, update the tag in the following line.
RUN git clone https://github.com/Dao-AILab/causal-conv1d.git && \
  cd causal-conv1d && \
  git checkout v1.2.0.post2  && \
  CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install .
# Full install of NeMo from source
RUN git clone https://github.com/NVIDIA/NeMo.git && \
    cd NeMo && \
    git checkout d28c1b2dd7c8539299a4c31f7c8d1678e2cbb9c8 && \
    ./reinstall.sh

# Install any additional dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    pre-commit \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/bionemo2

# install devtools and test dependencies
COPY ./requirements-dev.txt .
RUN pip install -r requirements-dev.txt

COPY ./requirements-test.txt .
RUN pip install -r requirements-test.txt

#
# install dependencies of all bionemo namespace packages
#

# TODO: install reqs of feature package(s)
# --
# COPY ./sub-packages/bionemo-.../requirements.txt ./sub-packages/bionemo-.../requirements.txt
# RUN pip install -r ./sub-packages/bionemo-.../requirements.txt
# ...

COPY ./sub-packages/bionemo-fw/requirements.txt ./sub-packages/bionemo-fw/requirements.txt
RUN pip install -r ./sub-packages/bionemo-fw/requirements.txt

COPY ./sub-packages/bionemo-contrib/requirements.txt ./sub-packages/bionemo-contrib/requirements.txt
RUN pip install -r ./sub-packages/bionemo-contrib/requirements.txt

#
# install all bionemo namespaced code
#

# NOTE: We do not install from the `_requirements.txt` file, which contains each namespaced packages'
#       inter-dependent bionemo2 subpackage relationships. This is because we manually install each
#       subpackage's code independently. This speeds up the image building significantly.

# TODO install code of feature package(s)
# --
# WORKDIR /workspace/bionemo2/
# COPY ./sub-packages/bionemo-... ./sub-packages/bionemo-...
# WORKDIR /workspace/bionemo2/sub-packages/bionemo-...
# RUN pip install -r _requirements.txt
# RUN pip install --no-deps -e .
# ...

WORKDIR /workspace/bionemo2/
COPY ./sub-packages/bionemo-fw/ ./sub-packages/bionemo-fw/
WORKDIR /workspace/bionemo2/sub-packages/bionemo-fw/
RUN pip install --no-deps -e .

WORKDIR /workspace/bionemo2/
COPY ./sub-packages/bionemo-contrib/ ./sub-packages/bionemo-contrib/
WORKDIR /workspace/bionemo2/sub-packages/bionemo-contrib/
RUN pip install --no-deps -e .

WORKDIR /workspace/bionemo2/
COPY ./scripts ./scripts
COPY ./README.md ./


# Copy the rest of the bionemo2 project
WORKDIR /workspace/
# COPY . /workspace/bionemo2/
WORKDIR /workspace/bionemo2/
