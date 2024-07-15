# Base image with apex and transformer engine, but without NeMo or Megatron-LM.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:24.02-py3
FROM ${BASE_IMAGE} AS bionemo2-base

# Install NeMo dependencies.
WORKDIR /build

ARG MAX_JOBS=4
ENV MAX_JOBS=${MAX_JOBS}

# See NeMo readme for the latest tested versions of these libraries
ARG APEX_COMMIT=810ffae374a2b9cb4b5c5e28eaeca7d7998fca0c
RUN git clone https://github.com/NVIDIA/apex.git && \
  cd apex && \
  git checkout ${APEX_COMMIT} && \
  pip install . -v --no-build-isolation --disable-pip-version-check --no-cache-dir \
  --config-settings "--build-option=--cpp_ext --cuda_ext --fast_layer_norm --distributed_adam --deprecated_fused_adam --group_norm"

# Transformer Engine pre-1.7.0. 1.7 standardizes the meaning of bits in the attention mask to match
ARG TE_COMMIT=7d576ed25266a17a7b651f2c12e8498f67e0baea
RUN git clone https://github.com/NVIDIA/TransformerEngine.git && \
  cd TransformerEngine && \
  git fetch origin ${TE_COMMIT} && \
  git checkout FETCH_HEAD && \
  git submodule init && git submodule update && \
  NVTE_FRAMEWORK=pytorch NVTE_WITH_USERBUFFERS=1 MPI_HOME=/usr/local/mpi pip install .

# Install core apt packages.
RUN apt-get update \
  && apt-get install -y \
  libsndfile1 \
  ffmpeg \
  git \
  curl \
  pre-commit \
  sudo \
  && rm -rf /var/lib/apt/lists/*

# Check the nemo dependency for causal conv1d and make sure this checkout
# tag matches. If not, update the tag in the following line.
RUN CAUSAL_CONV1D_FORCE_BUILD=TRUE pip --disable-pip-version-check --no-cache-dir install \
  git+https://github.com/Dao-AILab/causal-conv1d.git@v1.2.0.post2

# Copy and install pypi depedencies.
RUN mkdir /tmp/pip-tmp
COPY requirements-dev.txt /tmp/pip-tmp/
COPY requirements-test.txt /tmp/pip-tmp/
COPY sub-packages/bionemo-fw/requirements.txt /tmp/pip-tmp/requirements-fw.txt
COPY sub-packages/bionemo-contrib/requirements.txt /tmp/pip-tmp/requirements-contrib.txt

RUN pip --disable-pip-version-check --no-cache-dir install \
  -r /tmp/pip-tmp/requirements-dev.txt \
  -r /tmp/pip-tmp/requirements-test.txt \
  -r /tmp/pip-tmp/requirements-fw.txt \
  -r /tmp/pip-tmp/requirements-contrib.txt \
  && rm -rf /tmp/pip-tmp

# Change the workspace and delete the temporary /build directory.
WORKDIR /workspace
RUN rm -rf /build

# Create a non-root user to use inside a devcontainer.
ARG USERNAME=bionemo
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
  && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
  && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
  && chmod 0440 /etc/sudoers.d/$USERNAME


# Create a development image with NeMo and Megatron-LM pre-installed.
FROM bionemo2-base AS standalone

# This is the latest commit of megatron-lm on 2024/06/14
# feel free to try updating.
ARG MEGATRON_COMMIT=c7a1f82d761577e6ca0338d3521eac82f2aa0904
RUN pip --disable-pip-version-check --no-cache-dir install \
  git+https://github.com/NVIDIA/Megatron-LM.git@${MEGATRON_COMMIT}

# Full install of NeMo from source.
ARG NEMO_COMMIT=d28c1b2dd7c8539299a4c31f7c8d1678e2cbb9c8
RUN pip --disable-pip-version-check --no-cache-dir install \
  git+https://github.com/NVIDIA/NeMo.git@${NEMO_COMMIT}#egg=nemo_toolkit[all]


# Create a release image with bionemo2 installed.
FROM standalone AS release

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
