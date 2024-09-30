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

RUN apt-get install -y gnupg

# Check the nemo dependency for causal conv1d and make sure this checkout
# tag matches. If not, update the tag in the following line.
RUN CAUSAL_CONV1D_FORCE_BUILD=TRUE pip --disable-pip-version-check --no-cache-dir install \
  git+https://github.com/Dao-AILab/causal-conv1d.git@v1.2.0.post2

# Mamba dependancy installation
RUN pip --disable-pip-version-check --no-cache-dir install \
  git+https://github.com/state-spaces/mamba.git@v2.0.3

RUN pip install hatchling   # needed to install nemo-run
ARG NEMU_RUN_TAG=34259bd3e752fef94045a9a019e4aaf62bd11ce2
RUN pip install nemo_run@git+https://github.com/NVIDIA/NeMo-Run.git@${NEMU_RUN_TAG}

RUN mkdir -p /workspace/bionemo2/

# Delete the temporary /build directory.
WORKDIR /workspace
RUN rm -rf /build

# Addressing Security Scan Vulnerabilities
RUN rm -rf /opt/pytorch/pytorch/third_party/onnx
RUN apt-get update  && \
  apt-get install -y openssh-client=1:8.9p1-3ubuntu0.10 && \
  rm -rf /var/lib/apt/lists/*
RUN apt purge -y libslurm37 libpmi2-0 && \
  apt autoremove -y
RUN source /usr/local/nvm/nvm.sh && \
  NODE_VER=$(nvm current) && \
  nvm deactivate && \
  nvm uninstall $NODE_VER && \
  sed -i "/NVM/d" /root/.bashrc && \
  sed -i "/nvm.sh/d" /etc/bash.bashrc

# Use UV to install python packages from the workspace. This just installs packages into the system's python
# environment, and does not use the current uv.lock file.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy \
  UV_COMPILE_BYTECODE=1 \
  UV_PYTHON_DOWNLOADS=never \
  UV_SYSTEM_PYTHON=true

WORKDIR /workspace/bionemo2

# Install 3rd-party deps and bionemo submodules.
COPY ./3rdparty /workspace/bionemo2/3rdparty
COPY ./sub-packages /workspace/bionemo2/sub-packages

# Note, we need to mount the .git folder here so that setuptools-scm is able to fetch git tag for version.
RUN --mount=type=bind,source=./.git,target=./.git \
  --mount=type=cache,id=uv-cache,target=/root/.cache,sharing=locked \
  <<EOT
uv pip install --no-build-isolation -v ./3rdparty/* ./sub-packages/bionemo-*
rm -rf ./3rdparty
rm -rf /tmp/*
EOT

# In the devcontainer image, we just copy over the finished `dist-packages` folder from the build image back into the
# base pytorch container. We can then set up a non-root user and uninstall the bionemo and 3rd-party packages, so that
# they can be installed in an editable fashion from the workspace directory. This lets us install all the package
# dependencies in a cached fashion, so they don't have to be built from scratch every time the devcontainer is rebuilt.
FROM ${BASE_IMAGE} AS dev

RUN --mount=type=cache,id=apt-cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,id=apt-lib,target=/var/lib/apt,sharing=locked \
  <<EOT
apt-get update -qy
apt-get install -qyy \
  sudo
rm -rf /tmp/* /var/tmp/*
EOT

# Create a non-root user to use inside a devcontainer.
ARG USERNAME=bionemo
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
  && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
  && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
  && chmod 0440 /etc/sudoers.d/$USERNAME

RUN rm -rf /usr/local/lib/python3.10/dist-packages
COPY --from=bionemo2-base --chown=$USERNAME:$USERNAME --chmod=777 \
  /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
RUN <<EOT
  rm -rf /usr/local/lib/python3.10/dist-packages/bionemo*
  pip uninstall -y nemo_toolkit megatron_core
EOT

# The 'release' target needs to be last so that it's the default build target. In the future, we could consider a setup
# similar to the devcontainer above, where we copy the dist-packages folder from the build image into the release image.
# This would reduce the overall image size by reducing the number of intermediate layers. In the meantime, we match the
# existing release image build by copying over remaining files from the repo into the container.
FROM bionemo2-base AS release

COPY VERSION .
COPY ./scripts ./scripts
COPY ./README.md ./