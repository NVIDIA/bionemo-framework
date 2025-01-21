# Base image with apex and transformer engine, but without NeMo or Megatron-LM.
#  Note that the core NeMo docker container is defined here:
#   https://gitlab-master.nvidia.com/dl/JoC/nemo-ci/-/blob/main/llm_train/Dockerfile.train
#  with settings that get defined/injected from this config:
#   https://gitlab-master.nvidia.com/dl/JoC/nemo-ci/-/blob/main/.gitlab-ci.yml
#  We should keep versions in our container up to date to ensure that we get the latest tested perf improvements and
#   training loss curves from NeMo.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:24.12-py3

FROM rust:1.82.0 AS rust-env

RUN rustup set profile minimal && \
  rustup install 1.82.0 && \
  rustup target add x86_64-unknown-linux-gnu && \
  rustup default 1.82.0

FROM ${BASE_IMAGE} AS bionemo2-base

# Install NeMo dependencies.
WORKDIR /build

ARG MAX_JOBS=4
ENV MAX_JOBS=${MAX_JOBS}

# Install core apt packages.
RUN --mount=type=cache,id=apt-cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,id=apt-lib,target=/var/lib/apt,sharing=locked \
  <<EOF
set -eo pipefail
apt-get update -qy
apt-get install -qyy \
  libsndfile1 \
  ffmpeg \
  git \
  curl \
  pre-commit \
  sudo
apt-get upgrade -qyy \
  rsync
rm -rf /tmp/* /var/tmp/*
EOF

RUN apt-get install -y gnupg

# Check the nemo dependency for causal conv1d and make sure this checkout
# tag matches. If not, update the tag in the following line.
RUN CAUSAL_CONV1D_FORCE_BUILD=TRUE pip --disable-pip-version-check --no-cache-dir install \
  git+https://github.com/Dao-AILab/causal-conv1d.git@v1.2.2.post1

# Mamba dependancy installation
RUN pip --disable-pip-version-check --no-cache-dir install \
  git+https://github.com/state-spaces/mamba.git@v2.2.2

RUN pip install hatchling   # needed to install nemo-run
ARG NEMU_RUN_TAG=34259bd3e752fef94045a9a019e4aaf62bd11ce2
RUN pip install nemo_run@git+https://github.com/NVIDIA/NeMo-Run.git@${NEMU_RUN_TAG}

RUN mkdir -p /workspace/bionemo2/

# Delete the temporary /build directory.
WORKDIR /workspace
RUN rm -rf /build

# Addressing Security Scan Vulnerabilities
RUN rm -rf /opt/pytorch/pytorch/third_party/onnx


# Use UV to install python packages from the workspace. This just installs packages into the system's python
# environment, and does not use the current uv.lock file. Note that with python 3.12, we now need to set
# UV_BREAK_SYSTEM_PACKAGES, since the pytorch base image has made the decision not to use a virtual environment and UV
# does not respect the PIP_BREAK_SYSTEM_PACKAGES environment variable set in the base dockerfile.
COPY --from=ghcr.io/astral-sh/uv:0.4.25 /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy \
  UV_COMPILE_BYTECODE=1 \
  UV_PYTHON_DOWNLOADS=never \
  UV_SYSTEM_PYTHON=true \
  UV_BREAK_SYSTEM_PACKAGES=1

# Install the bionemo-geometric requirements ahead of copying over the rest of the repo, so that we can cache their
# installation. These involve building some torch extensions, so they can take a while to install.
RUN --mount=type=bind,source=./sub-packages/bionemo-geometric/requirements.txt,target=/requirements-pyg.txt \
    --mount=type=cache,target=/root/.cache \
  uv pip install --no-build-isolation -r /requirements-pyg.txt

COPY --from=rust-env /usr/local/cargo /usr/local/cargo
COPY --from=rust-env /usr/local/rustup /usr/local/rustup

ENV PATH="/usr/local/cargo/bin:/usr/local/rustup/bin:${PATH}"
ENV RUSTUP_HOME="/usr/local/rustup"

WORKDIR /workspace/bionemo2

# Install 3rd-party deps and bionemo submodules.
COPY ./LICENSE /workspace/bionemo2/LICENSE
COPY ./3rdparty /workspace/bionemo2/3rdparty
COPY ./sub-packages /workspace/bionemo2/sub-packages

# Note, we need to mount the .git folder here so that setuptools-scm is able to fetch git tag for version.
# Includes a hack to install tensorstore 0.1.45, which doesn't distribute a pypi wheel for python 3.12, and the metadata
# in the source distribution doesn't match the expected pypi version.
RUN --mount=type=bind,source=./.git,target=./.git \
    --mount=type=bind,source=./requirements-test.txt,target=/requirements-test.txt \
    --mount=type=bind,source=./requirements-cve.txt,target=/requirements-cve.txt \
    --mount=type=cache,target=/root/.cache <<EOF
set -eo pipefail

uv pip install maturin --no-build-isolation

uv pip install --no-build-isolation \
  ./3rdparty/* \
  ./sub-packages/bionemo-* \
  -r /requirements-cve.txt \
  -r /requirements-test.txt

rm -rf ./3rdparty
rm -rf /tmp/*
rm -rf ./sub-packages/bionemo-noodles/target
EOF

# In the devcontainer image, we just copy over the finished `dist-packages` folder from the build image back into the
# base pytorch container. We can then set up a non-root user and uninstall the bionemo and 3rd-party packages, so that
# they can be installed in an editable fashion from the workspace directory. This lets us install all the package
# dependencies in a cached fashion, so they don't have to be built from scratch every time the devcontainer is rebuilt.
FROM ${BASE_IMAGE} AS dev

RUN --mount=type=cache,id=apt-cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=apt-lib,target=/var/lib/apt,sharing=locked \
    <<EOF
set -eo pipefail
apt-get update -qy
apt-get install -qyy \
  sudo
rm -rf /tmp/* /var/tmp/*
EOF

# Use a non-root user to use inside a devcontainer (with ubuntu 23 and later, we can use the default ubuntu user).
ARG USERNAME=ubuntu
RUN echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
  && chmod 0440 /etc/sudoers.d/$USERNAME

# Here we delete the dist-packages directory from the pytorch base image, and copy over the dist-packages directory from
# the build image. This ensures we have all the necessary dependencies installed (megatron, nemo, etc.).
RUN <<EOF
  set -eo pipefail
  rm -rf /usr/local/lib/python3.12/dist-packages
  mkdir -p /usr/local/lib/python3.12/dist-packages
  chmod 777 /usr/local/lib/python3.12/dist-packages
  chmod 777 /usr/local/bin
EOF

USER $USERNAME

COPY --from=bionemo2-base --chown=$USERNAME:$USERNAME --chmod=777 \
  /usr/local/lib/python3.12/dist-packages /usr/local/lib/python3.12/dist-packages

COPY --from=ghcr.io/astral-sh/uv:0.4.25 /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy \
  UV_COMPILE_BYTECODE=0 \
  UV_PYTHON_DOWNLOADS=never \
  UV_SYSTEM_PYTHON=true \
  UV_BREAK_SYSTEM_PACKAGES=1

# Bring in the rust toolchain, as maturin is a dependency listed in requirements-dev
COPY --from=rust-env /usr/local/cargo /usr/local/cargo
COPY --from=rust-env /usr/local/rustup /usr/local/rustup

ENV PATH="/usr/local/cargo/bin:/usr/local/rustup/bin:${PATH}"
ENV RUSTUP_HOME="/usr/local/rustup"

RUN --mount=type=bind,source=./requirements-dev.txt,target=/workspace/bionemo2/requirements-dev.txt \
    --mount=type=cache,target=/root/.cache <<EOF
  set -eo pipefail
  uv pip install -r /workspace/bionemo2/requirements-dev.txt
  rm -rf /tmp/*
EOF

RUN <<EOF
  set -eo pipefail
  rm -rf /usr/local/lib/python3.12/dist-packages/bionemo*
  pip uninstall -y nemo_toolkit megatron_core
EOF


# Transformer engine attention defaults
# FIXME the following result in unstable training curves even if they are faster
#  see https://github.com/NVIDIA/bionemo-framework/pull/421
#ENV NVTE_FUSED_ATTN=1 NVTE_FLASH_ATTN=0
FROM dev AS development

WORKDIR /workspace/bionemo2
COPY --from=bionemo2-base /workspace/bionemo2/ .
COPY ./internal ./internal
# because of the `rm -rf ./3rdparty` in bionemo2-base
COPY ./3rdparty ./3rdparty

USER root
COPY --from=rust-env /usr/local/cargo /usr/local/cargo
COPY --from=rust-env /usr/local/rustup /usr/local/rustup

ENV PATH="/usr/local/cargo/bin:/usr/local/rustup/bin:${PATH}"
ENV RUSTUP_HOME="/usr/local/rustup"

RUN <<EOF
set -eo pipefail
find . -name __pycache__ -type d -print | xargs rm -rf
uv pip install --no-build-isolation --editable ./internal/infra-bionemo
for sub in ./3rdparty/* ./sub-packages/bionemo-*; do
    uv pip install --no-deps --no-build-isolation --editable $sub
done
EOF

# Since the entire repo is owned by root, switching username for development breaks things.
ARG USERNAME=ubuntu
RUN chown $USERNAME:$USERNAME -R /workspace/bionemo2/
USER $USERNAME

# The 'release' target needs to be last so that it's the default build target. In the future, we could consider a setup
# similar to the devcontainer above, where we copy the dist-packages folder from the build image into the release image.
# This would reduce the overall image size by reducing the number of intermediate layers. In the meantime, we match the
# existing release image build by copying over remaining files from the repo into the container.
FROM bionemo2-base AS release

RUN mkdir -p /workspace/bionemo2/.cache/

COPY VERSION .
COPY ./scripts ./scripts
COPY ./README.md ./
# Copy over folders so that the image can run tests in a self-contained fashion.
COPY ./ci/scripts ./ci/scripts
COPY ./docs ./docs

COPY --from=rust-env /usr/local/cargo /usr/local/cargo
COPY --from=rust-env /usr/local/rustup /usr/local/rustup


# RUN rm -rf /usr/local/cargo /usr/local/rustup
RUN chmod 777 -R /workspace/bionemo2/

# Transformer engine attention defaults
# We have to declare this again because the devcontainer splits from the release image's base.
# FIXME the following results in unstable training curves even if faster.
#  See https://github.com/NVIDIA/bionemo-framework/pull/421
# ENV NVTE_FUSED_ATTN=1 NVTE_FLASH_ATTN=0
