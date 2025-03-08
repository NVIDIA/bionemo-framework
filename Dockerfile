# Base image with apex and transformer engine, but without NeMo or Megatron-LM.
#  Note that the core NeMo docker container is defined here:
#   https://gitlab-master.nvidia.com/dl/JoC/nemo-ci/-/blob/main/llm_train/Dockerfile.train
#  with settings that get defined/injected from this config:
#   https://gitlab-master.nvidia.com/dl/JoC/nemo-ci/-/blob/main/.gitlab-ci.yml
#  We should keep versions in our container up to date to ensure that we get the latest tested perf improvements and
#   training loss curves from NeMo.
FROM rust:1.82.0 AS rust-env

RUN rustup set profile minimal && \
  rustup install 1.82.0 && \
  rustup target add x86_64-unknown-linux-gnu && \
  rustup default 1.82.0

FROM nvcr.io/nvidia/pytorch:25.01-py3 AS dev

# Install core apt packages.
RUN --mount=type=cache,id=apt-cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,id=apt-lib,target=/var/lib/apt,sharing=locked \
  <<EOF
set -eo pipefail
apt-get update -qy
apt-get install -qyy \
  -o APT::Install-Recommends=false \
  -o APT::Install-Suggests=false \
  libsndfile1 \
  ffmpeg \
  git \
  curl \
  pre-commit \
  sudo \
  gnupg \
  gosu
apt-get upgrade -qyy \
  rsync
rm -rf /tmp/* /var/tmp/*
EOF

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

WORKDIR /workspace

# Use a non-root user to use inside a devcontainer (with ubuntu 23 and later, we can use the default ubuntu user).
ARG USERNAME=ubuntu
RUN echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
  && chmod 0440 /etc/sudoers.d/$USERNAME

# Addressing Security Scan Vulnerabilities
RUN rm -rf /opt/pytorch/pytorch/third_party/onnx

COPY --from=rust-env /usr/local/cargo /usr/local/cargo
COPY --from=rust-env /usr/local/rustup /usr/local/rustup

# Use UV to install python packages from the workspace. This just installs packages into the system's python
# environment, and does not use the current uv.lock file. Note that with python 3.12, we now need to set
# UV_BREAK_SYSTEM_PACKAGES, since the pytorch base image has made the decision not to use a virtual environment and UV
# does not respect the PIP_BREAK_SYSTEM_PACKAGES environment variable set in the base dockerfile.
COPY --from=ghcr.io/astral-sh/uv:0.4.25 /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy \
  UV_COMPILE_BYTECODE=1 \
  UV_PYTHON_DOWNLOADS=never \
  UV_SYSTEM_PYTHON=true \
  UV_BREAK_SYSTEM_PACKAGES=1 \
  PATH="/usr/local/cargo/bin:/usr/local/rustup/bin:${PATH}" \
  RUSTUP_HOME="/usr/local/rustup"

# Install the bionemo-geometric requirements ahead of copying over the rest of the repo, so that we can cache their
# installation. These involve building some torch extensions, so they can take a while to install.
RUN --mount=type=bind,source=./sub-packages/bionemo-geometric/requirements.txt,target=/requirements-pyg.txt \
  --mount=type=cache,target=/root/.cache \
  uv pip install --no-build-isolation maturin -r /requirements-pyg.txt

COPY ci/docker/entrypoint.sh /usr/local/bin/
ENTRYPOINT [ "entrypoint.sh" ]

FROM dev AS release

# Install 3rd-party deps and bionemo submodules.
WORKDIR /workspace/bionemo2
RUN mkdir -p /workspace/bionemo2/.cache/
COPY ./LICENSE /workspace/bionemo2/LICENSE

# TODO: Delete these lines when BIONEMO-75 is complete.
COPY ./3rdparty /workspace/bionemo2/3rdparty
COPY ./sub-packages /workspace/bionemo2/sub-packages

COPY VERSION .
COPY ./scripts ./scripts
COPY ./README.md ./
# Copy over folders so that the image can run tests in a self-contained fashion.
COPY ./ci/scripts ./ci/scripts
COPY ./docs ./docs

# Note, we need to mount the .git folder here so that setuptools-scm is able to fetch git tag for version.
# Includes a hack to install tensorstore 0.1.45, which doesn't distribute a pypi wheel for python 3.12, and the metadata
# in the source distribution doesn't match the expected pypi version.

RUN --mount=type=bind,source=./.git,target=./.git \
  --mount=type=bind,source=./requirements-test.txt,target=/requirements-test.txt \
  --mount=type=bind,source=./requirements-cve.txt,target=/requirements-cve.txt \
  --mount=type=cache,target=/root/.cache <<EOF
set -eo pipefail

uv pip install --no-build-isolation \
  ./3rdparty/* \
  ./sub-packages/bionemo-* \
  -r /requirements-cve.txt \
  -r /requirements-test.txt

# Addressing security scan issue - CVE vulnerability https://github.com/advisories/GHSA-g4r7-86gm-pgqc The package is a
# dependency of lm_eval from NeMo requirements_eval.txt. We also remove zstandard, another dependency of lm_eval, which
# seems to be causing issues with NGC downloads. See https://nvbugspro.nvidia.com/bug/5149698
uv pip uninstall sqlitedict zstandard

rm -rf ./3rdparty
rm -rf /tmp/*
rm -rf ./sub-packages/bionemo-noodles/target
EOF

RUN chown -R ubuntu:ubuntu /workspace/bionemo2/*
