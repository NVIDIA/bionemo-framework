# Build instructions:
#
# Local build:
#   docker build -t bionemo .
#
# Multi-platform build:
#   docker buildx create --use
#   docker buildx build --platform linux/amd64,linux/arm64 -t bionemo .
#
# Match the recipes test base image. The recipes workflow currently uses a
# squashed copy of this same upstream image for faster pulls in CI.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.02-py3

FROM rust:1.86.0 AS rust-env

RUN rustup set profile minimal && \
  rustup install 1.82.0 && \
  rustup default 1.82.0

FROM ${BASE_IMAGE} AS framework-base

COPY --from=ghcr.io/astral-sh/uv:0.6.13 /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy \
  UV_COMPILE_BYTECODE=1 \
  UV_PYTHON_DOWNLOADS=never \
  UV_SYSTEM_PYTHON=true \
  UV_BREAK_SYSTEM_PACKAGES=1

COPY --from=rust-env /usr/local/cargo /usr/local/cargo
COPY --from=rust-env /usr/local/rustup /usr/local/rustup
ENV PATH="/usr/local/cargo/bin:/usr/local/rustup/bin:${PATH}" \
  RUSTUP_HOME="/usr/local/rustup"

WORKDIR /workspace/bionemo

COPY ./LICENSE ./LICENSE
COPY ./README.md ./README.md
COPY ./VERSION ./VERSION
COPY ./pyproject.toml ./pyproject.toml
COPY ./requirements-cve.txt ./requirements-cve.txt
COPY ./requirements-dev.txt ./requirements-dev.txt
COPY ./requirements-test.txt ./requirements-test.txt
COPY ./ci/scripts ./ci/scripts
COPY ./docs ./docs
COPY ./sub-packages ./sub-packages

RUN --mount=type=cache,target=/root/.cache <<EOF
set -eo pipefail
bash ./ci/scripts/install_framework_test_env.sh
rm -rf /tmp/*
rm -rf ./sub-packages/bionemo-noodles/target
EOF

FROM framework-base AS dev

RUN --mount=type=cache,id=apt-cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,id=apt-lib,target=/var/lib/apt,sharing=locked \
  <<EOF
set -eo pipefail
apt-get update -qy
apt-get install -qyy \
  git \
  sudo
rm -rf /tmp/* /var/tmp/*
EOF

RUN --mount=type=cache,target=/root/.cache <<EOF
set -eo pipefail
uv pip install --system -r /workspace/bionemo/requirements-dev.txt
rm -rf /tmp/*
EOF

RUN <<EOF
set -eo pipefail
PYTHON_SITE=$(python - <<'PY'
import sysconfig
print(sysconfig.get_paths()["purelib"])
PY
)
PYTHON_SCRIPTS=$(python - <<'PY'
import sysconfig
print(sysconfig.get_paths()["scripts"])
PY
)
chmod 777 "$PYTHON_SITE" "$PYTHON_SCRIPTS"
EOF

ARG USERNAME=ubuntu
RUN echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME && \
  chmod 0440 /etc/sudoers.d/$USERNAME && \
  chown -R $USERNAME:$USERNAME /workspace/bionemo

USER $USERNAME

FROM dev AS development

FROM framework-base AS release

RUN mkdir -p /workspace/bionemo/.cache/ && \
  uv pip install --system h11==0.16.0 && \
  chmod 777 -R /workspace/bionemo/
