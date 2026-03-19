# syntax=docker/dockerfile:1.4
FROM nvcr.io/nvidia/pytorch:26.02-py3

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=/requirements.txt \
    PIP_CONSTRAINT= pip install -r /requirements.txt

WORKDIR /workspace/bionemo
COPY . .

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/

USER ubuntu
ENV HOME=/home/ubuntu
ENV PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin

RUN set -eux; \
    curl -fsSL https://claude.ai/install.sh | bash; \
    ls -la /home/ubuntu/.local/bin; \
    test -x /home/ubuntu/.local/bin/claude

USER root
RUN ln -sf /home/ubuntu/.local/bin/claude /usr/local/bin/claude

ENV PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin
