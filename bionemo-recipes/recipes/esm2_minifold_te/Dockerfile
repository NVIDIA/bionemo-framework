# syntax=docker/dockerfile:1.4
FROM nvcr.io/nvidia/pytorch:26.02-py3

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=/requirements.txt \
    PIP_CONSTRAINT= pip install -r /requirements.txt

# Install Node.js (needed for Claude Code and Codex CLI)
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# Install Claude Code CLI
RUN npm install -g @anthropic-ai/claude-code

# Install OpenAI Codex CLI (authenticate at runtime with: codex login --device-auth)
RUN npm install -g @openai/codex

# Install pre-commit via uv
RUN uv tool install pre-commit --with pre-commit-uv --force-reinstall

WORKDIR /workspace/bionemo
COPY . .
