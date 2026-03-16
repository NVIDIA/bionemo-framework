# Uncomment to use the latest TE from the NGC registry for debugging changes with latest TE.
# FROM gitlab-master.nvidia.com/dl/transformerengine/transformerengine:main-pytorch-py3-base
FROM nvcr.io/nvidia/pytorch:26.02-py3

# FIXME: Fix for "No such file or directory: /workspace/TransformerEngine"
#  Remove once bug has been addressed in the nvidia/pytorch container.
RUN rm -f /usr/local/lib/python*/dist-packages/transformer_engine-*.dist-info/direct_url.json

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
        | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
        | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt-get update && apt-get install -y --no-install-recommends gh tmux \
    && rm -rf /var/log/* /tmp/*

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=/workspace/requirements.txt \
    PIP_CONSTRAINT= pip install -r /workspace/requirements.txt

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/
USER ubuntu
RUN curl https://cursor.com/install -fsS | bash  # Install cursor-agent CLI tool
RUN curl -fsSL https://claude.ai/install.sh | bash  # Install Claude CLI tool
RUN uv tool install pre-commit --with pre-commit-uv --force-reinstall
ENV PATH="/home/ubuntu/.local/bin:${PATH}"
