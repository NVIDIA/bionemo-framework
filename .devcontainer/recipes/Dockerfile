FROM nvcr.io/nvidia/pytorch:25.09-py3
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=/workspace/requirements.txt \
    PIP_CONSTRAINT= pip install -r /workspace/requirements.txt

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/
USER ubuntu
RUN curl https://cursor.com/install -fsS | bash  # Install cursor-agent CLI tool
RUN uv tool install pre-commit --with pre-commit-uv --force-reinstall
