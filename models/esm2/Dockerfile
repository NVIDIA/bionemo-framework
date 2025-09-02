FROM nvcr.io/nvidia/pytorch:25.06-py3
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /workspace/bionemo
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --break-system-packages -e .
