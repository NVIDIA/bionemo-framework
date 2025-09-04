FROM nvcr.io/nvidia/pytorch:25.01-py3

# Install xformers and TE manually.
RUN MAX_JOBS=4 pip --disable-pip-version-check --no-cache-dir install -v git+https://github.com/facebookresearch/xformers.git@v0.0.29.post1#egg=xformers
RUN PIP_CONSTRAINT= NVTE_FRAMEWORK=pytorch MAX_JOBS=4 pip --disable-pip-version-check --no-cache-dir install -v git+https://github.com/nvidia/TransformerEngine.git@v2.4

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /workspace/bionemo
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    PIP_CONSTRAINT= pip install -e .
