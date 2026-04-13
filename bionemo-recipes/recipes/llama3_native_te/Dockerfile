# syntax=docker/dockerfile:1.4
FROM nvcr.io/nvidia/pytorch:26.03-py3

# Rebuild TransformerEngine from main branch (includes PR #2753: MXFP8 FusedAdam support).
# To pin a specific commit, replace 'main' with the commit hash, e.g.: @fcceeb96
# Build: docker build -t llama3_native_te:te-main-26.03 .
# --no-build-isolation: build TE against the container's PyTorch (prevents ABI mismatch)
RUN pip uninstall -y transformer_engine transformer_engine_torch transformer_engine_cu12 && \
    MAX_JOBS=8 pip install --no-cache-dir --no-build-isolation \
    "transformer_engine[pytorch] @ git+https://github.com/NVIDIA/TransformerEngine.git@main"

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=/requirements.txt \
    PIP_CONSTRAINT= pip install -r /requirements.txt

WORKDIR /workspace/bionemo
COPY . .
