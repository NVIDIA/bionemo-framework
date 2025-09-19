FROM nvcr.io/nvidia/pytorch:25.08-py3
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=/workspace/requirements.txt \
    PIP_CONSTRAINT= pip install -r /workspace/requirements.txt
