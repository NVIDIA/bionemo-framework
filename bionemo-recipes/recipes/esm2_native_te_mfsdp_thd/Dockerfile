# syntax=docker/dockerfile:1.4
# FROM nvcr.io/nvidia/pytorch:25.08-py3
FROM gitlab-master.nvidia.com:5005/dl/transformerengine/transformerengine:te_ci-pytorch-py3-devel

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=/requirements.txt \
    PIP_CONSTRAINT= pip install -r /requirements.txt

WORKDIR /workspace/bionemo
COPY . .
