# syntax=docker/dockerfile:1.4
FROM nvcr.io/nvidia/pytorch:25.10-py3

WORKDIR /workspace/bionemo
COPY . .
# Fix for "No such file or directory: /workspace/TransformerEngine"
RUN rm -f /usr/local/lib/python*/dist-packages/transformer_engine-*.dist-info/direct_url.json
RUN pip freeze | grep transformer_engine > pip-constraints.txt
RUN --mount=type=secret,id=netrc,target=/root/.netrc \
    --mount=type=cache,target=/root/.cache/pip \
    PIP_CONSTRAINT= pip install -c pip-constraints.txt -e . --no-build-isolation
