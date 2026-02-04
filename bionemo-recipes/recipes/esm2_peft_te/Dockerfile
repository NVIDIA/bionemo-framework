FROM nvcr.io/nvidia/pytorch:25.12-py3

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=esm2_peft_te/requirements.txt,target=/requirements.txt \
    PIP_CONSTRAINT= pip install -r /requirements.txt

WORKDIR /workspace/bionemo-recipes/recipes/esm2_peft_te
COPY esm2_peft_te/ /workspace/bionemo-recipes/recipes/esm2_peft_te
COPY esm2_native_te/checkpoint.py /workspace/bionemo-recipes/recipes/esm2_native_te/checkpoint.py
COPY esm2_native_te/collator.py /workspace/bionemo-recipes/recipes/esm2_native_te/collator.py
COPY esm2_native_te/distributed_config.py /workspace/bionemo-recipes/recipes/esm2_native_te/distributed_config.py
COPY esm2_native_te/scheduler.py /workspace/bionemo-recipes/recipes/esm2_native_te/scheduler.py
