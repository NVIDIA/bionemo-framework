#!/bin/bash

# prelim required by release version of TAP
mkdir -p /workspace/bionemo2/.local/lib/python3.12/site-packages/
touch /workspace/bionemo2/.local/lib/python3.12/site-packages/usercustomize.py

# install from gitlab server
pip install git+https://gitlab-master.nvidia.com/dl/gwe/torch_automated_profiler@release

# RUN --mount=type=ssh cd /opt && git clone ssh://git@gitlab-master.nvidia.com:12051/dl/gwe/torch_automated_profiler.git\
#     && cd torch_automated_profiler\
#     && git fetch origin br_max_depth_1\
#     && git checkout -b br_max_depth_1 origin/br_max_depth_1\
#     && pip install -e . -v