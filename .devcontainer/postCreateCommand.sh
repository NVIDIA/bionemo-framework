#!/bin/bash

pip install --disable-pip-version-check --no-cache-dir -e 3rdparty/Megatron-LM
pip install --disable-pip-version-check --no-cache-dir -e 3rdparty/NeMo[all]

for SUB_PKG in sub-packages/bionemo-*;
do
    pip install --disable-pip-version-check --no-cache-dir --no-deps -e $SUB_PKG
done
