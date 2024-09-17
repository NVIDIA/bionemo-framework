#!/bin/bash

WORKSPACE_ROOT=$(pwd)
uv pip install --python=$UV_PROJECT_ENVIRONMENT --no-deps --editable \
  $WORKSPACE_ROOT/3rdparty/* $WORKSPACE_ROOT/sub-packages/bionemo-*
