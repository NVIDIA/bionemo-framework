#!/bin/bash

for sub in ./3rdparty/* ./sub-packages/*; do
    uv pip install -v --no-deps --no-build-isolation --editable $sub
done
