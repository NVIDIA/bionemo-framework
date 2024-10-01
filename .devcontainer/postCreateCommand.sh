#!/bin/bash

for sub in ./3rdparty/* ./sub-packages/*; do
    uv pip install --no-deps --no-build-isolation --editable $sub
done
