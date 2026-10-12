#!/usr/bin/env bash

# GitHub Actions invokes pytest after sourcing this file. Select the NeMo RL
# driver environment and expose the single-file LLaVA application modules.
export PATH="/opt/nemo_rl_venv/bin:$PATH"
export PYTHONPATH="$PWD/src:$PWD/examples/llava${PYTHONPATH:+:$PYTHONPATH}"
