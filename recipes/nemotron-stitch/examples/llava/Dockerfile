# syntax=docker/dockerfile:1
# LLaVA example image: the training/serving stack for examples/llava.
#
# The base mirrors NVIDIA's NeMo RL main CI build @ b03da0f4 (nesting
# AutoModel 1814c6c9). It is mirrored to Docker Hub because GitHub Actions
# runners cannot access the NVIDIA internal registry. All heavy dependencies —
# torch 2.11.0+cu130, transformers 5.12.1, vLLM 0.25.1, nemo-lens, and the
# worker venvs — ship in the base. This file only adds Stitch and the example.
#
# Pin the AMD64 image digest so a moving tag cannot change the tested stack.
#
#   docker build -f examples/llava/Dockerfile -t llava-example .
ARG BASE_IMAGE=svcbionemo023/bionemo-framework:nemo-rl-ci-b03da0f4-amd64@sha256:08fd971c29f8e76f0d589bc7195104231594e859707fda6fac82388770732140
FROM ${BASE_IMAGE}

# Use the prebuilt AutoModel environment for preparation, alignment/SFT, and
# the GRPO controller. NeMo RL selects its own worker environments as usual.
ENV PATH=/opt/ray_venvs/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2/bin:${PATH} \
    NEMO_RL_ROOT=/opt/nemo-rl \
    PYTHONPATH=/opt/llava-example \
    RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 \
    RAY_USAGE_STATS_ENABLED=0

# Install the package in every worker venv (Ray resolves workers in their own
# environments). The single-file example is importable through PYTHONPATH.
COPY pyproject.toml README.md /opt/nemotron-stitch/
COPY src /opt/nemotron-stitch/src
COPY examples/llava /opt/llava-example
# NeMo RL's config `defaults:` resolves relative to the config file; point it
# at the checkout.
RUN mkdir -p /opt/llava-example/third_party && ln -sfn /opt/nemo-rl /opt/llava-example/third_party/nemo-rl
RUN \
    for py in /opt/nemo_rl_venv/bin/python /opt/ray_venvs/*/bin/python; do \
        uv pip install --python "$py" --no-deps /opt/nemotron-stitch; \
    done

# Self-check the default training environment and direct vLLM registration.
RUN python -c "\
import nemotron_stitch, llava_example, llava_qwen, nemo_rl, nemo_automodel, nemo.lens; \
import transformer_engine.pytorch; \
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined; \
from causal_conv1d import causal_conv1d_fn; \
llava_qwen.build_moe_host_cls(); \
print('default AutoModel venv ok')" \
 && /opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker/bin/python -c "\
import llava_example, llava_qwen, vllm; \
assert vllm.__version__ == '0.25.1'; \
llava_example.register_vllm(); \
llava_qwen.register_vllm(); \
llava_qwen.register_moe_vllm(); \
print('vllm worker venv ok')"

WORKDIR /opt/llava-example
CMD ["sleep", "infinity"]
