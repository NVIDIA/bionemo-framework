# syntax=docker/dockerfile:1
# Shared runtime for LLaVA, genetics, and chemistry. The AMD64 NGC mirror
# carries NeMo RL 9a842983, AutoModel b916107a, and vLLM 0.29.0.
ARG BASE_IMAGE=svcbionemo023/bionemo-framework:nemo-rl-ci-9a842983-amd64@sha256:b246d90d969277441bbce418bd895c3f9a4ef5e0ab47587b6f4e0c91ad84af48
FROM ${BASE_IMAGE} AS runtime

ENV PATH=/opt/ray_venvs/nemo_rl.models.policy.workers.automodel_policy_worker.AutomodelPolicyWorker/bin:${PATH} \
    NEMO_RL_ROOT=/opt/nemo-rl \
    PYTHONPATH=/opt/stitch/src:/opt/stitch/examples/llava:/opt/stitch/examples/genetics:/opt/stitch/examples/chemistry \
    RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 \
    RAY_USAGE_STATS_ENABLED=0 \
    PYTHONDONTWRITEBYTECODE=1

COPY pyproject.toml README.md /opt/stitch/
COPY src /opt/stitch/src
COPY examples /opt/stitch/examples
COPY tests /opt/stitch/tests
# Supply the upstream client as a named context, without embedding credentials.
COPY --from=perception /clients/python /opt/perception-client
RUN set -eu; \
    mkdir -p /opt/stitch/examples/llava/third_party; \
    ln -s /opt/nemo-rl /opt/stitch/examples/llava/third_party/nemo-rl; \
    for interpreter in /opt/nemo_rl_venv/bin/python /opt/ray_venvs/*/bin/python; do \
      uv pip install --python "$interpreter" --no-deps /opt/stitch \
        /opt/stitch/examples/llava /opt/stitch/examples/genetics /opt/stitch/examples/chemistry; \
      uv pip install --python "$interpreter" /opt/perception-client pytest 'rdkit==2025.9.1'; \
    done

# Same fused-loss implementation as the production gene and chemistry recipes.
RUN uv pip install --python \
      /opt/ray_venvs/nemo_rl.models.policy.workers.automodel_policy_worker.AutomodelPolicyWorker/bin/python \
      --no-deps \
      'cut-cross-entropy @ https://github.com/apple/ml-cross-entropy/archive/87a86aba72cfd2f0d8abecaf81c13c4528ea07d8.tar.gz'

# Native CUDA imports and host construction are checked in a GPU runtime.
# Entry points must exist in both synchronous and asynchronous vLLM workers.
RUN set -eu; \
    python -c 'import nemotron_stitch, llava_example, llava_qwen, genetics_example, chemistry_example, bionemo_perception'; \
    for interpreter in /opt/ray_venvs/*vllm*/bin/python; do \
      "$interpreter" -c 'from importlib.metadata import entry_points; \
        plugins = {entry.name: entry for entry in entry_points(group="vllm.general_plugins")}; \
        names = {"llava_example", "llava_qwen", "llava_qwen_moe", "stitch_genetics", "stitch_chemistry"}; \
        assert names <= plugins.keys(); \
        [plugins[name].load()() for name in sorted(names)]'; \
    done

WORKDIR /opt/stitch/examples
CMD ["bash"]

# Public Super snapshots require Transformers 5.18 and a compatible AutoModel
# base. This target changes Transformers only; it does not upgrade AutoModel.
FROM runtime AS super
RUN set -eu; for interpreter in /opt/nemo_rl_venv/bin/python /opt/ray_venvs/*/bin/python; do \
      uv pip install --python "$interpreter" 'transformers==5.18.0'; \
    done

FROM runtime AS default
