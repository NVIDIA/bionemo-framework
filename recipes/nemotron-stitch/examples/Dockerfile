# syntax=docker/dockerfile:1
ARG BASE_IMAGE=svcbionemo023/bionemo-framework:nemo-rl-ci-b03da0f4-amd64@sha256:08fd971c29f8e76f0d589bc7195104231594e859707fda6fac82388770732140
FROM ${BASE_IMAGE} AS runtime

ENV PATH=/opt/ray_venvs/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2/bin:${PATH} \
    NEMO_RL_ROOT=/opt/nemo-rl \
    PYTHONPATH=/opt/stitch/src:/opt/stitch/examples/genetics:/opt/stitch/examples/chemistry \
    RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 \
    RAY_USAGE_STATS_ENABLED=0 \
    PYTHONDONTWRITEBYTECODE=1

COPY pyproject.toml README.md /opt/stitch/
COPY src /opt/stitch/src
COPY examples/README.md /opt/stitch/examples/README.md
COPY examples/genetics /opt/stitch/examples/genetics
COPY examples/chemistry /opt/stitch/examples/chemistry
COPY tests /opt/stitch/tests
# Supply the upstream client as a named context, without embedding credentials.
COPY --from=perception /clients/python /opt/perception-client
RUN set -eu; for interpreter in /opt/nemo_rl_venv/bin/python /opt/ray_venvs/*/bin/python; do \
      uv pip install --python "$interpreter" --no-deps /opt/stitch \
        /opt/stitch/examples/genetics /opt/stitch/examples/chemistry; \
      uv pip install --python "$interpreter" /opt/perception-client pytest; \
    done

# Same fused-loss implementation as the production gene and chemistry recipes.
RUN uv pip install --python \
      /opt/ray_venvs/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2/bin/python \
      --no-deps \
      'cut-cross-entropy @ https://github.com/apple/ml-cross-entropy/archive/87a86aba72cfd2f0d8abecaf81c13c4528ea07d8.tar.gz'

# Canonical atom/bond indexing is preparation-only; workers consume stored indices.
RUN set -eu; for interpreter in /opt/nemo_rl_venv/bin/python \
      /opt/ray_venvs/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2/bin/python; do \
      uv pip install --python "$interpreter" 'rdkit==2025.9.1'; \
    done

WORKDIR /opt/stitch/examples/genetics
CMD ["bash"]

# Select --target super with the Super-compatible BASE_IMAGE in genetics/README.md.
# Its exported configs require Transformers 5.18 in both training and generation.
FROM runtime AS super
RUN set -eu; for interpreter in /opt/nemo_rl_venv/bin/python \
      /opt/ray_venvs/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2/bin/python \
      /opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker/bin/python; do \
      uv pip install --python "$interpreter" 'transformers==5.18.0'; \
    done

FROM runtime AS default
