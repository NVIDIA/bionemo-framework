# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Owned NeMo RL bootstrap for encoder GRPO.

Workers are named through NeMo RL's config-driven extension keys
(``policy.worker_extension_cls_fqn`` and ``generation.worker_extension_cls_fqn``;
provided by RL#3809). This bootstrap only registers the runtime
environment each worker FQN resolves to — upstream validates that registration
before allocating workers and raises on an unregistered FQN, so a config key
without a matching registration fails closed instead of running the worker on
the wrong interpreter.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

_DEFAULT_POLICY_FQN = "nemotron_stitch.nemo_rl.policy.EncoderDTensorPolicyWorkerV2"
_DEFAULT_VLLM_FQN = "nemotron_stitch.nemo_rl.vllm_worker.EncoderVllmGenerationWorker"
# NeMo RL's stock worker FQNs: the frozen worker venvs are keyed by them.
_STOCK_POLICY_FQN = "nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2"
_STOCK_VLLM_FQN = "nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker"


def _register_exact(registry: dict, key: str, value) -> None:
    current = registry.get(key)
    if current is None:
        registry[key] = value
    elif current != value:
        raise RuntimeError(f"NeMo RL registry conflict for {key!r}: {current!r} != {value!r}")


def install_extensions(
    *,
    environment_name: str | None = None,
    environment_fqn: str | None = None,
    policy_worker_fqn: str | None = None,
    generation_worker_fqn: str | None = None,
    policy_worker_python: str | None = None,
    generation_worker_python: str | None = None,
) -> None:
    """Register the encoder environment and worker runtimes with NeMo RL.

    ``policy_worker_fqn``/``generation_worker_fqn`` default to the package's
    one-rank sidecar workers; they must match the FQNs the config names in
    ``policy.worker_extension_cls_fqn`` and
    ``generation.worker_extension_cls_fqn``. A consumer whose workers own the
    multi-rank protocol themselves passes its own FQNs (topology validation
    then belongs to that worker) and sets the same keys in its config.
    """
    policy_fqn = policy_worker_fqn or _DEFAULT_POLICY_FQN
    vllm_fqn = generation_worker_fqn or _DEFAULT_VLLM_FQN

    from nemo_rl.distributed.ray_actor_environment_registry import ACTOR_ENVIRONMENT_REGISTRY
    from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
    from nemo_rl.environments.utils import ENV_REGISTRY

    unified_runtime = os.environ.get("NEMO_RL_PY_EXECUTABLES_SYSTEM", "0") == "1"
    venv_root = Path(os.environ.get("NEMO_RL_VENV_DIR", "/opt/ray_venvs"))

    def frozen_worker_python(upstream_fqn: str, fallback: str) -> str:
        candidate = venv_root / upstream_fqn / "bin" / "python"
        return str(candidate) if candidate.is_file() else fallback

    if (environment_name is None) != (environment_fqn is None):
        raise ValueError("environment_name and environment_fqn must be configured together")
    if environment_name is not None and environment_fqn is not None:
        _register_exact(ENV_REGISTRY, environment_name, {"actor_class_fqn": environment_fqn})
        _register_exact(ACTOR_ENVIRONMENT_REGISTRY, environment_fqn, PY_EXECUTABLES.SYSTEM)
    policy_python = policy_worker_python or (
        PY_EXECUTABLES.SYSTEM if unified_runtime else frozen_worker_python(_STOCK_POLICY_FQN, PY_EXECUTABLES.AUTOMODEL)
    )
    vllm_python = generation_worker_python or (
        PY_EXECUTABLES.SYSTEM if unified_runtime else frozen_worker_python(_STOCK_VLLM_FQN, PY_EXECUTABLES.VLLM)
    )
    _register_exact(ACTOR_ENVIRONMENT_REGISTRY, policy_fqn, policy_python)
    _register_exact(ACTOR_ENVIRONMENT_REGISTRY, vllm_fqn, vllm_python)


def main() -> int:
    install_extensions(
        environment_name=os.environ.get("ENCODER_ENVIRONMENT_NAME"),
        environment_fqn=os.environ.get("ENCODER_ENVIRONMENT_FQN"),
        policy_worker_fqn=os.environ.get("ENCODER_POLICY_WORKER_FQN"),
        generation_worker_fqn=os.environ.get("ENCODER_GENERATION_WORKER_FQN"),
        policy_worker_python=os.environ.get("ENCODER_POLICY_WORKER_PYTHON"),
        generation_worker_python=os.environ.get("ENCODER_GENERATION_WORKER_PYTHON"),
    )
    nemo_root = Path(os.environ.get("NEMO_RL_ROOT", "/opt/nemo-rl"))
    launcher = nemo_root / "examples" / "run_grpo.py"
    if not launcher.is_file():
        raise FileNotFoundError(f"NeMo RL launcher not found: {launcher}")
    sys.argv = [str(launcher), *sys.argv[1:]]
    runpy.run_path(str(launcher), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
