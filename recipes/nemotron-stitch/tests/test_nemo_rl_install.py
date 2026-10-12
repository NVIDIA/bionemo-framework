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

import os
import subprocess
import sys
from pathlib import Path

import pytest

import nemotron_stitch

nemo_rl = pytest.importorskip("nemo_rl", reason="NeMo RL seam tests need the framework installed")


def test_encoder_policy_worker_accepts_keep_train_buffers_kwarg():
    # af0a11f's lm_policy calls prepare_for_lp_inference(keep_train_buffers=...);
    # e983576's signature lacks the kwarg. The encoder override must accept
    # and forward it on every supported pin.
    import inspect

    from nemotron_stitch.nemo_rl.policy import EncoderDTensorPolicyWorkerV2

    if EncoderDTensorPolicyWorkerV2 is None:
        pytest.skip("policy worker needs the framework environment")
    signature = inspect.signature(EncoderDTensorPolicyWorkerV2.prepare_for_lp_inference)
    assert any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()), (
        "prepare_for_lp_inference must accept **kwargs"
    )


def test_install_extensions_registers_and_is_idempotent():
    program = """
from nemotron_stitch.nemo_rl.runner import install_extensions
from nemo_rl.algorithms import grpo
from nemo_rl.distributed.ray_actor_environment_registry import ACTOR_ENVIRONMENT_REGISTRY
from nemo_rl.environments.utils import ENV_REGISTRY
from nemo_rl.models.generation.vllm import utils, vllm_generation
from nemo_rl.models.policy import lm_policy

# Native extension support: the worker FQNs travel in the config
# (policy.worker_extension_cls_fqn and generation.worker_extension_cls_fqn),
# so the bootstrap must only register runtimes — no Policy or resolver patch.
assert grpo.Policy.__module__ == 'nemo_rl.models.policy.lm_policy'
assert utils.resolve_generation_worker_cls.__module__ == 'nemo_rl.models.generation.vllm.utils'
assert vllm_generation.resolve_generation_worker_cls is utils.resolve_generation_worker_cls

env_name = 'test_encoder_env'
env_fqn = 'test_package.TestEnvironment'
install_extensions(environment_name=env_name, environment_fqn=env_fqn)
policy_runtime = ACTOR_ENVIRONMENT_REGISTRY['nemotron_stitch.nemo_rl.policy.EncoderDTensorPolicyWorkerV2']
vllm_runtime = ACTOR_ENVIRONMENT_REGISTRY['nemotron_stitch.nemo_rl.vllm_worker.EncoderVllmGenerationWorker']
install_extensions(environment_name=env_name, environment_fqn=env_fqn)
assert ACTOR_ENVIRONMENT_REGISTRY['nemotron_stitch.nemo_rl.policy.EncoderDTensorPolicyWorkerV2'] is policy_runtime
assert ACTOR_ENVIRONMENT_REGISTRY['nemotron_stitch.nemo_rl.vllm_worker.EncoderVllmGenerationWorker'] is vllm_runtime
assert ENV_REGISTRY[env_name] == {'actor_class_fqn': env_fqn}
assert env_fqn in ACTOR_ENVIRONMENT_REGISTRY
"""
    # The fresh subprocess needs the package on its path explicitly: pytest's
    # own sys.path munging does not propagate, and the package is not
    # pip-installed in the training image.
    src_root = Path(nemotron_stitch.__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": f"{src_root}:{os.environ.get('PYTHONPATH', '')}"}
    subprocess.run([sys.executable, "-c", program], check=True, env=env)


def test_config_worker_extension_fqns_resolve_after_install_extensions():
    program = """
from nemotron_stitch.nemo_rl.runner import install_extensions
from nemo_rl.distributed.ray_actor_environment_registry import get_actor_python_env

# NeMo RL fails closed on an unregistered extension FQN, and the config keys
# resolve through the same registry the bootstrap fills.
for fqn in (
    'nemotron_stitch.nemo_rl.policy.EncoderDTensorPolicyWorkerV2',
    'nemotron_stitch.nemo_rl.vllm_worker.EncoderVllmGenerationWorker',
):
    try:
        get_actor_python_env(fqn)
    except ValueError:
        pass
    else:
        raise AssertionError('unregistered FQN should fail closed')

install_extensions()
for fqn in (
    'nemotron_stitch.nemo_rl.policy.EncoderDTensorPolicyWorkerV2',
    'nemotron_stitch.nemo_rl.vllm_worker.EncoderVllmGenerationWorker',
):
    assert get_actor_python_env(fqn)
"""
    src_root = Path(nemotron_stitch.__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": f"{src_root}:{os.environ.get('PYTHONPATH', '')}"}
    subprocess.run([sys.executable, "-c", program], check=True, env=env)


def test_install_extensions_consumer_workers_register_their_runtimes():
    program = """
from nemotron_stitch.nemo_rl.runner import install_extensions
from nemo_rl.distributed.ray_actor_environment_registry import ACTOR_ENVIRONMENT_REGISTRY
from nemo_rl.distributed.ray_actor_environment_registry import get_actor_python_env
from nemo_rl.environments.utils import ENV_REGISTRY

env_name = 'test_module_env'
env_fqn = 'test_package.TestEnvironment'
install_extensions(
    environment_name=env_name,
    environment_fqn=env_fqn,
    policy_worker_fqn='consumer_pkg.policy.ConsumerPolicyWorker',
    generation_worker_fqn='consumer_pkg.generation.ConsumerVllmWorker',
)
assert get_actor_python_env('consumer_pkg.policy.ConsumerPolicyWorker')
assert get_actor_python_env('consumer_pkg.generation.ConsumerVllmWorker')
assert ACTOR_ENVIRONMENT_REGISTRY['consumer_pkg.policy.ConsumerPolicyWorker']
assert ACTOR_ENVIRONMENT_REGISTRY['consumer_pkg.generation.ConsumerVllmWorker']
assert ENV_REGISTRY[env_name] == {'actor_class_fqn': env_fqn}
# Idempotent re-entry.
install_extensions(
    environment_name=env_name,
    environment_fqn=env_fqn,
    policy_worker_fqn='consumer_pkg.policy.ConsumerPolicyWorker',
    generation_worker_fqn='consumer_pkg.generation.ConsumerVllmWorker',
)
"""
    src_root = Path(nemotron_stitch.__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": f"{src_root}:{os.environ.get('PYTHONPATH', '')}"}
    subprocess.run([sys.executable, "-c", program], check=True, env=env)


def test_install_extensions_rejects_a_conflicting_runtime():
    program = """
from nemotron_stitch.nemo_rl.runner import install_extensions

install_extensions(policy_worker_fqn='consumer_pkg.policy.ConsumerPolicyWorker')
try:
    install_extensions(
        policy_worker_fqn='consumer_pkg.policy.ConsumerPolicyWorker',
        policy_worker_python='/some/other/python',
    )
except RuntimeError as error:
    assert 'registry conflict' in str(error)
else:
    raise AssertionError('a conflicting runtime should fail closed')
"""
    src_root = Path(nemotron_stitch.__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": f"{src_root}:{os.environ.get('PYTHONPATH', '')}"}
    subprocess.run([sys.executable, "-c", program], check=True, env=env)
