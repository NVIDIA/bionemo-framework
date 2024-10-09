# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import random
import string

import pytest
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from bionemo.utils.hydra import load_model_config


def generate_random_key(num_keys: int):
    return "".join(random.choice(string.ascii_lowercase) for _ in range(num_keys))


def generate_random_value(num_keys: int):
    return random.choice(
        [random.randint(1, 100), random.uniform(0.0, 1.0), "".join(random.choices(string.ascii_letters, k=num_keys))]
    )


def generate_random_omegaconf_dict(depth: int, num_keys: int):
    if depth == 0:
        return generate_random_value(num_keys=num_keys)
    return OmegaConf.create(
        {
            generate_random_key(num_keys=num_keys): generate_random_omegaconf_dict(depth=depth - 1, num_keys=num_keys)
            for _ in range(num_keys)
        }
    )


@pytest.fixture(scope="module")
def config():
    seed_everything(42)
    return generate_random_omegaconf_dict(depth=2, num_keys=3)


def test_load_model_config(tmp_path, config):
    config_name = "config"
    OmegaConf.save(config, os.path.join(tmp_path, config_name + ".yaml"))

    config_loaded = load_model_config(config_name=config_name, config_path=str(tmp_path))
    assert config_loaded == config
