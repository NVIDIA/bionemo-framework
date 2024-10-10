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


import hashlib
import json
import os
import pathlib
import shutil
from collections import OrderedDict
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import apex
import numpy as np
import torch
import yaml
from hydra.core.config_search_path import ConfigSearchPath
from hydra.core.plugins import Plugins
from hydra.plugins.search_path_plugin import SearchPathPlugin
from megatron.core.parallel_state import destroy_model_parallel
from nemo.utils import logging
from omegaconf import OmegaConf, open_dict
from omegaconf.dictconfig import DictConfig
from pytorch_lightning import seed_everything


class BioNemoSearchPathConfig(SearchPathPlugin):
    def __init__(self, prepend_config_dir: Optional[str] = None, append_config_dir: Optional[str] = None) -> None:
        # TODO  allow lists for config directories
        self.prepend_config_dir = prepend_config_dir
        self.append_config_dir = append_config_dir

    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        """Manipulate existing search path"""
        if self.prepend_config_dir:
            search_path.prepend(provider="bionemo-searchpath-plugin", path=f"file://{self.prepend_config_dir}")

        if self.append_config_dir:
            search_path.append(provider="bionemo-searchpath-plugin", path=f"file://{self.append_config_dir}")


def register_searchpath_config_plugin(plugin: SearchPathPlugin) -> None:
    """Call this function before invoking @hydra.main"""
    Plugins.instance().register(plugin)


def update_relative_config_dir(prepend_config_dir: pathlib.Path, test_file_dir: pathlib.Path):
    """Update relative config directory for hydra

    Args:
        prepend_config_dir (str): directory for additional configs relative to python test file

    Returns:
        str: directory for additional configs relative to test execution directory
    """
    execution_dir = pathlib.Path(os.getcwd()).resolve()  # directory tests are executed from

    # absolute path for the config directory
    prepend_config_dir_absolute_path = os.path.join(test_file_dir, prepend_config_dir)
    prepend_config_dir_absolute_path = os.path.abspath(prepend_config_dir_absolute_path)

    # relative path for config directory vs execution directory
    updated_prepend_config_dir = os.path.relpath(prepend_config_dir_absolute_path, execution_dir)
    return updated_prepend_config_dir


def resolve_cfg(cfg: DictConfig):
    """Resolve hydra config to be pickleable and comparable

    Args:
        cfg (DictConfig): Input configuration
    """
    OmegaConf.resolve(cfg)
    with open_dict(cfg):
        # Check if the key `exp_manager` and the nested key `wandb_logger_kwargs` exist
        if "exp_manager" in cfg and "wandb_logger_kwargs" in cfg.exp_manager:
            cfg.exp_manager.wandb_logger_kwargs.pop("notes", None)
    cfg_str = yaml.load(OmegaConf.to_yaml(cfg), Loader=yaml.BaseLoader)
    return cfg_str


def clean_directory(directory):
    """Remove existing files in a directory

    Args:
        directory (str): path to directory to be cleaned
    """
    directory = pathlib.Path(directory)
    if os.path.exists(directory):
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)


def load_expected_training_results(results_comparison_dir: str, correct_results: str):
    """Loads a JSON file containing expected training results.

    Args:
        results_comparison_dir (str): Directory where the json file with config is located.
        correct_config (str): Name of the json file holding the configuration.

    Returns:
        dict: expected training results.
    """
    results_path = os.path.join(results_comparison_dir, correct_results)

    with open(results_path, "r") as fh:
        expected_results = json.load(fh, object_pairs_hook=OrderedDict)

    return expected_results


def save_expected_training_results(results_comparison_dir: str, correct_results: str, expected_results: dict) -> None:
    """
    Args:
        results_comparison_dir (str): Directory where the json file with config is located
        correct_config (str): Name of the json file holding the configuration
        expected_results (dict): expected training results

    Returns:
        None
    """
    results_path = os.path.join(results_comparison_dir, correct_results)
    logging.info(f"Saving expected training results to {results_path}")
    with open(results_path, "w") as fh:
        json.dump(expected_results, fh, indent=4, sort_keys=True)


def check_expected_training_results(
    trainer_results: dict,
    expected_results: dict,
    rel_tol: float = 0,
    test_rel_tol_overrides: Dict[str, float] = {},
    err_msg: str = "",
):
    """Compare expected training results

    Args:
        trainer (dict): PyTorch Lightning Trainer results
        expected_results (dict): expected training metrics
        rel_tol float: comparison relative tolerance, defaults to 0 which means that results are expected to be identical
        test_rel_tol_overrides (Dict[str, float]): Tolerance for override keys that match `override_key in key`.
    """
    mismatched_results = []
    assert 0 <= rel_tol <= 1, "Relative tolerance has to be in [0, 1]"
    for key in expected_results:
        if "timing" in key:
            # stats with this postfix are very hardware dependent, skipping them
            continue
        expected_value = expected_results[key]
        actual_value = trainer_results[key]
        if isinstance(actual_value, torch.Tensor):
            actual_value = actual_value.cpu().numpy().item()
        this_rel_tol = rel_tol
        for override_key, override_tol in test_rel_tol_overrides.items():
            if override_key in key:
                this_rel_tol = override_tol
        tol_key = abs(this_rel_tol * expected_value)  # absolute tolerance from expected value
        if not np.allclose(expected_value, actual_value, atol=tol_key):
            mismatched_results.append(
                f"Expected {key} = {expected_value}, got {actual_value} not within abs tol of {tol_key} from rel tol of {this_rel_tol}."
            )

    assert len(mismatched_results) == 0, f"Training results mismatched: {mismatched_results}{err_msg}"


def check_model_exists(model_fname: str):
    """
    Check if model exists and raise error if not.
    """
    assert os.path.exists(
        model_fname
    ), f"Model file not found model_fname = '{model_fname}', please use 'launch.sh download' to download the model."


def get_directory_hash(directory):
    """Calculate hash of all files in a directory"""
    md5_hash = hashlib.md5()
    for root, _, files in os.walk(directory):
        for name in sorted(files):
            filepath = os.path.join(root, name)
            with open(filepath, "rb") as fh:
                data = fh.read()
                md5_hash.update(data)
    return md5_hash.hexdigest()


# TODO(@jomitchell) Write a unit test somewhere for this helper function.
def list_to_tensor(data_list: List[Union[float, int]]) -> torch.Tensor:
    """Recursively convert a multi-dimensional list to a torch tensor."""

    # If the current item is not a list, return it wrapped in a tensor
    if not isinstance(data_list, list):
        return torch.tensor(data_list)

    # Convert the first level of the list to tensors or lists
    converted = [list_to_tensor(item) for item in data_list]

    # If all converted items are tensors and have the same shape, stack them
    if all(isinstance(item, torch.Tensor) for item in converted) and len({item.shape for item in converted}) == 1:
        return torch.stack(converted)

    return converted


def reset_microbatch_calculator():
    """
    Resets _GLOBAL_NUM_MICROBATCHES_CALCULATOR in apex which is used in NeMo to initilised model parallel in
    nemo.collections.nlp.modules.common.megatron.megatron_init.initialize_model_parallel_for_nemo
    """
    apex.transformer.pipeline_parallel.utils._GLOBAL_NUM_MICROBATCHES_CALCULATOR = None


def teardown_apex_megatron_cuda():
    """
    Cleans GPU allocation and model and data parallel settings after usage of a model:
    - sets the global variables related to model and data parallelism to None in Apex and Megatron:.
    - releases all unoccupied cached GPU memory currently held by the caching CUDA allocator, see torch.cuda.empty_cache
    """
    torch.cuda.empty_cache()
    reset_microbatch_calculator()
    destroy_model_parallel()


@contextmanager
def distributed_model_parallel_state():
    """Context manager for handling creating and cleaning up distributed model parallel state for tests.
    Use like:
    with distributed_model_parallel_state():
        # your test code here
    # After the block your state is cleaned up.
    """
    from bionemo.model.utils import initialize_distributed_parallel_state  # here to avoid circular import

    try:
        teardown_apex_megatron_cuda()
        initialize_distributed_parallel_state()
        yield
    finally:
        teardown_apex_megatron_cuda()


@dataclass
class Deterministic(AbstractContextManager):
    cudnn_deterministic: Optional[bool] = None
    cudnn_benchmark: Optional[bool] = None
    cudnn_deterministic: Optional[bool] = None
    cuda_matmul_allow_tf32: Optional[bool] = None
    cudnn_enabled: Optional[bool] = None

    def __enter__(self):
        self.cudnn_deterministic = torch.backends.cudnn.deterministic
        self.cudnn_benchmark = torch.backends.cudnn.benchmark
        self.cuda_matmul_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        self.cudnn_enabled = torch.backends.cudnn.enabled
        seed_everything(123)
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.allow_tf32 = False
        torch.backends.cudnn.enabled = False

    def __exit__(self, *exit_state):
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = self.cudnn_deterministic
        torch.backends.cudnn.benchmark = self.cudnn_benchmark
        torch.backends.cuda.matmul.allow_tf32 = self.cuda_matmul_allow_tf32
        torch.backends.cuda.allow_tf32 = True
        torch.backends.cudnn.enabled = self.cudnn_enabled
