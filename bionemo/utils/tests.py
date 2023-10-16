import hashlib
import json
import os
import pathlib
import shutil
from collections import OrderedDict
from typing import List, Optional, Union

import numpy as np
import torch
import yaml
from hydra.core.config_search_path import ConfigSearchPath
from hydra.core.plugins import Plugins
from hydra.plugins.search_path_plugin import SearchPathPlugin
from nemo.utils import logging
from omegaconf import OmegaConf, open_dict
from omegaconf.dictconfig import DictConfig


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
        cfg.exp_manager.wandb_logger_kwargs.pop('notes', None)
    cfg_str = yaml.load(OmegaConf.to_yaml(cfg), Loader=yaml.BaseLoader)
    return cfg_str


def save_cfg_to_json(cfg: dict, results_comparison_dir: str, correct_config: str):
    """Dump hydra configuration to JSON file

    Args:
        cfg (dict): configuration parameters
        results_comparison_dir (str): Directory to store the JSON file with config
        correct_config (str): Name of the JSON file holding the configuration
    """
    cfg_str = resolve_cfg(cfg)
    output_file = os.path.join(results_comparison_dir, correct_config)
    with open(output_file, 'w') as fh:
        fh.write(json.dumps(cfg_str))


def load_cfg_json(results_comparison_dir: str, correct_config: str):
    """Load hydra configuration from a json file for comparison

    Args:
        results_comparison_dir (str): Directory where the json file with config is located
        correct_config (str): Name of the json file holding the configuration

    Returns:
        dict: expected comparison configuration
    """
    config_file = os.path.join(results_comparison_dir, correct_config)
    with open(config_file, 'r') as fh:
        cfg_dict = json.load(fh)
    return cfg_dict


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

    with open(results_path, 'r') as fh:
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
    logging.info(f'Saving expected training results to {results_path}')
    with open(results_path, 'w') as fh:
        json.dump(expected_results, fh, indent=4, sort_keys=True)


def check_expected_training_results(
    trainer_results: dict, expected_results: dict, tol: float = 1.0e-4, err_msg: str = ""
):
    """Compare expected training results

    Args:
        trainer (dict): PyTorch Lightning Trainer results
        expected_results (dict): expected training metrics
        tol (float, optional): float comparison tolerance. Defaults to 1.0e-4.
    """
    mismatched_results = []
    for key in expected_results:
        expected_value = expected_results[key]
        actual_value = trainer_results[key].cpu().numpy().item()
        if not np.allclose(expected_value, actual_value, atol=tol):
            mismatched_results.append(f"Expected {key} = {expected_value}, got {actual_value}.")

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
            with open(filepath, 'rb') as fh:
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
