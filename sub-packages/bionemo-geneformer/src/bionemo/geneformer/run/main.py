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


import argparse
from dataclasses import dataclass, field
from typing import Optional, Type
from nemo_run import Config, autoconvert
import yaml

from bionemo.geneformer.run.config_models import (
    ExposedFineTuneSeqLenBioBertConfig,
    ExposedGeneformerPretrainConfig,
    GeneformerPretrainingDataConfig,
)
from bionemo.llm.run.config_models import MainConfig
from bionemo.llm.train import NsysConfig, train
from bionemo.geneformer.run.argument_parser import parse_args
from bionemo.geneformer.run.nemo_run import NRArgs

def args_to_args_dict(args) -> dict:
    '''Transforms the ArgumentParser namespace into a dictionary with one modification, `config`, which accepts a file path,
    is transformed into a serialized dictionary. This allows us to defer parsing until the job is scheduled.

    Arguments:
        args - argparse namesspace arguments, aquired from parser.parse_args()

    Returns:
        Dictionary of arguments with `config` replaced by `config_dict`.
    '''
    args_dict = vars(args)
    config_path = args_dict.pop("config")
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    args_dict['config_dict'] = config_dict
    return args_dict

def load_config_from_file(config_path: str, model_config_cls: Optional[str], data_config_cls: Optional[str]) -> MainConfig:
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return load_config(config_dict, model_config_cls=model_config_cls, data_config_cls=data_config_cls)

def load_config(config_dict: dict, model_config_cls: Optional[str], data_config_cls: Optional[str]) -> MainConfig:
    # model/data_config_cls is used to select the parser dynamically.
    if model_config_cls is None or model_config_cls == "ExposedGeneformerPretrainConfig":
        model_config_cls = ExposedGeneformerPretrainConfig
    elif model_config_cls == "ExposedFineTuneSeqLenBioBertConfig":
        # Hardcoded path for those who do not know the full path
        model_config_cls = ExposedFineTuneSeqLenBioBertConfig
    elif isinstance(model_config_cls, str):
        # We assume we get a string to some importable config... e.g. in the sub-package jensen, 'bionemo.jensen.configs.MyConfig'
        model_config_cls = string_to_class(model_config_cls)

    if data_config_cls is None:
        data_config_cls = GeneformerPretrainingDataConfig
    elif isinstance(data_config_cls, str):
        data_config_cls = string_to_class(data_config_cls)
    return MainConfig[model_config_cls, data_config_cls](**config_dict)

def string_to_class(path: str):
    import importlib

    module_path, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def main():  # noqa: D103
    args = parse_args()
    config = load_config_from_file(args.config, args.model_config_cls, args.data_config_cls)

    if args.nsys_profiling:
        nsys_config = NsysConfig(
            start_step=args.nsys_start_step,
            end_step=args.nsys_end_step,
            ranks=args.nsys_ranks,
        )
    else:
        nsys_config = None

    train(
        bionemo_exposed_model_config=config.bionemo_model_config,
        data_config=config.data_config,
        parallel_config=config.parallel_config,
        training_config=config.training_config,
        optim_config=config.optim_config,
        experiment_config=config.experiment_config,
        wandb_config=config.wandb_config,
        resume_if_exists=args.resume_if_exists,
        nsys_config=nsys_config,
    )


if __name__ == "__main__":
    main()
