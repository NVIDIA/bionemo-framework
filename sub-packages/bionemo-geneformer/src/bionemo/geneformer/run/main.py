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
import json
from typing import Optional

from bionemo.geneformer.run.config_models import (
    ExposedFineTuneSeqLenBioBertConfig,
    ExposedGeneformerPretrainConfig,
    GeneformerPretrainingDataConfig,
)
from bionemo.llm.run.config_models import MainConfig
from bionemo.llm.train import NsysConfig, train


def main():  # noqa: D103
    def parse_args():
        parser = argparse.ArgumentParser(description="Run Geneformer pretraining")
        parser.add_argument("--config", type=str, required=True, help="Path to the JSON configuration file")
        parser.add_argument(
            "--model-config-t",
            default=ExposedGeneformerPretrainConfig,
            required=False,
            help="fully resolvable python import path to the ModelConfig object. Builtin options are ExposedGeneformerPretrainConfig and ExposedFineTuneSeqLenBioBertConfig.",
        )
        parser.add_argument(
            "--data-config-t",
            default=GeneformerPretrainingDataConfig,
            required=False,
            help="fully resolvable python import path to the ModelConfig object.",
        )
        parser.add_argument(
            "--resume-if-exists",
            default=False,
            action="store_true",
            help="Resume training if a checkpoint exists that matches the current experiment configuration.",
        )

        # Debug options.
        parser.add_argument(
            "--nsys-profiling",
            action="store_true",
            default=False,
            help="Enable targeted `nsys` profiling on the training loop for a defined step range. To actually get profiling output you must run the whole program with `nsys`. For example: "
            " `nsys profile -s none -o output_report_name -t cuda,nvtx --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop  [regular python command here]`",
        )
        # start, end, rank
        parser.add_argument(
            "--nsys-start-step",
            type=int,
            required=False,
            default=0,
            help="Start nsys profiling after this step.",
        )
        parser.add_argument(
            "--nsys-end-step",
            type=int,
            required=False,
            help="End nsys profiling after this step.",
        )
        # rank as list of integers
        parser.add_argument(
            "--nsys-ranks",
            type=int,
            nargs="+",
            required=False,
            default=[0],
            help="Enable nsys profiling for these ranks.",
        )

        return parser.parse_args()

    def string_to_class(path: str):
        import importlib

        module_path, class_name = path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    def load_config(config_path: str, model_config_t: Optional[str], data_config_t: Optional[str]) -> MainConfig:
        with open(config_path, "r") as f:
            config_dict = json.load(f)

        # model/data_config_t is used to select the parser dynamically.
        if model_config_t is None or model_config_t == "ExposedGeneformerPretrainConfig":
            model_config_t = ExposedGeneformerPretrainConfig
        elif model_config_t == "ExposedFineTuneSeqLenBioBertConfig":
            # Hardcoded path for those who do not know the full path
            model_config_t = ExposedFineTuneSeqLenBioBertConfig
        elif isinstance(model_config_t, str):
            # We assume we get a string to some importable config... e.g. in the sub-package jensen, 'bionemo.jensen.configs.MyConfig'
            model_config_t = string_to_class(model_config_t)

        if data_config_t is None:
            data_config_t = GeneformerPretrainingDataConfig
        elif isinstance(data_config_t, str):
            data_config_t = string_to_class(data_config_t)
        return MainConfig[model_config_t, data_config_t](**config_dict)

    args = parse_args()
    config = load_config(args.config, args.model_config_t, args.data_config_t)

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
