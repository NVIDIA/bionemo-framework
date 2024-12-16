from dataclasses import dataclass, field
from typing import Any, Type, List
import argparse

from bionemo.geneformer.run.config_models import ExposedGeneformerPretrainConfig, GeneformerPretrainingDataConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Run Geneformer pretraining")
    parser.add_argument("--config", type=str, required=True, help="Path to the JSON configuration file")
    parser.add_argument(
        "--model-config-cls",
        default=ExposedGeneformerPretrainConfig,
        required=False,
        help="fully resolvable python import path to the ModelConfig class. Builtin options are ExposedGeneformerPretrainConfig and ExposedFineTuneSeqLenBioBertConfig.",
    )
    parser.add_argument(
        "--data-config-cls",
        default=GeneformerPretrainingDataConfig,
        required=False,
        help="fully resolvable python import path to the class.",
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

@dataclass
class NRArgs:
    config_dict: dict
    model_config_cls: Type
    data_config_cls: Type
    resume_if_exists: bool
    nsys_profiling: bool
    nsys_start_step: int
    nsys_end_step: int
    nsys_ranks: list[int] = field(default_factory=lambda: [0])
