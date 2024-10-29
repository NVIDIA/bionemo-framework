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
from pathlib import Path
from typing import Sequence, get_args

import torch
from nemo import lightning as nl

from bionemo.core.utils.dtypes import PrecisionTypes
from bionemo.esm2.api import ESM2Config
from bionemo.esm2.data.tokenizer import get_tokenizer
from bionemo.esm2.model.finetune.datamodule import ESM2FineTuneDataModule, InMemoryCSVDataset
from bionemo.llm.model.biobert.lightning import biobert_lightning_module
from bionemo.llm.utils.datamodule_utils import infer_global_batch_size


__all__: Sequence[str] = ("infer_model",)


def infer_model(
    data_path: Path,
    checkpoint_path: Path,
    results_path: Path,
    include_hiddens: bool = False,
    include_embeddings: bool = False,
    micro_batch_size: int = 64,
    accumulate_grad_batches: int = 1,
    precision: PrecisionTypes = "bf16-mixed",
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    devices: int = 1,
    num_nodes: int = 1,
) -> None:
    """Runs inference on a BioNeMo ESM2 model using PyTorch Lightning.

    Args:
        data_path (Path): Path to the input data.
        checkpoint_path (Path): Path to the model checkpoint.
        results_path (Path): Path to save the inference results.
        include_hiddens (bool, optional): Whether to include hidden states in the output. Defaults to False.
        include_embeddings (bool, optional): Whether to include embeddings in the output. Defaults to False.
        micro_batch_size (int, optional): Micro batch size for inference. Defaults to 64.
        accumulate_grad_batches (int, optional): Number of batches to accumulate gradients (not applicable for inference). Defaults to 1.
        precision (PrecisionTypes, optional): Precision type for inference. Defaults to "bf16-mixed".
        tensor_model_parallel_size (int, optional): Tensor model parallel size for distributed inference. Defaults to 1.
        pipeline_model_parallel_size (int, optional): Pipeline model parallel size for distributed inference. Defaults to 1.
        devices (int, optional): Number of devices to use for inference. Defaults to 1.
        num_nodes (int, optional): Number of nodes to use for distributed inference. Defaults to 1.
    """
    # Setup the strategy and trainer
    global_batch_size = infer_global_batch_size(
        micro_batch_size=micro_batch_size,
        num_nodes=num_nodes,
        devices=devices,
        accumulate_grad_batches=accumulate_grad_batches,
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
    )

    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        ddp="megatron",
        find_unused_parameters=True,
    )

    trainer = nl.Trainer(
        accelerator="gpu",
        devices=devices,
        strategy=strategy,
        num_nodes=num_nodes,
        plugins=nl.MegatronMixedPrecision(precision=precision),
    )

    dataset = InMemoryCSVDataset(data_path=data_path)
    data_module = ESM2FineTuneDataModule(
        predict_dataset=dataset, micro_batch_size=micro_batch_size, global_batch_size=global_batch_size
    )

    config = ESM2Config(
        include_hiddens=include_hiddens,
        include_embeddings=include_embeddings,
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        initial_ckpt_path=checkpoint_path,
    )

    tokenizer = get_tokenizer()
    module = biobert_lightning_module(config=config, tokenizer=tokenizer)
    results = trainer.predict(module, datamodule=data_module)

    assert isinstance(results, list) and len(results) == 1
    results_dict = results[0]
    non_none_keys = [key for key, val in results_dict.items() if val is not None]
    print(f"Writing output {str(non_none_keys)} into {results_path}")
    torch.save(results[0], results_path)


parser = argparse.ArgumentParser(description="Infer ESM2.")
parser.add_argument(
    "--checkpoint-path",
    type=Path,
    required=True,
    help="Path to the ESM2 pretrained checkpoint",
)
parser.add_argument(
    "--data-path",
    type=Path,
    required=True,
    help="Path to the CSV file containing sequences and label columns",
)
parser.add_argument("--results-path", type=Path, required=True, help="Path to the results file.")

parser.add_argument(
    "--precision",
    type=str,
    choices=get_args(PrecisionTypes),
    required=False,
    default="bf16-mixed",
    help="Precision type to use for training.",
)
parser.add_argument(
    "--num-gpus",
    type=int,
    required=False,
    default=1,
    help="Number of GPUs to use for training. Default is 1.",
)
parser.add_argument(
    "--num-nodes",
    type=int,
    required=False,
    default=1,
    help="Number of nodes to use for training. Default is 1.",
)
parser.add_argument(
    "--micro-batch-size",
    type=int,
    required=False,
    default=2,
    help="Micro-batch size. Global batch size is inferred from this.",
)
parser.add_argument(
    "--pipeline-model-parallel-size",
    type=int,
    required=False,
    default=1,
    help="Pipeline model parallel size. Default is 1.",
)
parser.add_argument(
    "--tensor-model-parallel-size",
    type=int,
    required=False,
    default=1,
    help="Tensor model parallel size. Default is 1.",
)
parser.add_argument(
    "--include-hiddens",
    type=bool,
    required=False,
    default=False,
    help="Include hiddens in output of inference",
)
parser.add_argument(
    "--include-embeddings",
    type=bool,
    required=False,
    default=False,
    help="Include embeddings in output of inference",
)
parser.add_argument(
    "--accumulate-grad-batches",
    type=int,
    required=False,
    default=1,
    help="Gradient accumulation steps. Global batch size is inferred from this.",
)

if __name__ == "__main__":
    args = parser.parse_args()

    results = infer_model(
        data_path=args.data_path,
        checkpoint_path=args.checkpoint_path,
        results_path=args.results_path,
        include_hiddens=args.include_hiddens,
        include_embeddings=args.include_embeddings,
        micro_batch_size=args.micro_batch_size,
        accumulate_grad_batches=args.accumulate_grad_batches,
        precision=args.precision,
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        devices=args.num_gpus,
        num_nodes=args.num_nodes,
    )
