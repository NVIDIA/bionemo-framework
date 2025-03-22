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
from pathlib import Path
from typing import Literal, cast

import pandas as pd
import typer
from nemo import lightning as nl

from bionemo.amplify.model import AMPLIFYConfig
from bionemo.core.utils.dtypes import PrecisionTypes, get_autocast_dtype
from bionemo.esm2.model.finetune.dataset import InMemoryProteinDataset
from bionemo.llm.data.datamodule import MockDataModule
from bionemo.llm.model.biobert.lightning import biobert_lightning_module
from bionemo.llm.utils.callbacks import IntervalT, PredictionWriter
from bionemo.llm.utils.datamodule_utils import infer_global_batch_size


app = typer.Typer()


def infer_model(
    data_path: Path,
    hf_model_name: str,
    results_path: Path,
    min_seq_length: int = 1024,
    include_hiddens: bool = False,
    include_embeddings: bool = False,
    include_logits: bool = False,
    include_input_ids: bool = False,
    micro_batch_size: int = 64,
    precision: str = "bf16-mixed",
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    devices: int = 1,
    num_nodes: int = 1,
    prediction_interval: IntervalT = "epoch",
) -> None:
    """Runs inference on a BioNeMo AMPLIFY model using PyTorch Lightning.

    Args:
        data_path: Path to the input data CSV file
        hf_model_name: HuggingFace model name/path to load
        results_path: Path to save inference results
        min_seq_length: Minimum sequence length for padding
        include_hiddens: Whether to include hidden states in output
        include_embeddings: Whether to include embeddings in output
        include_logits: Whether to include token logits in output
        include_input_ids: Whether to include input IDs in output
        micro_batch_size: Micro batch size for inference
        precision: Precision type for inference
        tensor_model_parallel_size: Tensor model parallel size
        pipeline_model_parallel_size: Pipeline model parallel size
        devices: Number of devices to use
        num_nodes: Number of nodes for distributed inference
        prediction_interval: Intervals to write predictions to disk
    """
    # Create results directory
    os.makedirs(results_path, exist_ok=True)

    # Setup strategy and trainer
    global_batch_size = infer_global_batch_size(
        micro_batch_size=micro_batch_size,
        num_nodes=num_nodes,
        devices=devices,
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
    )

    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        ddp="megatron",
        find_unused_parameters=True,
    )

    prediction_writer = PredictionWriter(output_dir=results_path, write_interval=prediction_interval)

    # Cast precision to literal type expected by MegatronMixedPrecision
    assert precision in ["16-mixed", "bf16-mixed", "32"], (
        f"Precision must be one of: 16-mixed, bf16-mixed, 32, got {precision}"
    )
    precision_literal: Literal["16-mixed", "bf16-mixed", "32"] = cast(
        Literal["16-mixed", "bf16-mixed", "32"], precision
    )

    trainer = nl.Trainer(
        accelerator="gpu",
        devices=devices,
        strategy=strategy,
        num_nodes=num_nodes,
        callbacks=prediction_writer,
        plugins=nl.MegatronMixedPrecision(precision=precision_literal),
    )

    # Setup data
    # Load CSV into HuggingFace dataset
    df = pd.read_csv(data_path)
    if "sequence" not in df.columns:
        raise ValueError("Input CSV must have a 'sequence' column")

    dataset = InMemoryProteinDataset.from_csv(data_path, ignore_labels=True)
    datamodule = MockDataModule(train_dataset=dataset, batch_size=micro_batch_size)

    # Initialize model config and model
    config = AMPLIFYConfig(
        params_dtype=get_autocast_dtype(cast(PrecisionTypes, precision)),
        pipeline_dtype=get_autocast_dtype(cast(PrecisionTypes, precision)),
        autocast_dtype=get_autocast_dtype(cast(PrecisionTypes, precision)),
        include_hiddens=include_hiddens,
        include_embeddings=include_embeddings,
        include_input_ids=include_input_ids,
        skip_logits=not include_logits,
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        initial_ckpt_path=hf_model_name,
    )

    module = biobert_lightning_module(config=config, tokenizer=tokenizer)  # type: ignore

    # Run inference
    trainer.predict(module, datamodule=datamodule)


@app.command()
def main(
    data_path: Path = typer.Argument(..., help="Path to input CSV file with sequences"),
    hf_model_name: str = typer.Argument(..., help="HuggingFace model name/path to load"),
    results_path: Path = typer.Argument(..., help="Path to save inference results"),
    min_seq_length: int = typer.Option(1024, help="Minimum sequence length for padding"),
    include_hiddens: bool = typer.Option(False, help="Include hidden states in output"),
    include_embeddings: bool = typer.Option(False, help="Include embeddings in output"),
    include_logits: bool = typer.Option(False, help="Include token logits in output"),
    include_input_ids: bool = typer.Option(False, help="Include input IDs in output"),
    micro_batch_size: int = typer.Option(64, help="Micro batch size for inference"),
    precision: PrecisionTypes = typer.Option("bf16-mixed", help="Precision type for inference"),
    tensor_model_parallel_size: int = typer.Option(1, help="Tensor model parallel size"),
    pipeline_model_parallel_size: int = typer.Option(1, help="Pipeline model parallel size"),
    devices: int = typer.Option(1, help="Number of devices to use"),
    num_nodes: int = typer.Option(1, help="Number of nodes for distributed inference"),
    prediction_interval: IntervalT = typer.Option("epoch", help="Intervals to write predictions to disk"),
) -> None:
    """Run inference using the AMPLIFY model on protein sequences."""
    infer_model(
        data_path=data_path,
        hf_model_name=hf_model_name,
        results_path=results_path,
        min_seq_length=min_seq_length,
        include_hiddens=include_hiddens,
        include_embeddings=include_embeddings,
        include_logits=include_logits,
        include_input_ids=include_input_ids,
        micro_batch_size=micro_batch_size,
        precision=precision,
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        devices=devices,
        num_nodes=num_nodes,
        prediction_interval=prediction_interval,
    )


if __name__ == "__main__":
    app()
