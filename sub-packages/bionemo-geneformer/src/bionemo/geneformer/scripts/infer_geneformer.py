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
from typing import Dict, Type, get_args

import torch
from nemo import lightning as nl
from nemo.utils import logging

from bionemo.core.utils.dtypes import PrecisionTypes, get_autocast_dtype
from bionemo.geneformer.api import FineTuneSeqLenBioBertConfig, GeneformerConfig
from bionemo.geneformer.data.singlecell.datamodule import SingleCellDataModule
from bionemo.geneformer.data.singlecell.preprocess import GeneformerPreprocess
from bionemo.llm.lightning import batch_collator
from bionemo.llm.model.biobert.lightning import biobert_lightning_module
from bionemo.llm.model.biobert.model import BioBertConfig
from bionemo.llm.utils.datamodule_utils import infer_global_batch_size
from bionemo.testing.data.load import load


def infer_model(
    data_path: Path,
    checkpoint_path: Path,
    results_path: Path,
    include_hiddens: bool = False,
    include_embeddings: bool = False,
    include_logits: bool = False,
    seq_length: int = 2048,
    micro_batch_size: int = 64,
    precision: PrecisionTypes = "bf16-mixed",
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    devices: int = 1,
    num_nodes: int = 1,
    num_dataset_workers: int = 0,
    config_class: Type[BioBertConfig] = GeneformerConfig,
) -> None:
    """Inference function (requires DDP and only training data that fits in memory)."""
    # This is just used to get the tokenizer :(
    train_data_path: Path = (
        load("single_cell/testdata-20240506") / "cellxgene_2023-12-15_small" / "processed_data" / "train"
    )

    # Setup the strategy and trainer
    pipeline_model_parallel_size = 1
    tensor_model_parallel_size = 1
    accumulate_grad_batches = 1
    global_batch_size = infer_global_batch_size(
        micro_batch_size=micro_batch_size,
        num_nodes=num_nodes,
        devices=devices,
        accumulate_grad_batches=accumulate_grad_batches,
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
    )

    preprocessor = GeneformerPreprocess(
        download_directory=train_data_path,
        medians_file_path=train_data_path / "medians.json",
        tokenizer_vocab_path=train_data_path / "geneformer.vocab",
    )
    match preprocessor.preprocess():
        case {"tokenizer": tokenizer, "median_dict": median_dict}:
            logging.info("*************** Preprocessing Finished ************")
        case _:
            logging.error("Preprocessing failed.")

    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        ddp="megatron",
        find_unused_parameters=True,
        ckpt_include_optimizer=True,
        progress_interval=1,
    )
    trainer = nl.Trainer(
        devices=devices,
        accelerator="gpu",
        strategy=strategy,
        num_nodes=num_nodes,
        callbacks=[],
        plugins=nl.MegatronMixedPrecision(precision=precision),
    )
    # Configure the data module and model
    data = SingleCellDataModule(
        seq_length=seq_length,
        tokenizer=tokenizer,
        train_dataset_path=None,
        val_dataset_path=None,
        test_dataset_path=None,
        predict_dataset_path=data_path,
        mask_prob=0,
        mask_token_prob=0,
        random_token_prob=0,  # changed to represent the incorrect setting we originally used.
        median_dict=median_dict,
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
        # persistent workers is supported when num_dataset_workers > 0
        persistent_workers=num_dataset_workers > 0,
        pin_memory=False,
        num_workers=num_dataset_workers,
    )
    geneformer_config = config_class(
        seq_length=seq_length,
        params_dtype=get_autocast_dtype(precision),
        pipeline_dtype=get_autocast_dtype(precision),
        autocast_dtype=get_autocast_dtype(precision),  # setting this speeds things up a lot
        # handle checkpoint resumption here rather than auto-resume so this supports fine-tuning capabilities
        initial_ckpt_path=str(checkpoint_path) if checkpoint_path is not None else None,
        include_embeddings=include_embeddings,
        include_hiddens=include_hiddens,
        skip_logits=not include_logits,
        initial_ckpt_skip_keys_with_these_prefixes=[],  # load everything from the checkpoint.
    )
    # The lightning class owns a copy of the actual model, and a loss function, both of which are configured
    #  and lazily returned by the `geneformer_config` object defined above.
    model = biobert_lightning_module(
        geneformer_config,
        tokenizer=tokenizer,
    )

    results_dict = batch_collator(trainer.predict(model, datamodule=data, return_predictions=True))
    non_none_keys = [key for key, val in results_dict.items() if val is not None]
    print(f"Writing output {str(non_none_keys)} into {results_path}")
    torch.save(results_dict, results_path)


def geneformer_infer_entrypoint():
    """Entrypoint for running inference on a geneformer checkpoint and data."""
    # 1. get arguments
    parser = get_parser()
    args = parser.parse_args()
    # 2. Call infer with args
    infer_model(
        data_path=args.data_dir,
        checkpoint_path=args.checkpoint_path,
        results_path=args.result_path,
        include_hiddens=args.include_hiddens,
        micro_batch_size=args.micro_batch_size,
        include_embeddings=not args.no_embeddings,
        include_logits=args.include_logits,
        seq_length=args.seq_length,
        precision=args.precision,
        devices=args.num_gpus,
        num_nodes=args.num_nodes,
        num_dataset_workers=args.num_dataset_workers,
        config_class=args.config_class,
    )


def get_parser():
    """Return the cli parser for this tool."""
    parser = argparse.ArgumentParser(
        description="Infer sc_memmap processed single cell data with Geneformer from a checkpiont."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to the data directory, for example this might be "
        "/workspace/bionemo2/data/cellxgene_2023-12-15_small/processed_train",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        required=False,
        default=None,
        help="Path to the checkpoint directory to restore from.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=get_args(PrecisionTypes),
        required=False,
        default="bf16-mixed",
        help="Precision type to use for training.",
    )
    parser.add_argument("--include-hiddens", action="store_true", default=False, help="Include hiddens in output.")
    parser.add_argument("--no-embeddings", action="store_true", default=False, help="Do not output embeddings.")
    parser.add_argument(
        "--include-logits", action="store_true", default=False, help="Include per-token logits in output."
    )

    parser.add_argument(
        "--result-path", type=Path, required=False, default=Path("./results.pt"), help="Path to the result file."
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
        "--num-dataset-workers",
        type=int,
        required=False,
        default=0,
        help="Number of steps to use for training. Default is 0.",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        required=False,
        default=2048,
        help="Sequence length of cell. Default is 2048.",
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        required=False,
        default=32,
        help="Micro-batch size. Global batch size is inferred from this.",
    )

    # TODO consider whether nemo.run or some other method can simplify this config class lookup.
    config_class_options: Dict[str, Type[BioBertConfig]] = {
        "GeneformerConfig": GeneformerConfig,
        "FineTuneSeqLenBioBertConfig": FineTuneSeqLenBioBertConfig,
    }

    def config_class_type(desc: str) -> Type[BioBertConfig]:
        try:
            return config_class_options[desc]
        except KeyError:
            raise argparse.ArgumentTypeError(
                f"Do not recognize key {desc}, valid options are: {config_class_options.keys()}"
            )

    parser.add_argument(
        "--config-class",
        type=config_class_type,
        default="GeneformerConfig",
        help="Model configs link model classes with losses, and handle model initialization (including from a prior "
        "checkpoint). This is how you can fine-tune a model. First train with one config class that points to one model "
        "class and loss, then implement and provide an alternative config class that points to a variant of that model "
        "and alternative loss. In the future this script should also provide similar support for picking different data "
        f"modules for fine-tuning with different data types. Choices: {config_class_options.keys()}",
    )
    return parser


if __name__ == "__main__":
    geneformer_infer_entrypoint()
