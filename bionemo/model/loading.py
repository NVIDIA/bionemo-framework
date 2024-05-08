# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
from typing import Sequence, Tuple

import pytorch_lightning as pl
from nemo.collections.nlp.data.language_modeling.text_memmap_dataset import CSVFieldsMemmapDataset
from nemo.utils import logging
from nemo.utils.model_utils import import_class_by_path
from omegaconf import DictConfig
from omegaconf.omegaconf import OmegaConf
from torch.utils.data import DataLoader

from bionemo.data.mapped_dataset import FilteredMappedDataset
from bionemo.data.memmap_fasta_fields_dataset import FASTAFieldsMemmapDataset
from bionemo.data.singlecell.dataset import SingleCellDataset
from bionemo.data.utils import expand_dataset_paths
from bionemo.model.core.infer import BaseEncoderInference, M


__all__: Sequence[str] = ("setup_inference",)


def setup_inference(cfg: DictConfig, *, interactive: bool = False) -> Tuple[M, pl.Trainer, DataLoader]:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    infer_class = import_class_by_path(cfg.infer_target)
    if isinstance(infer_class, BaseEncoderInference):
        kwargs = {"inference_batch_size_for_warmup": cfg.model.data.batch_size}
    else:
        kwargs = {}

    infer_model = infer_class(cfg, interactive=interactive, **kwargs)
    trainer = infer_model.trainer

    logging.info("\n\n************** Restored model configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(infer_model.model.cfg)}')

    # TODO [???]: move this code into a dataset builder in data utils
    if not cfg.model.data.data_impl:
        # try to infer data_impl from the dataset_path file extension
        if cfg.model.data.dataset_path.endswith('.fasta'):
            cfg.model.data.data_impl = 'fasta_fields_mmap'
        else:
            # Data are assumed to be CSV format if no extension provided
            logging.info('File extension not supplied for data, inferring csv.')
            cfg.model.data.data_impl = 'csv_fields_mmap'

        logging.info(f'Inferred data_impl: {cfg.model.data.data_impl}')

    if cfg.model.data.data_impl == "csv_fields_mmap":
        dataset_paths = expand_dataset_paths(cfg.model.data.dataset_path, ext=".csv")
        ds = CSVFieldsMemmapDataset(
            dataset_paths,
            index_mapping_dir=cfg.model.data.index_mapping_dir,
            **cfg.model.data.data_impl_kwargs.get("csv_fields_mmap", {}),
        )
        remove_too_long = True
    elif cfg.model.data.data_impl == "fasta_fields_mmap":
        dataset_paths = expand_dataset_paths(cfg.model.data.dataset_path, ext=".fasta")
        ds = FASTAFieldsMemmapDataset(
            dataset_paths,
            index_mapping_dir=cfg.model.data.index_mapping_dir,
            **cfg.model.data.data_impl_kwargs.get("fasta_fields_mmap", {}),
        )
        remove_too_long = True
    elif cfg.model.data.data_impl == "geneformer":
        # Get the medians and tokenizer from the inference model for our dataset.
        dataset_path = cfg.model.data.get("dataset_path", None)
        if dataset_path is None:
            raise ValueError(
                "model.data.dataset_path must be provided as the path to the dataset that we want to run inference on. "
                "Annoyingly you still need to also provide train/test/val paths in the config to support model loading, "
                "but those are not what are used for inference."
            )
        ds = SingleCellDataset(
            dataset_path,
            infer_model.model.tokenizer,
            infer_model.model.median_dict,
            max_len=cfg.model.seq_length,
            mask_prob=0,  # Assume we do not want to mask any genes
        )
        remove_too_long = False  # there is no "too long", the dataset will take the top N genes from each cell.
    else:
        raise ValueError(f'Unknown data_impl: {cfg.model.data.data_impl}')
        remove_too_long = True

    # remove too long sequences
    if remove_too_long:
        filtered_ds = FilteredMappedDataset(
            dataset=ds,
            criterion_fn=lambda x: len(infer_model._tokenize([x["sequence"]])[0]) <= infer_model.model.cfg.seq_length,
        )
    else:
        filtered_ds = ds  # no filtering

    dataloader = DataLoader(
        filtered_ds,
        batch_size=cfg.model.data.batch_size,
        num_workers=cfg.model.data.num_workers,
        drop_last=False,
    )
    return infer_model, trainer, dataloader
