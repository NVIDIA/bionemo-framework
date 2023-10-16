# Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

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

"""
Runs inference over all models. Supports extracting embeddings, and hiddens.

NOTE: If out of memory (OOM) error occurs, try spliting the data to multiple smaller files.
"""

import os
import pickle
import uuid

import torch
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.distributed import gather_objects
from nemo.utils.model_utils import import_class_by_path
from omegaconf.omegaconf import OmegaConf

from bionemo.data.mapped_dataset import FilteredMappedDataset
from bionemo.data.memmap_csv_fields_dataset import CSVFieldsMemmapDataset
from bionemo.data.memmap_fasta_fields_dataset import FASTAFieldsMemmapDataset
from bionemo.data.utils import expand_dataset_paths


@hydra_runner(config_path="conf", config_name="infer")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    infer_class = import_class_by_path(cfg.infer_target)
    infer_model = infer_class(cfg)
    trainer = infer_model.trainer

    logging.info("\n\n************** Restored model configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(infer_model.model.cfg)}')

    # TODO: move this code into a dataset builder in data utils
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
    elif cfg.model.data.data_impl == "fasta_fields_mmap":
        dataset_paths = expand_dataset_paths(cfg.model.data.dataset_path, ext=".fasta")
        ds = FASTAFieldsMemmapDataset(
            dataset_paths,
            index_mapping_dir=cfg.model.data.index_mapping_dir,
            **cfg.model.data.data_impl_kwargs.get("fasta_fields_mmap", {}),
        )
    else:
        raise ValueError(f'Unknown data_impl: {cfg.model.data.data_impl}')

    # remove too long sequences
    filtered_ds = FilteredMappedDataset(
        dataset=ds,
        criterion_fn=lambda x: len(infer_model._tokenize([x["sequence"]])[0]) <= infer_model.model.cfg.seq_length,
    )

    dataloader = torch.utils.data.DataLoader(
        filtered_ds,
        batch_size=cfg.model.data.batch_size,
        num_workers=cfg.model.data.num_workers,
        drop_last=False,
    )

    # predict outputs for all sequences in batch mode
    all_batch_predictions = trainer.predict(
        model=infer_model,
        dataloaders=dataloader,
        return_predictions=True,
    )

    if not len(all_batch_predictions):
        raise ValueError("No predictions were made")

    # break batched predictions into individual predictions (list of dics)
    predictions = []
    pred_keys = list(all_batch_predictions[0].keys())

    def cast_to_numpy(x):
        if torch.is_tensor(x):
            return x.cpu().numpy()
        return x

    for batch_predictions in all_batch_predictions:
        batch_size = len(batch_predictions[pred_keys[0]])
        for i in range(batch_size):
            predictions.append({k: cast_to_numpy(batch_predictions[k][i]) for k in pred_keys})

    # extract active hiddens if needed
    if "hiddens" in cfg.model.downstream_task.outputs:
        if ("hiddens" in predictions[0]) and ("mask" in predictions[0]):
            for p in predictions:
                p["hiddens"] = p['hiddens'][p['mask']]
                del p['mask']
    else:
        for p in predictions:
            del p['mask']
            del p['hiddens']

    # collect all results when using DDP
    logging.info("Collecting results from all GPUs...")
    predictions = gather_objects(predictions, main_rank=0)
    # all but rank 0 will return None
    if predictions is None:
        return

    # from here only rank 0 should continue
    output_fname = cfg.model.data.output_fname
    if not output_fname:
        output_fname = f"{cfg.model.data.dataset_path}.pkl"
        logging.info(f"output_fname not specified, using {output_fname}")
    if os.path.exists(cfg.model.data.output_fname):
        logging.warning(f"Output path {output_fname} already exists, appending a unique id to the path")
        output_fname += "." + str(uuid.uuid4())

    if ~os.path.exists(output_fname):
        os.makedirs(os.path.dirname(output_fname), exist_ok=True)

    logging.info(f"Saving {len(predictions)} samples to output_fname = {output_fname}")
    pickle.dump(predictions, open(output_fname, "wb"))


if __name__ == '__main__':
    main()
