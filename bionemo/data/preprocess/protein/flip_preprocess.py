# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import pathlib
import re
import urllib.request
from typing import Dict, List, Optional

import numpy as np
import pyfastx
from nemo.utils import logging


__all__ = ['FLIPPreprocess']

ROOT_DIR = '/tmp/FLIP'

SEQUENCE_FNAME = {
    "aav": "seven_vs_many.fasta",
    "bind": "sequences.fasta",
    "conservation": "sequences.fasta",
    "gb1": "two_vs_rest.fasta",
    "meltome": "mixed_split.fasta",
    "sav": "mixed.fasta",
    "scl": "mixed_soft.fasta",
    "secondary_structure": "sequences.fasta",
}

LABELS_FNAME = {
    "aav": None,
    "bind": "from_publication.fasta",
    "conservation": "sampled.fasta",
    "gb1": None,
    "meltome": None,
    "sav": None,
    "scl": None,
    "secondary_structure": "sampled.fasta",
}

RESOLVED_FNAME = {
    "aav": None,
    "bind": None,
    "conservation": None,
    "gb1": None,
    "meltome": None,
    "sav": None,
    "scl": None,
    "secondary_structure": "resolved.fasta",
}

HEADER = {
    "aav": ["id", "sequence", "target"],
    "bind": ["id", "sequence", "target"],
    "conservation": ["id", "sequence", "conservation"],
    "gb1": ["id", "sequence", "target"],
    "meltome": ["id", "sequence", "target"],
    "sav": ["id", "sequence", "target"],
    "scl": ["id", "sequence", "scl_label"],
    "secondary_structure": ["id", "sequence", "3state", "resolved"],
}

URL_PREFIX = "http://data.bioembeddings.com/public/FLIP/fasta/"


def get_attributes_from_seq(sequences: List) -> Dict[str, Dict[str, str]]:
    """
    :param sequences: a list of SeqRecords
    :return: A dictionary of ids and their attributes
    """
    result = {}
    for sequence in sequences:
        result[sequence.name] = dict(re.findall(r"([A-Z_]+)=(-?[A-z0-9]+-?[A-z0-9]*[.0-9]*)", sequence.description))

    return result


class FLIPPreprocess:
    def __init__(
        self,
        root_directory: Optional[str] = ROOT_DIR,
    ):
        super().__init__()
        self.root_directory = pathlib.Path(root_directory)

    def download_FLIP_data(self, download_dir, task_name="secondary_structure"):
        url = URL_PREFIX + task_name + "/"
        filenames = [SEQUENCE_FNAME[task_name], LABELS_FNAME[task_name], RESOLVED_FNAME[task_name]]
        exists = os.path.exists(download_dir)
        if not exists:
            os.makedirs(download_dir)
        all_file_paths = []
        for filename in filenames:
            if filename is not None:
                file_url = url + filename
                file_path = os.path.join(download_dir, filename)
                all_file_paths.append(file_path)
                req = urllib.request.Request(
                    file_url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
                    },
                )
                response = urllib.request.urlopen(req)
                with open(file_path, "w") as f:
                    f.write(response.read().decode("utf-8"))
                logging.info(f"{filename} downloaded successfully!")
            else:
                all_file_paths.append(None)

        return all_file_paths

    def get_train_val_test_splits(self, splits):
        splits_dict = get_attributes_from_seq(splits)

        train_samples = []
        test_samples = []
        val_samples = []
        for key, label in splits_dict.items():
            if label["SET"] == "train" and label["VALIDATION"] == "False":
                train_samples.append(key)
            elif label["SET"] == "train" and label["VALIDATION"] == "True":
                val_samples.append(key)
            elif label["SET"] == "test":
                test_samples.append(key)

        return train_samples, val_samples, test_samples

    def write_csv_files(
        self,
        train_samples,
        val_samples,
        test_samples,
        sequence_indexer,
        labels_indexer,
        resolved_indexer,
        num_csv_files,
        output_dir,
        task_name,
    ):
        name_to_seqs = {seq.name: str(seq.seq) for seq in sequence_indexer}
        if resolved_indexer is not None:
            name_to_masks = {mask.name: str(mask.seq) for mask in resolved_indexer}
        else:
            name_to_masks = None
        if labels_indexer is not None:
            name_to_labels = {label.name: str(label.seq) for label in labels_indexer}
        else:
            seq_attr = get_attributes_from_seq(sequence_indexer)
            name_to_labels = {key: seq_attr[key]["TARGET"] for key in seq_attr.keys()}
        for split_name, record_id_list in zip(['train', 'val', 'test'], [train_samples, val_samples, test_samples]):
            logging.info(f'Saving {split_name} split...')

            for file_index, record_id_split in enumerate(np.array_split(record_id_list, num_csv_files)):
                logging.debug(f'Writing file number {file_index}...')
                self._csv_files_writer(
                    record_id_list=record_id_split,
                    file_index=file_index,
                    split_name=split_name,
                    seq_dict=name_to_seqs,
                    labels_dict=name_to_labels,
                    masks_dict=name_to_masks,
                    output_dir=output_dir,
                    task_name=task_name,
                )
        return

    def _csv_files_writer(
        self,
        record_id_list,
        file_index,
        split_name,
        seq_dict,
        labels_dict,
        masks_dict,
        output_dir,
        task_name,
        delimiter=",",
    ):
        split_path = os.path.join(output_dir, split_name)
        pathlib.Path(split_path).mkdir(parents=True, exist_ok=True)
        file_name = os.path.join(split_path, f'x{str(file_index).zfill(3)}.csv')

        with open(file_name, 'w') as fh:
            header_str = ",".join(HEADER[task_name])
            fh.write(header_str + '\n')
            for record_id in record_id_list:
                sequence = seq_dict[record_id]
                labels = labels_dict[record_id]
                if masks_dict is not None:
                    mask = masks_dict[record_id]
                    output = delimiter.join([record_id, sequence, labels, str(mask)])
                else:
                    output = delimiter.join([record_id, sequence, labels])
                fh.write(output + '\n')
        return

    def prepare_all_datasets(self, output_dir="/data/FLIP", num_csv_files=1):
        tasks_list = ["aav", "bind", "conservation", "gb1", "meltome", "sav", "scl", "secondary_structure"]
        for task_name in tasks_list:
            self.prepare_dataset(os.path.join(output_dir, task_name), task_name=task_name, num_csv_files=num_csv_files)

    def prepare_dataset(
        self,
        output_dir,
        task_name="secondary_structure",
        num_csv_files=1,
    ):
        """Download FLIP secondary structure dataset and split into train, valid, and test sets.
        Splits are pre-defined.

        Args:
            url (str): URL for FLIP SS dataset location.
            num_csv_files (int): Number of CSV files to create for each train/val/test split.
        """
        download_dir = os.path.join(self.root_directory, task_name, 'raw')
        exists = os.path.exists(output_dir)
        if not exists:
            os.makedirs(output_dir)
        seq_path, labels_path, resolved_path = self.download_FLIP_data(download_dir=download_dir, task_name=task_name)
        sequence_indexer = pyfastx.Fasta(seq_path, build_index=True, uppercase=True)
        if labels_path is not None:
            labels_indexer = pyfastx.Fasta(labels_path, build_index=True, uppercase=True)
        else:
            labels_indexer = None
        if resolved_path is not None:
            resolved_indexer = pyfastx.Fasta(resolved_path, build_index=True, uppercase=True)
        else:
            resolved_indexer = None
        logging.info('FLIP data download complete.')

        logging.info('Processing FLIP dataset.')
        if labels_indexer is not None:
            train_samples, val_samples, test_samples = self.get_train_val_test_splits(labels_indexer)
        else:
            train_samples, val_samples, test_samples = self.get_train_val_test_splits(sequence_indexer)

        logging.info(f'Writing processed dataset files to {output_dir}...')
        self.write_csv_files(
            train_samples=train_samples,
            val_samples=val_samples,
            test_samples=test_samples,
            num_csv_files=num_csv_files,
            sequence_indexer=sequence_indexer,
            labels_indexer=labels_indexer,
            resolved_indexer=resolved_indexer,
            output_dir=output_dir,
            task_name=task_name,
        )
        logging.info('FLIP dataset preprocessing completed')
