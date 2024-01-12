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
import urllib.request
from pathlib import Path

import pandas as pd
from nemo.utils import logging


__all__ = ['PhysChemPreprocess']


class PhysChemPreprocess:
    '''
    Handles download of PhysChem datasets
    '''

    def __init__(self):
        super().__init__()

    def prepare_dataset(self, links_file: str, output_dir: str):
        '''
        Downloads Physical Chemistry datasets

        Params:
            links_file: File containing links to be downloaded.
            output_dir: Directory to save the processed data to.
        '''
        logging.info(f'Downloading files from {links_file}...')

        os.makedirs(output_dir, exist_ok=True)
        with open(links_file, 'r') as f:
            links = list({x.strip() for x in f})

        download_dir_path = Path(output_dir)

        for url in links:
            filename = url.split('/')[-1]
            if os.path.exists(os.path.join(output_dir, filename)):
                logging.info(f'{url} already downloaded...')
                continue

            logging.info(f'Downloading file {filename}...')
            urllib.request.urlretrieve(url, download_dir_path / filename)

        logging.info('Download complete.')

    def _process_split(self, links_file: str, output_dir: str, test_frac: float, val_frac: float, seed=0):
        logging.info(f'Splitting files in {output_dir} into train, validation, and test data')

        os.makedirs(output_dir, exist_ok=True)

        with open(links_file, 'r') as f:
            links = list({x.strip() for x in f})

        dir_path = Path(output_dir)

        for url in links:
            filename = url.split('/')[-1]

            df = pd.read_csv(os.path.join(dir_path, filename), header=0)

            # Calculate sample sizes before size of dataframe changes
            test_samples = max(int(test_frac * df.shape[0]), 1)
            val_samples = max(int(val_frac * df.shape[0]), 1)

            test_df = df.sample(n=test_samples, random_state=seed)
            df = df.drop(test_df.index)  # remove test data from training data

            val_df = df.sample(n=val_samples, random_state=seed)
            df = df.drop(val_df.index)  # remove validation data from training data

            splits_path = os.path.join(dir_path, filename.split('.')[0])
            os.makedirs(splits_path, exist_ok=True)

            train_path = os.path.join(splits_path, "train")
            os.makedirs(train_path, exist_ok=True)

            val_path = os.path.join(splits_path, "val")
            os.makedirs(val_path, exist_ok=True)

            test_path = os.path.join(splits_path, "test")
            os.makedirs(test_path, exist_ok=True)

            df.to_csv(f'{train_path}/x000.csv', index=False)
            test_df.to_csv(f'{val_path}/x000.csv', index=False)
            val_df.to_csv(f'{test_path}/x000.csv', index=False)

            del df
            del test_df
            del val_df
