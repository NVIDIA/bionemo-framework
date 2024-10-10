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
import shutil
from typing import Optional

import pandas as pd
from nemo.utils import logging
from rdkit import Chem

from bionemo.data.utils import download_registry_from_ngc, get_ngc_registry_file_list, verify_checksum_matches


HUMAN_DOWNLOAD_PATH = "https://az.app.box.com/s/7eci3nd9vy0xplqniitpk02rbg9q2zcq/file/854847820319"
MD5_CHECKSUM = "d956c753c757f19c8e9d913f51cf0eed"


class USPTO50KPreprocess:
    """
    Downloads and prepares the reaction data USPTO50K for model training, ie by cleaning and splitting
    the data into train, validation and tests sets.
    """

    def __init__(self, data_dir: str, max_smiles_length: Optional[str] = None, checksum: Optional[str] = MD5_CHECKSUM):
        self.data_dir = data_dir
        self.max_smiles_length = max_smiles_length
        self.download_dir = os.path.join(data_dir, "raw")
        output_file = "uspto_50.pickle"
        datapath_raw = os.path.join(self.download_dir, output_file)
        self.checksum = checksum
        if os.path.exists(datapath_raw) and verify_checksum_matches(datapath_raw, checksum):
            self.datapath_raw = datapath_raw
        else:
            self.datapath_raw = None
        self.processed_dir = os.path.join(data_dir, "processed")
        self.data_file = "data.csv"
        self.splits = ["train", "val", "test"]

    def get_split_dir(self, split: str) -> str:
        return os.path.join(self.processed_dir, split)

    def prepare_dataset(
        self,
        ngc_registry_target: Optional[str] = None,
        ngc_registry_version: Optional[str] = None,
        force: bool = False,
    ):
        """
        Downloads reaction dataset and splits it into train, validation, and test sets.
        Args:
            ngc_registry_target: NGC registry target name for dataset
            ngc_registry_version: NGC registry version for dataset
            filename_raw (str): the name of the file with the raw data
            force (bool): if the generation of the dataset should be forced
        """
        if os.path.exists(self.processed_dir) and not force:
            logging.info(f"Path to the processed dataset {self.processed_dir} exists!")
            return
        download_file = True
        if self.datapath_raw is not None and os.path.exists(self.datapath_raw):
            download_file = not verify_checksum_matches(self.datapath_raw, self.checksum)

        if download_file:
            if ngc_registry_target is None or ngc_registry_version is None:
                raise ValueError(
                    "Internal Nvidia employees must provide NGC registry target and version must be defined to download the dataset, or alternatively you must first "
                    f"manually download from {HUMAN_DOWNLOAD_PATH} and place in {self.download_dir} with the name uspto_50.pickle."
                )
            self.datapath_raw = self.download_raw_data_file(
                ngc_registry_target=ngc_registry_target, ngc_registry_version=ngc_registry_version
            )

        if self.datapath_raw:
            self.train_val_test_split(datapath_raw=self.datapath_raw)
        else:
            logging.error(
                f"Failed to download dataset target {ngc_registry_target} and version {ngc_registry_version}!"
            )

    def download_raw_data_file(self, ngc_registry_target: str, ngc_registry_version: str) -> Optional[str]:
        """
        Downloads raw data from the url link and saves it in a local directory
        Args:
            ngc_registry_target (str): NGC registry target for the data to be downloaded.
            ngc_registry_version (str): NGC registry version for the data to be downloaded.
            filename (str): optional, the name of the file with raw data
        Returns:
            output_path (str): path to the location of the downloaded file
        """
        logging.info(
            f"Downloading dataset from target {ngc_registry_target} and version "
            f"{ngc_registry_version} from NGC ..."
        )
        if not os.path.exists(self.download_dir):
            os.makedirs(self.download_dir, exist_ok=True)

        try:
            assert os.environ.get("NGC_CLI_API_KEY", False), AssertionError(
                """NGC API key not defined as environment variable "NGC_CLI_API_KEY".
                                                                            Aborting resource download."""
            )
            ngc_org = os.environ.get("NGC_CLI_ORG", None)
            assert ngc_org, AssertionError("NGC org must be defined by the environment variable NGC_CLI_ORG")
            ngc_team = os.environ.get("NGC_CLI_TEAM", None)

            # Check if resource already exists at final destination
            file_list = get_ngc_registry_file_list(ngc_registry_target, ngc_registry_version, ngc_org, ngc_team)
            file_exists = False
            if len(file_list) > 1:
                logging.info("Checksum verification not supported if resource contains more than one file.")
            else:
                file_name = file_list[0]
                output_path = os.path.join(self.download_dir, file_name)
                if os.path.exists(output_path):
                    file_exists = True if verify_checksum_matches(output_path, MD5_CHECKSUM) else False

            # Download resource and copy if needed
            if not file_exists:
                tmp_download_path = download_registry_from_ngc(
                    ngc_registry_target=ngc_registry_target,
                    ngc_registry_version=ngc_registry_version,
                    ngc_org=ngc_org,
                    ngc_team=ngc_team,
                    dest=self.data_dir,
                    expected_checksum=MD5_CHECKSUM,
                )

                # Move to destination directory and clean up
                file_name = os.path.basename(tmp_download_path)
                output_path = os.path.join(
                    self.download_dir, file_name
                )  # Ensures output_path is defined when file is downloaded
                shutil.copyfile(tmp_download_path, output_path)
                logging.info(f"Download complete at {output_path}.")
            else:
                logging.info(f"File download skipped because file exists at {output_path} and has expected checksum.")

            return output_path

        except Exception as e:
            logging.error(
                f"Could not download from NGC dataset from target {ngc_registry_target} and version {ngc_registry_version}: {e}"
            )
            raise e

    def train_val_test_split(self, datapath_raw: str):
        """
        Splits downloaded raw dataset into train, validation and tests sets
        Args:
            datapath_raw (str): local path to the file with the raw data
        """
        logging.info(
            f'Splitting file {datapath_raw} into {", ".join(self.splits)} data and ' f'saving in {self.processed_dir}'
        )

        df = pd.read_pickle(datapath_raw)
        assert all(col in df for col in ["reactants_mol", "products_mol", "reaction_type", "set"])

        # TODO parallelize in the future!
        for name in ["reactants", "products"]:
            df[f"{name}_correct"] = df[f"{name}_mol"].apply(
                lambda mol: True if Chem.MolToSmiles(mol, canonical=True) else False
            )

        df = df[df["reactants_correct"] & df["products_correct"]]
        for name in ["reactants", "products"]:
            df[name] = df[f"{name}_mol"].apply(lambda mol: Chem.MolToSmiles(mol, canonical=False))
            df[f"{name}_len"] = df[f"{name}"].apply(lambda smi: len(smi))

        df.dropna(axis=0, how="any", subset=["reactants", "products"], inplace=True)
        if self.max_smiles_length:
            df = df[(df["reactants_len"] <= self.max_smiles_length) & (df["products_len"] <= self.max_smiles_length)]
        df.drop(columns=set(df.columns) - {"reactants", "products", "set", "reaction_type"}, inplace=True)
        df.set.replace(to_replace="valid", value="val", inplace=True)

        for split in self.splits:
            df_tmp = df[df["set"] == split]
            df_tmp.reset_index(drop=True, inplace=True)
            dir_tmp = self.get_split_dir(split)
            if not os.path.exists(dir_tmp):
                os.makedirs(dir_tmp, exist_ok=True)
            df_tmp.to_csv(f"{dir_tmp}/{self.data_file}", index=False)
            with open(f"{dir_tmp}/metadata.txt", "w") as f:
                f.write(f"file size: {df_tmp.shape[0]} \n")
