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
import pickle
import shutil
import subprocess
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import pandas as pd
import psutil
from nemo.utils import logging
from pympler.asizeof import asizeof

from bionemo.data.protein.openfold.helpers import apply_func_parallel
from bionemo.data.protein.openfold.mmcif import compress_mmcif_dict_atoms, load_mmcif_gz_file, parse_mmcif_string


__all__ = ["PDBMMCIFPreprocess"]


def create_mmcif_dict(
    mmcif_gz_filepath: Path,
    compress_atoms: bool,
) -> Tuple[Optional[dict], dict]:
    """Creates `mmcif_dict` from a `.cif.gz` file.

    Returns tuple: (`mmcif_dict`, `preprocessing_log`).

    `mmcif_dict` will be `None` if it cannot be created.

    `preprocessing_log` contains maximum information always.

    """
    pdb_id = mmcif_gz_filepath.name.split(".")[0]
    subdirname = mmcif_gz_filepath.parent.name

    start_time = time.perf_counter()
    mmcif_string = load_mmcif_gz_file(mmcif_gz_filepath)
    end_time = time.perf_counter()
    loading_time = end_time - start_time

    preprocessing_log = {
        "pdb_id": pdb_id,
        "subdirname": subdirname,
        "mmcif_gz_path": str(mmcif_gz_filepath.relative_to(mmcif_gz_filepath.parents[2])),
        "mmcif_gz_size": mmcif_gz_filepath.stat().st_size,
        "loading_time": f"{loading_time:.6f}",
        "mmcif_string_len": len(mmcif_string),
    }

    try:
        start_time = time.perf_counter()
        mmcif_dict = parse_mmcif_string(mmcif_string)
        end_time = time.perf_counter()
        parsing_time = end_time - start_time
    except Exception as e:
        if isinstance(e, KeyboardInterrupt):
            logging.info("exit(1) due to KeyboardInterrupt")
            exit(1)
        else:
            preprocessing_log["mmcif_dict_size"] = 0
            preprocessing_log["parsing_time"] = float("nan")
            preprocessing_log["error"] = repr(e)
            return None, preprocessing_log

    if compress_atoms:
        mmcif_dict["atoms"] = compress_mmcif_dict_atoms(mmcif_dict["atoms"])

    preprocessing_log["mmcif_dict_size"] = asizeof(mmcif_dict)
    preprocessing_log["parsing_time"] = f"{parsing_time:.6f}"
    preprocessing_log["error"] = None

    return mmcif_dict, preprocessing_log


def load_pdb_cluster_ids(clusters_by_entity_filepath: Path) -> Dict[str, int]:
    cluster_ids = {}
    with open(clusters_by_entity_filepath) as f:
        clusters = f.read().strip().split("\n")
    for cluster_id, cluster in enumerate(clusters):
        pdb_entity_ids = cluster.split()
        for pdb_entity_id in pdb_entity_ids:
            cluster_ids[pdb_entity_id.upper()] = cluster_id
    return cluster_ids


def create_mmcif_chains(
    pdb_mmcif_dicts_filepath: Path,
    pdb_cluster_ids: Dict[str, int],
) -> pd.DataFrame:
    with open(pdb_mmcif_dicts_filepath, "rb") as f:
        mmcif_dicts = pickle.load(f)

    mmcif_chains = []
    for pdb_id, mmcif_dict in mmcif_dicts.items():
        assert pdb_id == mmcif_dict["pdb_id"]

        author_chain_id_to_mmcif_chain_ids = defaultdict(list)
        author_chain_id_to_entity_ids = defaultdict(set)

        mmcif_to_author_mapping = mmcif_dict["mmcif_chain_id_to_author_chain_id"]
        entity_to_mmcifs_mapping = mmcif_dict["entity_id_to_mmcif_chain_ids"]

        for mmcif_chain_id, author_chain_id in mmcif_to_author_mapping.items():
            author_chain_id_to_mmcif_chain_ids[author_chain_id].append(mmcif_chain_id)
            for entity_id, mmcif_chain_ids in entity_to_mmcifs_mapping.items():
                if mmcif_chain_id in mmcif_chain_ids:
                    author_chain_id_to_entity_ids[author_chain_id].add(entity_id)

        for author_chain_id in mmcif_dict["author_chain_ids"]:
            pdb_chain_id = pdb_id + "_" + author_chain_id
            mmcif_chain_ids = author_chain_id_to_mmcif_chain_ids[author_chain_id]
            mmcif_chain_ids = ";".join(mmcif_chain_ids)

            chain_cluster_ids = []
            for entity_id in list(author_chain_id_to_entity_ids[author_chain_id]):
                pdb_entity_id = f"{pdb_id}_{entity_id}".upper()
                if pdb_entity_id in pdb_cluster_ids:
                    chain_cluster_ids.append(pdb_cluster_ids[pdb_entity_id])

            if len(chain_cluster_ids) == 1:
                pdb_cluster_id = chain_cluster_ids[0]
            elif len(chain_cluster_ids) == 0:
                pdb_cluster_id = -1
            else:
                # should never happen,
                # but when it does,
                # count and take the most common id
                pdb_cluster_id = Counter(chain_cluster_ids).most_common()[0][0]

            mmcif_chain = {
                "pdb_chain_id": pdb_chain_id,  # format: `{pdb_id}_{author_chain_id}`
                "pdb_id": pdb_id,
                "author_chain_id": author_chain_id,
                "mmcif_chain_ids": mmcif_chain_ids,
                "release_date": mmcif_dict["release_date"],
                "resolution": mmcif_dict["resolution"],
                "pdb_cluster_id": pdb_cluster_id,
                "sequence_length": len(mmcif_dict["sequences"][author_chain_id]),
                "sequence": mmcif_dict["sequences"][author_chain_id],
            }
            mmcif_chains.append(mmcif_chain)

    mmcif_chains_df = pd.DataFrame(mmcif_chains)
    return mmcif_chains_df


NUM_PHYSICAL_CPU_CORES = psutil.cpu_count(logical=False)


class PDBMMCIFPreprocess:
    def __init__(
        self,
        dataset_root_path: Union[str, Path],
        mmcif_ref: Optional[str],
        pdb_clusters_by_entity_ref: Optional[str],
        pdb_obolete_ref: Optional[str],
        num_parallel_processes: int = NUM_PHYSICAL_CPU_CORES,
        force: bool = False,
    ):
        """
        PDB preprocessing decompresses each .gz file; parses each .cif file; extracts sequence,
        atom positions and metadata like release date or resolution; creates mmcif_dict object
        containing extracted information; groups mmcif_dict's using second and third characters
        of their lowercase four-letter pdb_id code; saves each group inside dicts/ subdirectory
        using pickle module; saves chains.csv dataframe containing chain-level metadata and
        additional information like pdb cluster id; and finally copies obsolete.dat to output
         directory.

        Args:
            dataset_root_path (Union[str, Path]): Path where root directory of raw and processed
            dataset will be stored.
            mmcif_ref (Optional[str]): _description_
            pdb_clusters_by_entity_ref (Optional[str]): _description_
            pdb_obolete_ref (Optional[str]): A link to database that stores information about PDB
            files and their changed IDs.
            num_parallel_processes (int, optional): Number of workers for parallel processing.
            Defaults to NUM_PHYSICAL_CPU_CORES, which is number of physical threads. If you are using
            SLRUM-based cluster, please make sure this is not less than number of threads you were
            assigned.
            force (bool, optional): The processor will raise an exception  if files will be found in
            download directory or in processing directory. Set it to true to allow overriding data.
            Defaults to False.
        """
        self.dataset_root_path = Path(dataset_root_path)
        self.pdb_mmcif_dirpath = self.dataset_root_path / "pdb_mmcif"
        self.original_dirpath = self.pdb_mmcif_dirpath / "original"
        self.processed_dirpath = self.pdb_mmcif_dirpath / "processed"
        self.processed_dirpath.mkdir(exist_ok=True, parents=True)

        self.mmcif_ref = mmcif_ref or "rsync.rcsb.org::ftp_data/structures/divided/mmCIF/"
        self.pdb_clusters_by_entity_ref = (
            pdb_clusters_by_entity_ref or "https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-40.txt"
        )
        self.pdb_obolete_ref = pdb_obolete_ref or "ftp://ftp.wwpdb.org/pub/pdb/data/status/obsolete.dat"

        self.pdb_clusters_by_entity_filepath = self.original_dirpath / os.path.basename(
            self.pdb_clusters_by_entity_ref
        )
        self.pdb_obsolete_filepath = self.original_dirpath / os.path.basename(self.pdb_obolete_ref)

        self.num_parallel_processes = num_parallel_processes
        self.force = force

    def download(self):
        self.original_dirpath.mkdir(exist_ok=True, parents=True)

        if any(self.original_dirpath.iterdir()) and not self.force:
            raise FileExistsError(
                "Download directory is non-empty. Download aborted." "Use 'force' flag to override this behavior"
            )

        # 1. Download PDB MMCIFs

        rsync_mmcif_cmd = [
            "rsync",
            "--recursive",
            "--links",
            "--perms",
            "--times",
            "--compress",
            "--info=progress2",
            "--delete",
            "--port=33444",
        ]
        # add source and donwload location
        pdb_mmcif_raw_dirpath = self.original_dirpath / "raw"
        rsync_mmcif_cmd.extend([self.mmcif_ref, str(pdb_mmcif_raw_dirpath)])
        rsync_mmcif_cmd_txt = " ".join(rsync_mmcif_cmd)

        logging.info(f"Running {rsync_mmcif_cmd_txt}")
        # raise unhandled if download fails
        subprocess.check_output(rsync_mmcif_cmd)

        # 2. Download "obsolete" file - this file maps old PDB ids to new ones

        urllib.request.urlretrieve(self.pdb_obolete_ref, self.pdb_obsolete_filepath)

        # 3. Download clusters-by-entity

        urllib.request.urlretrieve(self.pdb_clusters_by_entity_ref, self.pdb_clusters_by_entity_filepath)

    def prepare(self, download: bool = False, preprocess: bool = False):
        if download:
            self.download()

        if preprocess:
            self.preprocess()

    def preprocess(self):
        logging.info("preprocess_pdb_mmcif has started...")

        if not self.pdb_mmcif_dirpath.exists():
            raise FileNotFoundError(f"{repr(self.pdb_mmcif_dirpath)} does not exist!")
        if not self.pdb_clusters_by_entity_filepath.exists():
            raise FileNotFoundError(f"{repr(self.pdb_clusters_by_entity_filepath)} does not exist!")
        if not self.pdb_obsolete_filepath.exists():
            raise FileNotFoundError(f"{repr(self.pdb_obsolete_filepath)} does not exist!")

        pdb_mmcif_raw_dirpath = self.original_dirpath / "raw"
        if not pdb_mmcif_raw_dirpath.exists():
            raise FileNotFoundError(f"{repr(pdb_mmcif_raw_dirpath)} does not exist!")

        self.processed_dirpath.mkdir(exist_ok=True, parents=True)
        if any(self.processed_dirpath.iterdir()) and not self.force:
            raise FileExistsError(
                "Processing directory is non-empty. Processing aborted. \
                             Use 'force' flag to override this behavior"
            )

        pdb_mmcif_raw_subdirpaths = sorted(pdb_mmcif_raw_dirpath.glob("*"))
        pdb_mmcif_raw_subdirpaths = [subdirpath for subdirpath in pdb_mmcif_raw_subdirpaths if subdirpath.is_dir()]
        num_mmcif_gz_files = sum(len(list(subdirpath.glob("*.cif.gz"))) for subdirpath in pdb_mmcif_raw_subdirpaths)
        logging.info(
            f"Found {len(pdb_mmcif_raw_subdirpaths)} subdirectories"
            f" inside {repr(pdb_mmcif_raw_dirpath)}"
            f" containing {num_mmcif_gz_files} `.cif.gz` files in total."
        )

        pdb_mmcif_dicts_dirpath = self.processed_dirpath / "dicts"
        pdb_mmcif_dicts_dirpath.mkdir(exist_ok=True)
        logging.info(f"mmcif dicts will be saved to {repr(pdb_mmcif_dicts_dirpath)}")

        logging.info("Preprocessing (creating mmcif dicts)...")
        preprocessing_logs_dfs = apply_func_parallel(
            func=self._create_mmcif_dicts,
            args_list=[
                (pdb_mmcif_raw_subdirpath, pdb_mmcif_dicts_dirpath)
                for pdb_mmcif_raw_subdirpath in pdb_mmcif_raw_subdirpaths
            ],
            num_parallel_processes=self.num_parallel_processes,
        )

        preprocessing_logs_df = pd.concat(preprocessing_logs_dfs)
        summary = preprocessing_logs_df["error"].fillna("SUCCESS").value_counts(dropna=False)
        header = pd.Series(index=["__num_mmcif_gz_files__"], data=[num_mmcif_gz_files])
        preprocessing_logs_filepath = self.processed_dirpath / "dicts_preprocessing_logs.csv"
        preprocessing_logs_df.to_csv(preprocessing_logs_filepath, index=False)
        logging.info("preprocessing_logs_df.shape", preprocessing_logs_df.shape)
        logging.info("Preprocessing summary:")
        logging.info(pd.concat([header, summary]).to_string())
        logging.info(f"Preprocessing logs saved to {repr(preprocessing_logs_filepath)} successfully!")

        pdb_cluster_ids = load_pdb_cluster_ids(self.pdb_clusters_by_entity_filepath)
        logging.info("Generating mmcif chains...")
        pdb_mmcif_dicts_filepaths = sorted(pdb_mmcif_dicts_dirpath.glob("*"))
        mmcif_chains_dfs = apply_func_parallel(
            func=create_mmcif_chains,
            args_list=[
                (pdb_mmcif_dicts_filepath, pdb_cluster_ids) for pdb_mmcif_dicts_filepath in pdb_mmcif_dicts_filepaths
            ],
            num_parallel_processes=self.num_parallel_processes,
        )
        mmcif_chains_df = pd.concat(mmcif_chains_dfs)
        pdb_mmcif_chains_filepath = self.processed_dirpath / "chains.csv"
        mmcif_chains_df.to_csv(pdb_mmcif_chains_filepath, index=False)
        logging.info(f"mmcif chains saved to {repr(pdb_mmcif_chains_filepath)} successfully!")

        logging.info("copying pdb obsolete file...")
        src_pdb_obsolete_filepath = self.pdb_obsolete_filepath
        dst_pdb_obsolete_filepath = self.processed_dirpath / self.pdb_obsolete_filepath.name
        shutil.copyfile(src=src_pdb_obsolete_filepath, dst=dst_pdb_obsolete_filepath)
        logging.info(f"pdb obsolete file copied to {repr(dst_pdb_obsolete_filepath)} successfully!")

        logging.info("preprocess_pdb_mmcif finished successfully!")

    def _create_mmcif_dicts(self, pdb_mmcif_raw_subdirpath: Path, pdb_mmcif_dicts_dirpath: Path) -> pd.DataFrame:
        mmcif_dicts = {}
        preprocessing_logs = []

        subdirname = pdb_mmcif_raw_subdirpath.name
        mmcif_gz_filepaths = sorted(pdb_mmcif_raw_subdirpath.glob("*.cif.gz"))
        for mmcif_gz_filepath in mmcif_gz_filepaths:
            pdb_id = mmcif_gz_filepath.name.split(".")[0]

            mmcif_dict, preprocessing_log = create_mmcif_dict(
                mmcif_gz_filepath=mmcif_gz_filepath,
                compress_atoms=True,
            )

            if mmcif_dict is not None:
                assert pdb_id == mmcif_dict["pdb_id"]
                assert pdb_id not in mmcif_dicts
                mmcif_dicts[pdb_id] = mmcif_dict

            assert pdb_id == preprocessing_log["pdb_id"]
            assert subdirname == preprocessing_log["subdirname"]
            preprocessing_logs.append(preprocessing_log)

        output_filepath = pdb_mmcif_dicts_dirpath / subdirname
        if not self.force:
            if output_filepath.exists():
                raise FileExistsError(f"{output_filepath} already exists.")
        with open(output_filepath, "wb") as f:
            pickle.dump(mmcif_dicts, f)

        preprocessing_logs_df = pd.DataFrame(preprocessing_logs)
        return preprocessing_logs_df
