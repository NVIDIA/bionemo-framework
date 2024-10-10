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


import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import psutil
from nemo.utils import logging
from tqdm import tqdm

from bionemo.data.protein.openfold.helpers import (
    apply_func_parallel,
    flatten_list,
    hash_string_into_number,
    split_list_into_n_chunks,
)
from bionemo.data.protein.openfold.protein import Protein


__all__ = ["OpenProteinSetPreprocess"]

NUM_PHYSICAL_CPU_CORES = psutil.cpu_count(logical=False)


def _verify_uniclust30_targets(
    uniclust30_target_pdb_filepaths: List[Path],
) -> List[bool]:
    output_flags = []
    for uniclust30_target_pdb_filepath in uniclust30_target_pdb_filepaths:
        with open(uniclust30_target_pdb_filepath, "rb") as f:
            pdb_string = f.read().decode("utf-8")
        protein = Protein.from_pdb_string(pdb_str=pdb_string)
        atom_positions_has_nans = np.any(np.isnan(protein.atom_positions))
        is_correct = not atom_positions_has_nans
        output_flags.append(is_correct)
    return output_flags


def load_pdb_chain_clusters(pdb_chain_clusters_filepath: Path) -> List[List[str]]:
    with open(pdb_chain_clusters_filepath) as f:
        pdb_chain_clusters = f.read().strip().split("\n")
    pdb_chain_clusters = [pdb_chain_cluster.split() for pdb_chain_cluster in pdb_chain_clusters]
    return pdb_chain_clusters


def load_uniclust30_target(targets_super_index: dict, target_key: str, uniclust30_targets_dirpath: Path) -> bytes:
    """Load uniclust30 target from bytes"""
    targets_index = targets_super_index[target_key]
    targets_db_path = uniclust30_targets_dirpath / targets_index["db"]
    assert len(targets_index["files"]) == 1
    file_index = targets_index["files"][0]
    filename, start, size = file_index
    with open(targets_db_path, "rb") as f:
        f.seek(start)
        pdb_bytes = f.read(size)
    return pdb_bytes


class OpenProteinSetPreprocess:
    S3_BUCKET = "s3://openfold"

    def __init__(
        self,
        dataset_root_path: Union[str, Path],
        num_shards: int = 10,
        num_parallel_processes: int = NUM_PHYSICAL_CPU_CORES,
        force: bool = False,
    ):
        """
        OpenProteinSet preprocessing restores mapping from all pdb chains to their alignments
        using clusters from duplicate_pdb_chains.txt; shards pdb alignment files into configurable
        number of .db files (default is 10) into pdb_alignments/ subdirectory; and finally saves
        JSON-like super.index file next to .db files, mapping from pdb_chain_id to specific .db
        filename and byte location inside.

        Args:
            dataset_root_path (Union[str, Path]): Path where root directory of raw and processed
            dataset will be stored.
            num_shards (int): number of shards pdbs should be put into.
            num_parallel_processes (int, optional): Number of workers for parallel processing.
            Defaults to NUM_PHYSICAL_CPU_CORES, which is number of physical threads. If you are using
            SLRUM-based cluster, please make sure this is not less than number of threads you were
            assigned.
            force (bool, optional): The processor will raise an exception  if files will be found in
            download directory  or in processing directory. Set it to true to allow overriding data.
            Defaults to False.
        """

        self.dataset_root_path = Path(dataset_root_path)
        self.open_protein_set_dirpath = self.dataset_root_path / "open_protein_set"
        self.original_dirpath = self.open_protein_set_dirpath / "original"
        self.processed_dirpath = self.open_protein_set_dirpath / "processed"
        self.processed_dirpath.mkdir(exist_ok=True, parents=True)

        self.num_shards = num_shards
        self.num_parallel_processes = num_parallel_processes
        self.force = force

    def _download_s3(self, filepath, recursive=False):
        bucket_filepath = os.path.join(self.S3_BUCKET, filepath)
        local_filepath = os.path.join(self.original_dirpath, filepath)
        cmd = ["aws", "s3", "cp", "--no-sign-request", str(bucket_filepath), str(local_filepath)]
        if recursive:
            cmd += ["--recursive"]

        logging.info(f"Running {cmd}")
        # do not try to recover if subcommand fails
        subprocess.check_output(cmd)

    def download(self):
        self.original_dirpath.mkdir(exist_ok=True, parents=True)
        if any(self.original_dirpath.iterdir()) and not self.force:
            raise FileExistsError(
                "Download directory is non-empty. Processing aborted. \
                             Use 'force' flag to override this behavior"
            )

        logging.warning(
            "The following script is going to download around 1.1 TB of data."
            "Please make sure you have anough space in destination path."
        )
        # download root files:
        self._download_s3("LICENSE")
        self._download_s3("duplicate_pdb_chains.txt")

        # download pdb directory:
        self._download_s3("pdb", recursive=True)

        # download uniclust30 directory:

        self._download_s3("uniclust30", recursive=True)

    def prepare(
        self,
        download=False,
        preprocess_pdb_alignments=True,
        preprocess_uniclust30_alignments=True,
        preprocess_uniclust30_targets=True,
    ):
        if download:
            self.download()

        if preprocess_pdb_alignments:
            self.preprocess_open_protein_set_pdb_alignments()

        # Prepare self-distilation (UniClust30) required for fine-tuning.
        # Alignments here mean a3m and hhr files, while targets are
        # predicted structures in pdb format.

        if preprocess_uniclust30_alignments:
            self.preprocess_open_protein_set_uniclust30_alignments()

        if preprocess_uniclust30_targets:
            self.preprocess_open_protein_set_uniclust30_targets()

    def preprocess_open_protein_set_uniclust30_targets(self):
        logging.info("Uniclust30 targets preprocessing has started...")

        output_uniclust30_targets_dirpath = self.processed_dirpath / "uniclust30_targets"
        output_uniclust30_targets_dirpath.mkdir(exist_ok=True)

        logging.info(f"output uniclust30 targets shards will be saved to {output_uniclust30_targets_dirpath}")

        if any(output_uniclust30_targets_dirpath.iterdir()) and not self.force:
            raise FileExistsError(
                "uniclust30_targets directory is non-empty. Processing aborted. \
                             Use 'force' flag to override this behavior"
            )

        open_protein_set_uniclust30_dirpath = self.original_dirpath / "uniclust30"
        if not open_protein_set_uniclust30_dirpath.exists():
            raise FileNotFoundError(f"{open_protein_set_uniclust30_dirpath} does not exist!")

        open_protein_set_uniclust30_subdirpaths = sorted(open_protein_set_uniclust30_dirpath.glob("*"))
        logging.info(
            f"Found {len(open_protein_set_uniclust30_subdirpaths)} subdirectories"
            f" inside {open_protein_set_uniclust30_dirpath}."
        )

        all_uniclust30_target_pdb_filepaths = []
        for open_protein_set_uniclust30_subdirpath in tqdm(open_protein_set_uniclust30_subdirpaths):
            target_pdb_filepaths = sorted(open_protein_set_uniclust30_subdirpath.glob("pdb/*.pdb"))
            all_uniclust30_target_pdb_filepaths.extend(target_pdb_filepaths)
        logging.info(
            f"Found {len(all_uniclust30_target_pdb_filepaths)} pdb files"
            f" inside {len(open_protein_set_uniclust30_subdirpaths)} subdirectories"
            f" inside {open_protein_set_uniclust30_dirpath}."
        )

        logging.info("verifying uniclust30 targets...")
        verification_flags = flatten_list(
            apply_func_parallel(
                func=_verify_uniclust30_targets,
                args_list=[
                    (uniclust30_target_pdb_filepaths_chunk,)
                    for uniclust30_target_pdb_filepaths_chunk in split_list_into_n_chunks(
                        all_uniclust30_target_pdb_filepaths,
                        n=(self.num_parallel_processes * 10),
                    )
                ],
                num_parallel_processes=self.num_parallel_processes,
            )
        )
        verified_uniclust30_target_pdb_filepaths = [
            uniclust30_target_pdb_filepath
            for uniclust30_target_pdb_filepath, is_correct in zip(
                all_uniclust30_target_pdb_filepaths, verification_flags
            )
            if is_correct
        ]
        logging.info(f"{len(verified_uniclust30_target_pdb_filepaths)} pdb files passed verification.")

        shards = {shard_id: {} for shard_id in range(self.num_shards)}
        logging.info("sharding uniclust30 targets...")
        for pdb_filepath in tqdm(verified_uniclust30_target_pdb_filepaths):
            dirname = pdb_filepath.parent.parent.name
            shard_id = hash_string_into_number(dirname) % self.num_shards
            assert dirname not in shards[shard_id]
            shards[shard_id][dirname] = [pdb_filepath]
        logging.info("uniclust30 targets shards:")
        for shard_id, shard in shards.items():
            logging.info(f"shard_id={shard_id} len(shard)={len(shard)}")

        logging.info("writing uniclust30 targets shards...")
        shard_index_list = apply_func_parallel(
            func=self._write_uniclust30_targets_shard,
            args_list=list(shards.items()),
            num_parallel_processes=self.num_parallel_processes,
        )
        uniclust30_targets_super_index = {}
        for shard_index in shard_index_list:
            for dirname, targets_index in shard_index.items():
                assert dirname not in uniclust30_targets_super_index
                uniclust30_targets_super_index[dirname] = targets_index
        logging.info(f"len(uniclust30_targets_super_index)={len(uniclust30_targets_super_index)}")

        uniclust30_targets_super_index_filepath = output_uniclust30_targets_dirpath / "super.index"
        if not self.force:
            assert not uniclust30_targets_super_index_filepath.exists()
        with open(uniclust30_targets_super_index_filepath, "w") as f:
            json.dump(uniclust30_targets_super_index, f)
        logging.info(
            f"uniclust30 targets super index saved to {uniclust30_targets_super_index_filepath} successfully!"
        )

        logging.info("preprocess_open_protein_set_uniclust30_targets finished successfully!")

    def preprocess_open_protein_set_pdb_alignments(self):
        logging.info("preprocess_open_protein_set_pdb_alignments has started...")

        open_protein_set_pdb_dirpath = self.original_dirpath / "pdb"
        if not open_protein_set_pdb_dirpath.exists():
            raise FileNotFoundError(f"{open_protein_set_pdb_dirpath} does not exist!")

        open_protein_set_pdb_subdirpaths = sorted(open_protein_set_pdb_dirpath.glob("*"))
        logging.info(
            f"Found {len(open_protein_set_pdb_subdirpaths)} subdirectories" f" inside {open_protein_set_pdb_dirpath}."
        )

        output_pdb_alignments_dirpath = self.processed_dirpath / "pdb_alignments"
        output_pdb_alignments_dirpath.mkdir(exist_ok=True)

        logging.info(f"output pdb alignments shards will be saved to {output_pdb_alignments_dirpath}")

        if any(output_pdb_alignments_dirpath.iterdir()) and not self.force:
            raise FileExistsError(
                "pdb_alignments directory is non-empty. Processing aborted. \
                             Use 'force' flag to override this behavior"
            )

        pdb_chain_clusters_filepath = self.original_dirpath / "duplicate_pdb_chains.txt"
        pdb_chain_clusters = load_pdb_chain_clusters(pdb_chain_clusters_filepath=pdb_chain_clusters_filepath)
        pdb_chain_id_to_cluster_index = {}
        for cluster_index, pdb_chain_cluster in enumerate(pdb_chain_clusters):
            for pdb_chain_id in pdb_chain_cluster:
                pdb_chain_id_to_cluster_index[pdb_chain_id] = cluster_index

        shards = {shard_id: [] for shard_id in range(self.num_shards)}
        assigned_clusters_indexes = set()
        logging.info("sharding pdb alignments...")
        for open_protein_set_pdb_subdirpath in tqdm(open_protein_set_pdb_subdirpaths):
            pdb_chain_id = open_protein_set_pdb_subdirpath.name
            shard_id = hash_string_into_number(pdb_chain_id) % self.num_shards
            assert pdb_chain_id in pdb_chain_id_to_cluster_index
            cluster_index = pdb_chain_id_to_cluster_index[pdb_chain_id]
            assert cluster_index not in assigned_clusters_indexes
            pdb_chain_cluster = pdb_chain_clusters[cluster_index]
            pdb_subdirpath_cluster_pair = (
                open_protein_set_pdb_subdirpath,
                pdb_chain_cluster,
            )
            shards[shard_id].append(pdb_subdirpath_cluster_pair)
            assigned_clusters_indexes.add(cluster_index)
        logging.info("pdb alignments shards:")
        for shard_id, shard in shards.items():
            logging.info(f"shard_id={shard_id} len(shard)={len(shard)}")

        logging.info("writing pdb alignments shards...")
        shard_index_list = apply_func_parallel(
            func=self._write_pdb_alignments_shard,
            args_list=list(shards.items()),
            num_parallel_processes=self.num_parallel_processes,
        )
        pdb_alignments_super_index = {}
        for shard_index in shard_index_list:
            for pdb_chain_id, alignments_index in shard_index.items():
                assert pdb_chain_id not in pdb_alignments_super_index
                pdb_alignments_super_index[pdb_chain_id] = alignments_index
        logging.info(f"len(pdb_alignments_super_index)={len(pdb_alignments_super_index)}")

        pdb_alignments_super_index_filepath = output_pdb_alignments_dirpath / "super.index"
        if not self.force and pdb_alignments_super_index_filepath.exists():
            raise FileExistsError(
                f"super.index found under {output_pdb_alignments_dirpath} but not forced. \
                                    Processing aborted."
            )
        with open(pdb_alignments_super_index_filepath, "w") as f:
            json.dump(pdb_alignments_super_index, f)
        logging.info(f"pdb alignments super index saved to {pdb_alignments_super_index_filepath} successfully!")

        logging.info("preprocess_open_protein_set_pdb_alignments finished successfully!")

    def preprocess_open_protein_set_uniclust30_alignments(self):
        logging.info("preprocess_open_protein_set_uniclust30_alignments has started...")

        open_protein_set_uniclust30_dirpath = self.original_dirpath / "uniclust30"
        if not open_protein_set_uniclust30_dirpath.exists():
            raise FileNotFoundError(f"{open_protein_set_uniclust30_dirpath} does not exist!")

        output_uniclust30_alignments_dirpath = self.processed_dirpath / "uniclust30_alignments"
        output_uniclust30_alignments_dirpath.mkdir(exist_ok=True)
        if any(output_uniclust30_alignments_dirpath.iterdir()) and not self.force:
            raise FileExistsError(
                "uniclust30_alignments directory is non-empty. Processing aborted. \
                             Use 'force' flag to override this behavior."
            )
        logging.info(f"output uniclust30 alignments shards will be saved to {output_uniclust30_alignments_dirpath}")

        open_protein_set_uniclust30_subdirpaths = sorted(open_protein_set_uniclust30_dirpath.glob("*"))
        logging.info(
            f"Found {len(open_protein_set_uniclust30_subdirpaths)} subdirectories"
            f" inside {open_protein_set_uniclust30_dirpath}."
        )

        shards = {shard_id: [] for shard_id in range(self.num_shards)}
        logging.info("sharding uniclust30 alignments...")
        for open_protein_set_uniclust30_subdirpath in tqdm(open_protein_set_uniclust30_subdirpaths):
            dirname = open_protein_set_uniclust30_subdirpath.name
            shard_id = hash_string_into_number(dirname) % self.num_shards
            shards[shard_id].append(open_protein_set_uniclust30_subdirpath)
        logging.info("uniclust30 alignments shards:")
        for shard_id, shard in shards.items():
            logging.info(f"shard_id={shard_id} len(shard)={len(shard)}")

        logging.info("writing uniclust30 alignments shards...")
        shard_index_list = apply_func_parallel(
            func=self._write_uniclust30_alignments_shard,
            args_list=list(shards.items()),
            num_parallel_processes=self.num_parallel_processes,
        )
        uniclust30_alignments_super_index = {}
        for shard_index in shard_index_list:
            for dirname, alignments_index in shard_index.items():
                assert dirname not in uniclust30_alignments_super_index
                uniclust30_alignments_super_index[dirname] = alignments_index
        logging.info(f"len(uniclust30_alignments_super_index)={len(uniclust30_alignments_super_index)}")

        uniclust30_alignments_super_index_filepath = output_uniclust30_alignments_dirpath / "super.index"
        if not self.force:
            assert not uniclust30_alignments_super_index_filepath.exists()
        with open(uniclust30_alignments_super_index_filepath, "w") as f:
            json.dump(uniclust30_alignments_super_index, f)
        logging.info(
            f"uniclust30 alignments super index saved to {uniclust30_alignments_super_index_filepath} successfully!"
        )

        logging.info("preprocess_open_protein_set_uniclust30_alignments finished successfully!")

    def _write_uniclust30_alignments_shard(self, shard_id: int, shard: List[Path]) -> Dict[str, dict]:
        output_uniclust30_alignments_dirpath = self.processed_dirpath / "uniclust30_alignments"
        shard_index = {}
        shard_filename = f"{shard_id}.db"
        shard_filepath = output_uniclust30_alignments_dirpath / shard_filename
        if not self.force:
            assert not shard_filepath.exists()
        with open(shard_filepath, "wb") as f_out:
            start = 0
            for open_protein_set_uniclust30_subdirpath in shard:
                alignments_index = {
                    "db": shard_filename,
                    "files": [],
                }
                filepaths = sorted(open_protein_set_uniclust30_subdirpath.glob("*/*"))
                for filepath in filepaths:
                    if filepath.suffix in {".a3m", ".hhr"}:
                        filename = filepath.name
                        with open(filepath, "rb") as f_in:
                            filebytes = f_in.read()
                        f_out.write(filebytes)
                        size = len(filebytes)
                        file_index = [filename, start, size]
                        alignments_index["files"].append(file_index)
                        start += size
                dirname = open_protein_set_uniclust30_subdirpath.name
                shard_index[dirname] = alignments_index
        return shard_index

    def _write_uniclust30_targets_shard(self, shard_id: int, shard: Dict[str, List[Path]]) -> Dict[str, dict]:
        shard_index = {}
        shard_filename = f"{shard_id}.db"
        output_uniclust30_targets_dirpath = self.processed_dirpath / "uniclust30_targets"
        shard_filepath = output_uniclust30_targets_dirpath / shard_filename
        if not self.force:
            assert not shard_filepath.exists()
        with open(shard_filepath, "wb") as f_out:
            start = 0
            for dirname, pdb_filepaths in shard.items():
                targets_index = {
                    "db": shard_filename,
                    "files": [],
                }
                for pdb_filepath in pdb_filepaths:
                    assert pdb_filepath.suffix == ".pdb"
                    if pdb_filepath.stem.endswith("_model_5_unrelaxed"):
                        filename = "model_5_unrelaxed.pdb"
                    else:
                        filename = pdb_filepath.stem
                    with open(pdb_filepath, "rb") as f_in:
                        filebytes = f_in.read()
                    f_out.write(filebytes)
                    size = len(filebytes)
                    file_index = [filename, start, size]
                    targets_index["files"].append(file_index)
                    start += size
                shard_index[dirname] = targets_index
        return shard_index

    def _write_pdb_alignments_shard(self, shard_id: int, shard: List[Tuple[Path, List[str]]]) -> Dict[str, dict]:
        output_pdb_alignments_dirpath = self.processed_dirpath / "pdb_alignments"
        shard_index = {}
        shard_filename = f"{shard_id}.db"
        shard_filepath = output_pdb_alignments_dirpath / shard_filename
        if not self.force:
            assert not shard_filepath.exists()
        with open(shard_filepath, "wb") as f_out:
            start = 0
            for pdb_subdirpath_cluster_pair in shard:
                open_protein_set_pdb_subdirpath = pdb_subdirpath_cluster_pair[0]
                pdb_chain_cluster = pdb_subdirpath_cluster_pair[1]
                alignments_index = {
                    "db": shard_filename,
                    "files": [],
                }
                filepaths = sorted(open_protein_set_pdb_subdirpath.glob("*/*"))
                for filepath in filepaths:
                    filename = filepath.name
                    with open(filepath, "rb") as f_in:
                        filebytes = f_in.read()
                    f_out.write(filebytes)
                    size = len(filebytes)
                    file_index = [filename, start, size]
                    alignments_index["files"].append(file_index)
                    start += size
                for pdb_chain_id in pdb_chain_cluster:
                    assert pdb_chain_id not in shard_index
                    shard_index[pdb_chain_id] = alignments_index
        return shard_index
