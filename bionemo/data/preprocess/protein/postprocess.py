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


import hashlib
import json
import pickle
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import pandas as pd
from nemo.utils import logging

from bionemo.data.preprocess.protein.open_protein_set import load_uniclust30_target
from bionemo.data.protein.openfold.alignments import load_alignments, load_alignments_super_index
from bionemo.data.protein.openfold.helpers import hash_string_into_number
from bionemo.data.protein.openfold.mmcif import load_mmcif_chains_df
from bionemo.data.protein.openfold.parsers import parse_hhr
from bionemo.data.protein.openfold.templates import _get_pdb_id_and_chain_id


class OpenFoldSampleCreator:
    """
    OpenFoldSampleCreator allows to create a training data sample for OpenFold.
    It uses full training data and picks selected pdbs to form a small dataset
    with the same structure as full processed training data. For full training
    data creation process, please refer to
    bionemo.data.preprocess.protein.open_protein_set.OpenProteinSetPreprocess
        and
    bionemo.data.preprocess.protein.pdb_mmcif.PDBMMCIFPreprocess
    """

    SAMPLE_PDB_CHAIN_IDS = [
        # 10 chains from date range: ("2018-01-01", "2020-12-31")
        "5vf6_A",
        "5wka_D",
        "5vm0_B",
        "5xjm_A",
        "5y2g_A",
        "6gfe_L",
        "6ilc_B",
        "6ilc_A",
        "6v6a_A",
        "6ljg_A",
        # 22 chains selected by length for perf-infer tests:
        "5xqn_B",
        "6kyu_B",
        "7can_A",
        "6fr5_A",
        "1fdp_A",
        "5b3n_A",
        "5ne1_A",
        "2qu6_B",
        "4b70_A",
        "7ju4_i",
        "5xet_C",
        "4udx_X",
        "5uw7_B",
        "6m8p_K",
        "4res_C",
        "3eql_D",
        "6ta1_E",
        "4o9y_J",
        "6gjc_F",
        "6x32_B",
        "5tan_G",
        "5gky_C",
    ]
    SAMPLE_CAMEO_CHAIN_IDS = [
        # 2 chains for CAMEO target non-overlapping with training date range
        "7f17_B",  # 2021-10-23
        "7lc5_A",  # 2021-10-30
    ]
    SAMPLE_UNICLUST30_IDS = [
        # the first 16 uniclust30 ids
        "A0A009JVE4",
        "A0A010PP53",
        "A0A010Q9I4",
        "A0A010QBD7",
        "A0A010QES4",
        "A0A010QS11",
        "A0A010QU37",
        "A0A010QX93",
        "A0A010R2Y6",
        "A0A010RA97",
        "A0A010RHX7",
        "A0A010RKM9",
        "A0A010RW12",
        "A0A010Z0A0",
        "A0A011NJF4",
        "A0A011PLK7",
    ]

    def __init__(
        self,
        dataset_root_path: Union[str, Path],
        output_root_path: Optional[Union[str, Path]] = None,
        num_shards: int = 3,
        source_variant: str = "processed",
        sample_variant: str = "processed_sample",
        force: bool = False,
    ):
        """Initialises OpenFoldSampleCreator

        Args:
            dataset_root_path (Union[str, Path]): root path for full training data. This corresponds to
            model.data.dataset_path in OpenFold config.
            output_root_path (Optional[Union[str, Path]], optional): where sample is going to be stored.
            If None, sample is created within dataset_root_path. Defaults to None.
            num_shards (int, optional): how many shards should a sample be divided into. Defaults to 3.
            source_variant (str, optional): full training dataset variant name. Defaults to 'processed'.
            sample_variant (str, optional): sample dataset variant name. Defaults to 'processed_sample'.
            force (bool, optional): if set to True and {output_root_path}/{sample_variant} path is
            non-empty, it will override data in this path. Otherwise, sample creation will be aborted.
            Defaults to False.
        """
        self.dataset_root_path = Path(dataset_root_path)
        self.output_root_path = Path(output_root_path if output_root_path else dataset_root_path)
        self.num_shards = num_shards
        self.source_variant = source_variant
        self.sample_variant = sample_variant
        self.force = force

    def prepare(
        self,
        sample_pdb_chain_ids: List[str] = SAMPLE_PDB_CHAIN_IDS,
        sample_cameo_chain_ids: List[str] = SAMPLE_CAMEO_CHAIN_IDS,
        sample_uniclust30_ids: List[str] = SAMPLE_UNICLUST30_IDS,
    ):
        """Prepares sample.

        Args:
            sample_pdb_chain_ids (List[str], optional): List of pdb ids that should be a part
            of sample training data. Requested PDB ids must be present in full data but they might
            be a part of both train and test subsets. Defaults to SAMPLE_PDB_CHAIN_IDS.
            sample_cameo_chain_ids (List[str], optional): List of pdb ids that should be part of
            sample validation data in cameo if within validation date range. Defaults to
            SAMPLE_CAMEO_CHAIN_IDS.
            sample_uniclust30_ids (List[str], optional): List of uniclust30 ids that should be a
            part of sample training data in fine-tuning. Defaults to SAMPLE_UNICLUST30_IDS.

        Raises:
            ValueError: if destination for sample is the same as for full training data -- to prevent
            overriding.
            FileExistsError: If destination for writing sample data is non-empty and force flag is
            turned off
        """
        if sample_pdb_chain_ids is None:
            sample_pdb_chain_ids = self.SAMPLE_PDB_CHAIN_IDS
        if sample_cameo_chain_ids is None:
            sample_cameo_chain_ids = self.SAMPLE_CAMEO_CHAIN_IDS
        if sample_uniclust30_ids is None:
            sample_uniclust30_ids = self.SAMPLE_UNICLUST30_IDS
        logging.info("Generating pdb_mmcif/open_protein_set sample has started...")
        logging.info(f"len(sample_pdb_chain_ids)={len(sample_pdb_chain_ids)}")
        logging.info(f"len(sample_cameo_chain_ids)={len(sample_cameo_chain_ids)}")
        logging.info(f"len(sample_uniclust30_ids)={len(sample_uniclust30_ids)}")

        if self.source_variant == self.sample_variant:
            raise ValueError(
                "source_variant and sample_variant is the same and equal to {self.source_variant}. "
                "That would mix source and target shards and/or override them."
            )

        # In case self.source_variant or self.sample_variant change (intentionally), we create input
        # and output dirs dynamically here

        input_pdb_mmcif_dirpath = self.dataset_root_path / "pdb_mmcif" / self.source_variant
        input_open_protein_set_dirpath = self.dataset_root_path / "open_protein_set" / self.source_variant

        output_pdb_mmcif_dirpath = self.output_root_path / "pdb_mmcif" / self.sample_variant
        output_open_protein_set_dirpath = self.output_root_path / "open_protein_set" / self.sample_variant

        # Create output directories for the `processed_sample` variant:
        output_pdb_mmcif_dirpath.mkdir(exist_ok=True, parents=True)
        output_open_protein_set_dirpath.mkdir(exist_ok=True, parents=True)

        # check if output directories are empty and only proceed if force flag is used

        if any(output_pdb_mmcif_dirpath.iterdir()) and not self.force:
            raise FileExistsError(
                f"Output directory {output_pdb_mmcif_dirpath} for sampled pdb_mmcif is non-empty. \
                Processing aborted.  Use 'force' flag to override this behavior"
            )

        if any(output_open_protein_set_dirpath.iterdir()) and not self.force:
            raise FileExistsError(
                f"Output directory {output_open_protein_set_dirpath} for sampled open_protein_set is non-empty. \
                Processing aborted.  Use 'force' flag to override this behavior"
            )

        # Load mmcif chains:
        input_mmcif_chains_df = load_mmcif_chains_df(
            mmcif_chains_filepath=(input_pdb_mmcif_dirpath / "chains.csv"),
            verbose=True,
        )

        # Load and filter uniclust30_targets for the `processed_sample` variant:
        if sample_uniclust30_ids is not None:
            with open(input_open_protein_set_dirpath / "uniclust30_targets" / "super.index") as f:
                input_uniclust30_targets_super_index = json.load(f)

            output_uniclust30_targets = {}
            for sample_uniclust30_id in sample_uniclust30_ids:
                output_uniclust30_target = load_uniclust30_target(
                    targets_super_index=input_uniclust30_targets_super_index,
                    target_key=sample_uniclust30_id,
                    uniclust30_targets_dirpath=input_open_protein_set_dirpath / "uniclust30_targets",
                )
                output_uniclust30_targets[sample_uniclust30_id] = output_uniclust30_target

        # Load pdb alignments for the `processed_sample` variant:
        output_pdb_alignments = load_processed_sample_alignments(
            input_alignments_dirpath=(input_open_protein_set_dirpath / "pdb_alignments"),
            sample_ids=[*sample_pdb_chain_ids, *sample_cameo_chain_ids],
            verbose=True,
        )

        # Filter mmcif chains for the `processed_sample` variant:
        output_mmcif_chains_df = _filter_mmcif_chains(
            input_mmcif_chains_df=input_mmcif_chains_df,
            output_pdb_alignments=output_pdb_alignments,
            sample_pdb_chain_ids=[*sample_pdb_chain_ids, *sample_cameo_chain_ids],
        )
        logging.info(f"len(output_mmcif_chains_df)={len(output_mmcif_chains_df)}")

        # Filter uniclust30 for the `processed_sample` variant:
        if sample_uniclust30_ids is not None:
            output_uniclust30_alignments = load_processed_sample_alignments(
                input_alignments_dirpath=(input_open_protein_set_dirpath / "uniclust30_alignments"),
                sample_ids=sample_uniclust30_ids,
                verbose=True,
            )

        # Save the `processed_sample` variant of pdb_alignments in `open_protein_set`:
        _save_processed_sample_pdb_alignments(
            output_pdb_alignments=output_pdb_alignments,
            output_pdb_alignments_dirpath=(output_open_protein_set_dirpath / "pdb_alignments"),
            num_shards=self.num_shards,
            force=self.force,
        )

        # Save the `processed_sample` variant of uniclust30_alignments in `open_protein_set`
        if sample_uniclust30_ids is not None:
            _save_processed_sample_pdb_alignments(
                output_pdb_alignments=output_uniclust30_alignments,
                output_pdb_alignments_dirpath=(output_open_protein_set_dirpath / "uniclust30_alignments"),
                num_shards=self.num_shards,
                force=self.force,
            )

            # Save the `processed_sample` variant of uniclust30_targets in `open_protein_set`
            _save_processed_sample_uniclust30_targets(
                input_uniclust30_targets_super_index=input_uniclust30_targets_super_index,
                output_uniclust30_targets=output_uniclust30_targets,
                output_uniclust30_targets_filepath=(output_open_protein_set_dirpath / "uniclust30_targets"),
                num_shards=self.num_shards,
                force=self.force,
            )

        # Save the `processed_sample` variant of `pdb_mmcif`:
        _save_processed_sample_dicts(
            output_mmcif_chains_df=output_mmcif_chains_df,
            input_pdb_mmcif_dirpath=input_pdb_mmcif_dirpath,
            output_pdb_mmcif_dirpath=output_pdb_mmcif_dirpath,
            force=self.force,
        )
        _save_processed_sample_mmcif_chains(
            output_mmcif_chains_df=output_mmcif_chains_df,
            output_pdb_mmcif_dirpath=output_pdb_mmcif_dirpath,
            force=self.force,
        )
        _copy_pdb_obsolete_file(
            input_pdb_mmcif_dirpath=input_pdb_mmcif_dirpath,
            output_pdb_mmcif_dirpath=output_pdb_mmcif_dirpath,
            force=self.force,
        )
        logging.info("generate_processed_sample finished successfully!")


def _save_processed_sample_uniclust30_targets(
    input_uniclust30_targets_super_index: Dict[str, dict],
    output_uniclust30_targets: Dict[str, bytes],
    output_uniclust30_targets_filepath: Path,
    num_shards: int,
    force: bool,
) -> None:
    output_uniclust30_targets_filepath.mkdir(exist_ok=force, parents=True)

    shards = {shard_id: [] for shard_id in range(num_shards)}
    for uniclust30_id, output_uniclust30_target in output_uniclust30_targets.items():
        shard_id = hash_string_into_number(uniclust30_id) % num_shards
        shards[shard_id].append([uniclust30_id, output_uniclust30_target])

    output_uniclust30_targets_super_index = {}

    for shard_id, shard_data in shards.items():
        shard_filepath = output_uniclust30_targets_filepath / f"{shard_id}.db"
        with shard_filepath.open("wb") as f_out:
            start = 0
            for uniclust30_id, output_uniclust30_target in shard_data:
                f_out.write(output_uniclust30_target)

                # Update super index
                target_files = input_uniclust30_targets_super_index[uniclust30_id]["files"]
                assert len(target_files) == 1

                size = len(output_uniclust30_target)
                output_uniclust30_targets_super_index[uniclust30_id] = {
                    "db": f"{shard_id}.db",
                    "files": [[uniclust30_id, start, size]],
                }
                start += size

    with (output_uniclust30_targets_filepath / "super.index").open("w") as f:
        json.dump(output_uniclust30_targets_super_index, f)


def _save_processed_sample_dicts(
    output_mmcif_chains_df: pd.DataFrame,
    input_pdb_mmcif_dirpath: Path,
    output_pdb_mmcif_dirpath: Path,
    force: bool,
) -> None:
    input_dicts_dirpath = input_pdb_mmcif_dirpath / "dicts"
    output_dicts_dirpath = output_pdb_mmcif_dirpath / "dicts"
    output_dicts_dirpath.mkdir(exist_ok=force, parents=True)
    pdb_ids = list(set(output_mmcif_chains_df["pdb_id"].tolist()))
    pdb_ids_grouped = defaultdict(list)
    for pdb_id in pdb_ids:
        filename = pdb_id[1:3]
        pdb_ids_grouped[filename].append(pdb_id)
    for filename, pdb_ids in pdb_ids_grouped.items():
        input_mmcif_dicts_filepath = input_dicts_dirpath / filename
        with open(input_mmcif_dicts_filepath, "rb") as f:
            processed_mmcif_dicts = pickle.load(f)
        processed_sample_mmcif_dicts = {}
        for pdb_id in pdb_ids:
            processed_sample_mmcif_dicts[pdb_id] = processed_mmcif_dicts[pdb_id]
        output_mmcif_dicts_filepath = output_dicts_dirpath / filename
        if not force:
            assert not output_mmcif_dicts_filepath.exists()
        with open(output_mmcif_dicts_filepath, "wb") as f:
            pickle.dump(processed_sample_mmcif_dicts, f)
    logging.info(f"mmcif dicts saved to {output_dicts_dirpath} successfully!")


def _save_processed_sample_mmcif_chains(
    output_mmcif_chains_df: pd.DataFrame,
    output_pdb_mmcif_dirpath: Path,
    force: bool,
) -> None:
    output_pdb_mmcif_chains_filepath = output_pdb_mmcif_dirpath / "chains.csv"
    if not force:
        assert not output_pdb_mmcif_chains_filepath.exists()
    output_mmcif_chains_df.to_csv(output_pdb_mmcif_chains_filepath, index=False)
    logging.info(f"mmcif chains saved to {output_pdb_mmcif_chains_filepath} successfully!")


def _copy_pdb_obsolete_file(
    input_pdb_mmcif_dirpath: Path,
    output_pdb_mmcif_dirpath: Path,
    force: bool,
) -> None:
    input_pdb_obsolete_filepath = input_pdb_mmcif_dirpath / "obsolete.dat"
    output_pdb_obsolete_filepath = output_pdb_mmcif_dirpath / "obsolete.dat"
    if not force:
        assert not output_pdb_obsolete_filepath.exists()
    shutil.copyfile(src=input_pdb_obsolete_filepath, dst=output_pdb_obsolete_filepath)
    logging.info("pdb obsolete file copied to " f"{output_pdb_obsolete_filepath} successfully!")


def _get_template_pdb_chain_ids(output_pdb_alignments: Dict[str, dict]) -> Set[str]:
    sample_template_pdb_chain_ids = set()
    for pdb_chain_id in output_pdb_alignments.keys():
        hhr_string = output_pdb_alignments[pdb_chain_id].get("pdb70_hits.hhr", "")
        template_hits = parse_hhr(hhr_string)
        for template_hit in template_hits:
            template_hit_pdb_chain = _get_pdb_id_and_chain_id(template_hit=template_hit)
            template_hit_pdb_id = template_hit_pdb_chain[0]
            template_hit_chain_id = template_hit_pdb_chain[1]
            template_hit_pdb_chain_id = f"{template_hit_pdb_id}_{template_hit_chain_id}"
            sample_template_pdb_chain_ids.add(template_hit_pdb_chain_id)
    return sample_template_pdb_chain_ids


def _filter_mmcif_chains(
    input_mmcif_chains_df: pd.DataFrame,
    output_pdb_alignments: Dict[str, dict],
    sample_pdb_chain_ids: List[str],
) -> pd.DataFrame:
    sample_primary_pdb_chain_ids = set(sample_pdb_chain_ids)
    sample_template_pdb_chain_ids = _get_template_pdb_chain_ids(output_pdb_alignments)
    is_primary = input_mmcif_chains_df["pdb_chain_id"].isin(sample_primary_pdb_chain_ids)
    is_template = input_mmcif_chains_df["pdb_chain_id"].isin(sample_template_pdb_chain_ids)
    selector = is_primary | is_template
    output_mmcif_chains_df = input_mmcif_chains_df[selector].copy()
    return output_mmcif_chains_df


def _save_processed_sample_pdb_alignments(
    output_pdb_alignments: Dict[str, dict],
    output_pdb_alignments_dirpath: Path,
    num_shards: int,
    force: bool,
    chunk_size: int = 2**16,
) -> None:
    shards = get_shards(
        output_pdb_alignments=output_pdb_alignments,
        num_shards=num_shards,
    )
    output_pdb_alignments_dirpath.mkdir(exist_ok=force, parents=True)
    alignments_super_index = {}
    for shard_id, shard_alignments in shards.items():
        alignments_shard_filename = f"{shard_id}.db"
        alignments_shard_filepath = output_pdb_alignments_dirpath / alignments_shard_filename
        if not force:
            assert not alignments_shard_filepath.exists()
        with open(alignments_shard_filepath, "wb") as f_out:
            start = 0
            for shard_alignment in shard_alignments:
                alignments, pdb_chain_cluster = shard_alignment
                alignments_index = {
                    "db": alignments_shard_filename,
                    "files": [],
                }
                filenames = sorted(alignments.keys())
                for filename in filenames:
                    filebytes = alignments[filename].encode("utf-8")
                    size = len(filebytes)
                    for chunk_i in range(0, size, chunk_size):
                        f_out.write(filebytes[chunk_i : chunk_i + chunk_size])
                    file_index = [filename, start, size]
                    alignments_index["files"].append(file_index)
                    start += size
                for pdb_chain_id in pdb_chain_cluster:
                    assert pdb_chain_id not in alignments_super_index
                    alignments_super_index[pdb_chain_id] = alignments_index
        logging.info(f"alignments shard saved to {alignments_shard_filepath} successfully!")
    alignments_super_index_filepath = output_pdb_alignments_dirpath / "super.index"
    if not force:
        assert not alignments_super_index_filepath.exists()
    with open(alignments_super_index_filepath, "w") as f:
        json.dump(alignments_super_index, f)
    logging.info("alignments super index saved to " f"{alignments_super_index_filepath} successfully!")


def load_processed_sample_alignments(
    input_alignments_dirpath: Path,
    sample_ids: List[str],
    verbose: bool,
) -> Dict[str, dict]:
    pdb_alignments_super_index = load_alignments_super_index(
        alignments_super_index_filepath=(input_alignments_dirpath / "super.index"),
        verbose=verbose,
    )
    processed_sample_alignments = {}
    for sample_id in sample_ids:
        processed_sample_alignments[sample_id] = load_alignments(
            alignments_super_index=pdb_alignments_super_index,
            alignments_dirpath=input_alignments_dirpath,
            key=sample_id,
        )
    return processed_sample_alignments


def get_alignments_hash(alignments: dict) -> str:
    alignments_bytes = b""
    for filename in sorted(alignments.keys()):
        alignments_bytes += filename.encode("utf-8")
        alignments_bytes += alignments[filename].encode("utf-8")
    alignments_hash = hashlib.sha256(alignments_bytes).hexdigest()
    return alignments_hash


def get_shards(
    output_pdb_alignments: Dict[str, dict],
    num_shards: int,
) -> Dict[int, List[Tuple[dict, List[str]]]]:
    # Deduplicate pdb alignments:
    pdb_alignments_grouped = {}
    pdb_chain_clusters = defaultdict(list)
    for pdb_chain_id in output_pdb_alignments.keys():
        alignments = output_pdb_alignments[pdb_chain_id]
        alignments_hash = get_alignments_hash(alignments)
        pdb_chain_clusters[alignments_hash].append(pdb_chain_id)
        pdb_alignments_grouped[alignments_hash] = alignments
    logging.info(f"len(pdb_alignments_grouped)={len(pdb_alignments_grouped)}")
    logging.info(f"len(pdb_chain_clusters)={len(pdb_chain_clusters)}")
    # Assign pdb alignments to shards:
    shards = {shard_id: [] for shard_id in range(num_shards)}
    for alignments_hash in pdb_chain_clusters:
        shard_id = hash_string_into_number(alignments_hash) % num_shards
        alignments = pdb_alignments_grouped[alignments_hash]
        pdb_chain_cluster = pdb_chain_clusters[alignments_hash]
        shard_alignment = (alignments, pdb_chain_cluster)
        shards[shard_id].append(shard_alignment)
    logging.info("shards:")
    for shard_id, shard_alignments in shards.items():
        logging.info(f"shard_id={shard_id} len(shard_alignments)={len(shard_alignments)}")
    return shards
