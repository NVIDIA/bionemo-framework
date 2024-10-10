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


import datetime
import json
import math
from collections import Counter
from importlib import resources
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from nemo.utils import logging
from omegaconf import DictConfig
from torch.utils.data import Dataset

from bionemo.data.preprocess.protein.open_protein_set import load_uniclust30_target
from bionemo.data.protein.openfold.alignments import load_alignments, load_alignments_super_index
from bionemo.data.protein.openfold.features import (
    create_mmcif_features,
    create_msa_features,
    create_sequence_features,
    create_target_features,
    create_template_features,
    process_features,
)
from bionemo.data.protein.openfold.helpers import datetime_from_string
from bionemo.data.protein.openfold.mmcif import load_mmcif_chains_df, load_mmcif_dict
from bionemo.data.protein.openfold.parsers import parse_fasta
from bionemo.data.protein.openfold.protein import Protein
from bionemo.data.protein.openfold.templates import TemplateHitFeaturizer, create_empty_template_feats
from bionemo.data.protein.openfold.tools.hhsearch import HHSearch


class PDBDataset(Dataset):
    """Dataset containing Protein Data Bank (PDB) structures."""

    def __init__(
        self,
        mmcif_chains_df: pd.DataFrame,
        alignments_super_index: Dict[str, dict],
        pdb_mmcif_dicts_dirpath: Path,
        pdb_alignments_dirpath: Path,
        template_hit_featurizer: TemplateHitFeaturizer,
        alphafold_config: Union[DictConfig, Dict],
        mode: str,  # "train" or "eval"
        verbose: bool = False,
        name: str = "InitialDataset",
    ) -> None:
        assert mode in {"train", "eval"}
        self.mmcif_chains = mmcif_chains_df.to_dict("records")
        self.alignments_super_index = alignments_super_index
        self.pdb_mmcif_dicts_dirpath = pdb_mmcif_dicts_dirpath
        self.pdb_alignments_dirpath = pdb_alignments_dirpath
        self.template_hit_featurizer = template_hit_featurizer
        self.alphafold_config = alphafold_config
        self.mode = mode
        self.verbose = verbose
        self.name = name
        if verbose:
            logging.debug(f"{name}: initialized successfully!")

    def __getitem__(self, index_seed_pair: Tuple[int, int]) -> dict:
        if not isinstance(index_seed_pair, tuple):
            raise TypeError(
                f"__getitem__ expects {tuple} in format (index, seed),"
                f" but provided {type(index_seed_pair)} argument"
                f" contains value {repr(index_seed_pair)}"
            )
        index, seed = index_seed_pair
        assert isinstance(index, int)
        assert isinstance(seed, int)

        # Get sample metadata:
        sample = self.mmcif_chains[index]
        sequence = sample["sequence"]
        pdb_id = sample["pdb_id"]
        pdb_chain_id = sample["pdb_chain_id"]
        author_chain_id = sample["author_chain_id"]
        release_date = sample["release_date"]
        seqlen = len(sequence)

        # Load sample data (mmcif):
        mmcif_dict = load_mmcif_dict(
            mmcif_dicts_dirpath=self.pdb_mmcif_dicts_dirpath,
            pdb_id=pdb_id,
        )

        # Load sample data (alignments):
        alignments = load_alignments(
            alignments_super_index=self.alignments_super_index,
            alignments_dirpath=self.pdb_alignments_dirpath,
            key=pdb_chain_id,
        )

        # Create mmCIF features:
        mmcif_features = create_mmcif_features(
            mmcif_dict=mmcif_dict,
            author_chain_id=author_chain_id,
        )

        # Create template features:
        template_features = create_template_features(
            sequence=sequence,
            hhr_string=alignments.get("pdb70_hits.hhr", ""),
            template_hit_featurizer=self.template_hit_featurizer,
            release_date=release_date,
            pdb_id=pdb_id,
            days_before_release=60,
            shuffling_seed=seed,
        )

        # Create MSA features:
        msa_features = create_msa_features(
            sequence=sequence,
            a3m_strings=[
                alignments.get("uniref90_hits.a3m", ""),
                alignments.get("bfd_uniclust_hits.a3m", ""),
                alignments.get("mgnify_hits.a3m", ""),
            ],
        )

        # Process features:
        raw_features = {**mmcif_features, **template_features, **msa_features}
        feats = process_features(
            raw_features=raw_features,
            alphafold_config=self.alphafold_config,
            mode=self.mode,
            seed=seed,
        )

        # Add id tuple:
        feats["id"] = (self.name, index, seed, pdb_chain_id, seqlen)

        return feats

    def __len__(self) -> int:
        return len(self.mmcif_chains)


class InitialTrainingDataset(PDBDataset):
    """Dataset for the initial training stage."""

    def __init__(
        self,
        pdb_mmcif_chains_filepath: Path,
        pdb_mmcif_dicts_dirpath: Path,
        pdb_obsolete_filepath: Path,
        pdb_alignments_dirpath: Path,
        max_pdb_release_date: str,
        alphafold_config: Union[DictConfig, Dict],
        filter_by_alignments: bool = False,
        use_only_pdb_chain_ids: Optional[List[str]] = None,
        realign_when_required: bool = True,
        verbose: bool = False,
        name: str = "InitialTrainingDataset",
    ) -> None:
        if verbose:
            logging.debug(f"{name}: initialization...")

        # Load pdb mmcif chains metadata:
        pdb_mmcif_chains_df = load_mmcif_chains_df(
            mmcif_chains_filepath=pdb_mmcif_chains_filepath,
            verbose=verbose,
            pprefix=f"{name}: ",
        )

        # Load alignments metadata:
        alignments_super_index = load_alignments_super_index(
            alignments_super_index_filepath=(pdb_alignments_dirpath / "super.index"),
            verbose=verbose,
            pprefix=f"{name}: ",
        )

        # Select pdb chains used as training samples:
        mmcif_chains_df = _filter_pdb_chains_for_training(
            mmcif_chains_df=pdb_mmcif_chains_df,
            min_release_date="1900-01-01",
            max_release_date=max_pdb_release_date,
        )

        if use_only_pdb_chain_ids is not None:
            assert isinstance(use_only_pdb_chain_ids, list)
            selector = mmcif_chains_df["pdb_chain_id"].isin(set(use_only_pdb_chain_ids))
            mmcif_chains_df = mmcif_chains_df[selector].copy()

        if filter_by_alignments:
            mmcif_chains_df = _filter_by_alignments(
                mmcif_chains_df=mmcif_chains_df,
                alignments_super_index=alignments_super_index,
            )
        else:
            _assert_pdb_alignments(
                mmcif_chains_df=mmcif_chains_df,
                alignments_super_index=alignments_super_index,
            )

        if verbose:
            logging.debug(f"{name}: mmcif_chains_df.shape={mmcif_chains_df.shape}")

        # Compute pdb cluster size:
        mmcif_chains_df = _compute_pdb_cluster_size(mmcif_chains_df)

        if verbose:
            logging.debug(f"{name}: mmcif_chains_df.shape={mmcif_chains_df.shape}")

        # Create template hit featurizer:
        template_hit_featurizer = TemplateHitFeaturizer(
            max_template_hits=alphafold_config.max_templates,
            pdb_mmcif_dicts_dirpath=pdb_mmcif_dicts_dirpath,
            template_pdb_chain_ids=set(pdb_mmcif_chains_df["pdb_chain_id"]),
            pdb_release_dates=_get_pdb_release_dates(pdb_mmcif_chains_df),
            pdb_obsolete_filepath=pdb_obsolete_filepath,
            shuffle_top_k_prefiltered=alphafold_config.shuffle_top_k_prefiltered,
            realign_when_required=realign_when_required,
            verbose=False,
        )

        super().__init__(
            mmcif_chains_df=mmcif_chains_df,
            alignments_super_index=alignments_super_index,
            pdb_mmcif_dicts_dirpath=pdb_mmcif_dicts_dirpath,
            pdb_alignments_dirpath=pdb_alignments_dirpath,
            template_hit_featurizer=template_hit_featurizer,
            alphafold_config=alphafold_config,
            mode="train",
            verbose=verbose,
            name=name,
        )

    def get_sampler_weights(self) -> torch.Tensor:
        """Get weights for training sampler (Supplementary '1.2.5 Filtering')."""
        return torch.tensor(
            data=[_get_weight(sample) for sample in self.mmcif_chains],
            dtype=torch.float64,
        )


class ValidationDataset(PDBDataset):
    """Validation dataset."""

    def __init__(
        self,
        pdb_mmcif_chains_filepath: Path,
        pdb_mmcif_dicts_dirpath: Path,
        pdb_obsolete_filepath: Path,
        pdb_alignments_dirpath: Path,
        min_cameo_submission_date: str,
        max_cameo_submission_date: str,
        max_sequence_length: int,
        alphafold_config: Union[DictConfig, Dict],
        filter_by_alignments: bool = False,
        use_only_pdb_chain_ids: Optional[List[str]] = None,
        realign_when_required: bool = True,
        verbose: bool = False,
        name: str = "ValidationDataset",
    ) -> None:
        if verbose:
            logging.debug(f"{name}: initialization...")

        # Load pdb mmcif chains metadata:
        pdb_mmcif_chains_df = load_mmcif_chains_df(
            mmcif_chains_filepath=pdb_mmcif_chains_filepath,
            verbose=verbose,
            pprefix=f"{name}: ",
        )

        # Load alignments metadata:
        alignments_super_index = load_alignments_super_index(
            alignments_super_index_filepath=(pdb_alignments_dirpath / "super.index"),
            verbose=verbose,
            pprefix=f"{name}: ",
        )

        # Select pdb chains used as validation samples:
        mmcif_chains_df = _select_cameo_targets_for_validation(
            mmcif_chains_df=pdb_mmcif_chains_df,
            min_submission_date=min_cameo_submission_date,
            max_submission_date=max_cameo_submission_date,
            max_sequence_length=max_sequence_length,
        )

        if use_only_pdb_chain_ids is not None:
            assert isinstance(use_only_pdb_chain_ids, list)
            selector = mmcif_chains_df["pdb_chain_id"].isin(set(use_only_pdb_chain_ids))
            mmcif_chains_df = mmcif_chains_df[selector].copy()

        if filter_by_alignments:
            mmcif_chains_df = _filter_by_alignments(
                mmcif_chains_df=mmcif_chains_df,
                alignments_super_index=alignments_super_index,
            )
        else:
            _assert_pdb_alignments(
                mmcif_chains_df=mmcif_chains_df,
                alignments_super_index=alignments_super_index,
            )

        if verbose:
            logging.debug(f"{name}: mmcif_chains_df.shape={mmcif_chains_df.shape}")

        # Create template hit featurizer:
        template_hit_featurizer = TemplateHitFeaturizer(
            max_template_hits=alphafold_config.max_templates,
            pdb_mmcif_dicts_dirpath=pdb_mmcif_dicts_dirpath,
            template_pdb_chain_ids=set(pdb_mmcif_chains_df["pdb_chain_id"]),
            pdb_release_dates=_get_pdb_release_dates(pdb_mmcif_chains_df),
            pdb_obsolete_filepath=pdb_obsolete_filepath,
            shuffle_top_k_prefiltered=None,
            realign_when_required=realign_when_required,
            verbose=False,
        )

        super().__init__(
            mmcif_chains_df=mmcif_chains_df,
            alignments_super_index=alignments_super_index,
            pdb_mmcif_dicts_dirpath=pdb_mmcif_dicts_dirpath,
            pdb_alignments_dirpath=pdb_alignments_dirpath,
            template_hit_featurizer=template_hit_featurizer,
            alphafold_config=alphafold_config,
            mode="eval",
            verbose=verbose,
            name=name,
        )

    @property
    def pdb_chain_ids(self) -> List[str]:
        return [mmcif_chain["pdb_chain_id"] for mmcif_chain in self.mmcif_chains]


class SelfDistillationDataset(Dataset):
    """Self-distillation dataset containing predicted Uniclust30 structures."""

    def __init__(
        self,
        uniclust30_alignments_dirpath: Path,
        uniclust30_targets_dirpath: Path,
        pdb_mmcif_chains_filepath: Path,
        pdb_mmcif_dicts_dirpath: Path,
        pdb_obsolete_filepath: Path,
        max_pdb_release_date: str,
        alphafold_config: Union[DictConfig, Dict],
        realign_when_required: bool = True,
        verbose: bool = False,
        name: str = "SelfDistillationDataset",
    ) -> None:
        if verbose:
            logging.debug(f"{name}: initialization...")

        alignments_super_index = load_alignments_super_index(
            alignments_super_index_filepath=(uniclust30_alignments_dirpath / "super.index"),
            verbose=verbose,
            pprefix=f"{name}: ",
        )

        targets_super_index = self._load_targets_super_index(
            targets_super_index_filepath=(uniclust30_targets_dirpath / "super.index"),
            verbose=verbose,
            pprefix=f"{name}: ",
        )

        sample_keys = list(targets_super_index.keys())
        sample_keys = _filter_self_distillation_samples(
            sample_keys=sample_keys,
            targets_super_index=targets_super_index,
        )
        if verbose:
            logging.debug(f"{name}: len(sample_keys)={len(sample_keys)}")

        pdb_mmcif_chains_df = load_mmcif_chains_df(
            mmcif_chains_filepath=pdb_mmcif_chains_filepath,
            verbose=verbose,
            pprefix=f"{name}: ",
        )

        template_hit_featurizer = TemplateHitFeaturizer(
            max_template_hits=alphafold_config.max_templates,
            pdb_mmcif_dicts_dirpath=pdb_mmcif_dicts_dirpath,
            template_pdb_chain_ids=set(pdb_mmcif_chains_df["pdb_chain_id"]),
            pdb_release_dates=_get_pdb_release_dates(pdb_mmcif_chains_df),
            pdb_obsolete_filepath=pdb_obsolete_filepath,
            shuffle_top_k_prefiltered=alphafold_config.shuffle_top_k_prefiltered,
            realign_when_required=realign_when_required,
            verbose=False,
        )

        self.sample_keys = sample_keys
        self.targets_super_index = targets_super_index
        self.alignments_super_index = alignments_super_index
        self.uniclust30_targets_dirpath = uniclust30_targets_dirpath
        self.uniclust30_alignments_dirpath = uniclust30_alignments_dirpath
        self.template_hit_featurizer = template_hit_featurizer
        self.max_pdb_release_date = max_pdb_release_date
        self.alphafold_config = alphafold_config
        self.verbose = verbose
        self.name = name
        if verbose:
            logging.debug(f"{name}: initialized successfully!")

    def __getitem__(self, index_seed_pair: Tuple[int, int]) -> dict:
        if not isinstance(index_seed_pair, tuple):
            raise TypeError(
                f"__getitem__ expects {tuple} in format (index, seed),"
                f" but provided {type(index_seed_pair)} argument"
                f" contains value {repr(index_seed_pair)}"
            )
        index, seed = index_seed_pair
        assert isinstance(index, int)
        assert isinstance(seed, int)

        # Get sample key:
        sample_key = self.sample_keys[index]

        # Load sample data (target):
        target_dict = self._load_target_dict(target_key=sample_key)
        sequence = target_dict["sequence"]
        seqlen = len(sequence)

        # Load sample data (alignments):
        alignments = load_alignments(
            alignments_super_index=self.alignments_super_index,
            alignments_dirpath=self.uniclust30_alignments_dirpath,
            key=sample_key,
        )

        # Create target features:
        target_features = create_target_features(
            target_dict=target_dict,
            plddt_threshold=self.alphafold_config.self_distillation_plddt_threshold,
        )

        # Create template features:
        template_features = create_template_features(
            sequence=sequence,
            hhr_string=alignments.get("pdb70_hits.hhr", ""),
            template_hit_featurizer=self.template_hit_featurizer,
            release_date=self.max_pdb_release_date,
            pdb_id=None,
            days_before_release=60,
            shuffling_seed=seed,
        )

        # Create MSA features:
        msa_features = create_msa_features(
            sequence=sequence,
            a3m_strings=[
                alignments.get("uniclust30.a3m", ""),
            ],
        )

        # Process features:
        raw_features = {**target_features, **template_features, **msa_features}
        feats = process_features(
            raw_features=raw_features,
            alphafold_config=self.alphafold_config,
            mode="train",
            seed=seed,
        )

        # Add id tuple:
        feats["id"] = (self.name, index, seed, sample_key, seqlen)

        return feats

    def __len__(self) -> int:
        return len(self.sample_keys)

    def get_sampler_weights(self) -> torch.Tensor:
        return torch.ones(len(self), dtype=torch.float64)

    def _load_targets_super_index(
        self,
        targets_super_index_filepath: Path,
        verbose: bool = False,
        pprefix: str = "",
    ) -> Dict[str, dict]:
        if verbose:
            logging.debug(f"{pprefix}Loading {targets_super_index_filepath}...")
        with open(targets_super_index_filepath) as f:
            targets_super_index = json.load(f)
        if verbose:
            logging.debug(
                f"{pprefix}targets_super_index ({len(targets_super_index)})"
                f" loaded from {targets_super_index_filepath} successfully!"
            )
        return targets_super_index

    def _load_target_dict(self, target_key: str) -> dict:
        pdb_bytes = load_uniclust30_target(self.targets_super_index, target_key, self.uniclust30_targets_dirpath)
        pdb_string = pdb_bytes.decode("utf-8")
        protein = Protein.from_pdb_string(pdb_str=pdb_string)
        target_dict = {
            "id": target_key,
            "sequence": protein.sequence,
            "aatype": protein.aatype,
            "atom_positions": protein.atom_positions,
            "atom_mask": protein.atom_mask,
            "plddt": protein.b_factors,
        }
        return target_dict


class FinetuningDataset(Dataset):
    """Dataset for the fine-tuning stage."""

    def __init__(
        self,
        initial_training_dataset: InitialTrainingDataset,
        self_distillation_dataset: SelfDistillationDataset,
    ) -> None:
        self.initial_training_dataset = initial_training_dataset
        self.self_distillation_dataset = self_distillation_dataset

    def __getitem__(self, dataset_index_seed_triple: Tuple[str, int, int]) -> dict:
        dataset = dataset_index_seed_triple[0]
        index_seed_pair = dataset_index_seed_triple[1:]
        if dataset == "initial_training":
            return self.initial_training_dataset[index_seed_pair]
        elif dataset == "self_distillation":
            return self.self_distillation_dataset[index_seed_pair]
        else:
            raise ValueError(
                f"unknown dataset={repr(dataset)} value"
                f" in dataset_index_seed_triple={repr(dataset_index_seed_triple)}"
            )

    def __len__(self) -> int:
        return len(self.initial_training_dataset) + len(self.self_distillation_dataset)

    def get_sampler_weights(self) -> Dict[str, torch.Tensor]:
        return {
            "initial_training": self.initial_training_dataset.get_sampler_weights(),
            "self_distillation": self.self_distillation_dataset.get_sampler_weights(),
        }


class PredictDataset(Dataset):
    def __init__(
        self,
        pdb_mmcif_chains_filepath: Path,
        pdb_mmcif_dicts_dirpath: Path,
        pdb_obsolete_filepath: Path,
        sequences: Union[str, List[str]],
        seq_names: List[str],
        template_hhr_filepaths: Optional[List[List[Path]]],
        msa_a3m_filepaths: Optional[List[List[Path]]],
        generate_templates_if_missing: bool,
        pdb70_database_path: Optional[str],
        cfg: Union[DictConfig, Dict],
    ):
        """Dataset class for AlphaFold inference.

        Args:
            pdb_mmcif_chains_filepath (Path): Path to pdb_mmcif chain database.
            pdb_mmcif_dicts_dirpath (Path): Path to pdb_mmcif dict database.
            pdb_obsolete_filepath (Path): Path to obsolete pdb database.
            sequences (Union[str, List[str]]): Path to fasta or list of sequences in string.
            seq_names (List[str]): List of sequence names
            template_hhr_filepaths (Optional[List[List[Path]]]): Lists of paths to template hhr. One list per sequence.
            msa_a3m_filepaths (Optional[List[List[Path]]]): Lists of paths to msa in a3m format. One list per sequence.
            generate_templates_if_missing (bool): Generate template if none is given. Default to false.
            pdb70_database_path: (Optional[str]): Path to pdb70 database.
            cfg (Union[DictConfig, Dict]): AlphaFold model config.
        """
        self.cfg = cfg

        if isinstance(sequences, str):
            # parse_fasta returns a  tuple (sequences, descriptions)
            with open(sequences, "r") as fp:
                self.sequences = parse_fasta(fp.read())[0]
        else:
            self.sequences = sequences

        self.seq_names = seq_names
        if self.seq_names is None or len(seq_names) != len(sequences):
            raise ValueError(
                "Each sequence must have a name if sequence names are provided."
                "Please check length of sequence list and how many names were provided."
            )
        self.msa_a3m_filepaths = msa_a3m_filepaths
        self.template_hhr_filepaths = template_hhr_filepaths
        self.generate_templates_if_missing = generate_templates_if_missing
        if self.generate_templates_if_missing:
            if pdb70_database_path is None:
                raise ValueError("User must provide pdb70_database_path if generate_templates_if_missing.")
            self.hhsearch_pdb70_runner = HHSearch(
                databases=[pdb70_database_path],
            )

        # locate templates only if needed
        if self.template_hhr_filepaths or self.generate_templates_if_missing:
            pdb_mmcif_chains_df = load_mmcif_chains_df(
                mmcif_chains_filepath=pdb_mmcif_chains_filepath,
                verbose=False,
            )

            self.template_hit_featurizer = TemplateHitFeaturizer(
                max_template_hits=self.cfg.max_templates,
                pdb_mmcif_dicts_dirpath=pdb_mmcif_dicts_dirpath,
                template_pdb_chain_ids=set(pdb_mmcif_chains_df["pdb_chain_id"]),
                pdb_release_dates={},
                pdb_obsolete_filepath=pdb_obsolete_filepath,
                shuffle_top_k_prefiltered=None,
                realign_when_required=False,
                verbose=False,
            )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        ex_msa_a3m_filepaths = (
            self.msa_a3m_filepaths[idx] if self.msa_a3m_filepaths else None
        )  # FIXME: better resolution
        ex_template_hhr_filepath = (
            self.template_hhr_filepaths[idx] if self.template_hhr_filepaths else None
        )  # FIXME: better resolution

        # Read MSAs and concatenate (this is inteded). Save UniRef90 separately for PDB70 template search
        a3m_strings = []
        uniref90_msa_as_a3m = None
        if ex_msa_a3m_filepaths:
            for msa_a3m_filepath in ex_msa_a3m_filepaths:
                with open(msa_a3m_filepath, "r") as f:
                    a3m_string = f.read()
                    if "uniref" in msa_a3m_filepath:
                        uniref90_msa_as_a3m = a3m_string
                    a3m_strings.append(a3m_string)

        if ex_template_hhr_filepath:
            with open(ex_template_hhr_filepath, "r") as f:
                hhr_string = f.read()
        elif self.generate_templates_if_missing and uniref90_msa_as_a3m:
            hhr_string = self.hhsearch_pdb70_runner.query(uniref90_msa_as_a3m)
        else:
            hhr_string = ""

        sequence_features = create_sequence_features(
            sequence=sequence,
            domain_name="description",  # TODO: is description required here?
        )

        if hhr_string:  # search template only when generate_if_missing or template hhr is given
            template_features = create_template_features(
                sequence=sequence,
                hhr_string=hhr_string,
                template_hit_featurizer=self.template_hit_featurizer,
                release_date="2999-12-31",
                pdb_id=None,
                days_before_release=0,
                shuffling_seed=None,
            )
        else:
            logging.warning(f"No template hhr is given/found for sequence {idx}. Pass empty template features.")
            seq_len = len(sequence_features["residue_index"])
            template_features = create_empty_template_feats(seq_len)

        msa_features = create_msa_features(
            sequence=sequence,
            a3m_strings=a3m_strings,
        )

        raw_features = {**sequence_features, **template_features, **msa_features}

        sample = process_features(
            raw_features=raw_features,
            alphafold_config=self.cfg,
            mode="predict",
            seed=self.cfg.seed,
        )
        sample["seq_index"] = idx
        if self.seq_names:
            sample["seq_name"] = self.seq_names[idx]
        # return index so it might be saved with predictiona
        return sample


def _filter_pdb_chains_for_training(
    mmcif_chains_df: pd.DataFrame,
    min_release_date: str = "1900-01-01",
    max_release_date: str = "2999-12-31",
    max_resolution: float = 9.0,
    max_aa_frequency: float = 0.8,
) -> pd.DataFrame:
    # Supplementary '1.2.5 Filtering'
    is_release_date_between = mmcif_chains_df["release_date"].between(
        left=min_release_date,
        right=max_release_date,
        inclusive="both",
    )
    is_resolution_below_max = mmcif_chains_df["resolution"] < max_resolution
    is_resolution_nan = mmcif_chains_df["resolution"].isnull()
    is_resolution_acceptable = is_resolution_below_max | is_resolution_nan
    is_aa_frequency_acceptable = pd.Series(
        [_is_aa_frequency_acceptable(sequence, max_aa_frequency) for sequence in mmcif_chains_df["sequence"].tolist()]
    )
    is_in_pdb_clusters = ~mmcif_chains_df["pdb_cluster_id"].eq(-1)
    selector = is_release_date_between & is_resolution_acceptable & is_aa_frequency_acceptable & is_in_pdb_clusters
    mmcif_chains_df = mmcif_chains_df[selector].copy()
    return mmcif_chains_df


def _is_aa_frequency_acceptable(sequence: str, max_aa_frequency: float) -> bool:
    if len(sequence) == 0:
        return False
    cnt = Counter(sequence)
    top = cnt.most_common(1)[0]
    top_aa_count = top[1]
    top_aa_freq = top_aa_count / len(sequence)
    return top_aa_freq <= max_aa_frequency


def _get_weight(sample: dict) -> float:
    # Supplementary '1.2.5 Filtering'
    sequence_length = sample["sequence_length"]
    pdb_cluster_size = sample["pdb_cluster_size"]

    length_probability = max(min(sequence_length, 512), 256) / 512

    cluster_probability = 1.0
    if pdb_cluster_size > 0:
        cluster_probability = 1 / math.sqrt(pdb_cluster_size)

    weight = length_probability * cluster_probability
    return weight


def _select_cameo_targets_for_validation(
    mmcif_chains_df: pd.DataFrame,
    min_submission_date: str,
    max_submission_date: str,
    max_sequence_length: int = 700,
) -> pd.DataFrame:
    # Supplementary '1.11.7 Evaluator setup'

    # The following data comes from: https://cameo3d.org.
    # Example link: https://cameo3d.org/modeling/targets/1-year/?to_date=2022-07-02
    # CAMEO target id format is mostly "pdb_id [mmcif_chain_id]"
    # and occasionally "pdb_id [author_chain_id]".

    with resources.open_text("bionemo.data.protein.openfold.resources", "cameo_targets.json") as file:
        CAMEO_TARGETS = json.load(file)

    selected_cameo_targets = {}
    pdb_chain_ids_mapping = _get_pdb_chain_ids_mapping(mmcif_chains_df)
    cameo_submission_dates = list(CAMEO_TARGETS.keys())
    for cameo_submission_date in cameo_submission_dates:
        if not (min_submission_date <= cameo_submission_date <= max_submission_date):
            # ignore dates out of query range
            continue
        for cameo_target_id in CAMEO_TARGETS[cameo_submission_date]:
            cameo_pdb_mmcif_chain_id = _parse_cameo_target(cameo_target_id)
            if cameo_pdb_mmcif_chain_id not in pdb_chain_ids_mapping:
                # ignore CAMEO targets not found in mmcif chains (PDB)
                continue
            cameo_pdb_author_chain_id = pdb_chain_ids_mapping[cameo_pdb_mmcif_chain_id]
            if cameo_pdb_author_chain_id in selected_cameo_targets:
                # ignore CAMEO targets that map to already selected pdb author chain id
                continue
            selected_cameo_targets[cameo_pdb_author_chain_id] = cameo_target_id
    is_cameo_target = mmcif_chains_df["pdb_chain_id"].isin(selected_cameo_targets)
    mmcif_chains_df = mmcif_chains_df[is_cameo_target].copy()
    # Filter by max sequence length:
    is_sequence_length_acceptable = mmcif_chains_df["sequence_length"] <= max_sequence_length
    mmcif_chains_df = mmcif_chains_df[is_sequence_length_acceptable].copy()
    # Add cameo target column:
    selected_cameo_targets_df = pd.DataFrame(
        data={
            "pdb_chain_id": list(selected_cameo_targets.keys()),
            "cameo_target_id": list(selected_cameo_targets.values()),
        }
    )
    mmcif_chains_df = mmcif_chains_df.merge(
        right=selected_cameo_targets_df,
        how="left",
        on="pdb_chain_id",
    )
    return mmcif_chains_df


def _get_pdb_chain_ids_mapping(mmcif_chains_df: pd.DataFrame) -> Dict[str, str]:
    """Get mapping from `{pdb_id}_{mmcif_chain_id}` to `{pdb_id}_{author_chain_id}`."""
    pdb_chain_ids_mapping = {}
    pdb_author_chain_ids = set()
    for pdb_id, author_chain_id, mmcif_chain_ids in zip(
        mmcif_chains_df["pdb_id"].values,
        mmcif_chains_df["author_chain_id"].values,
        mmcif_chains_df["mmcif_chain_ids"].values,
    ):
        mmcif_chain_ids = mmcif_chain_ids.split(";")
        pdb_author_chain_id = f"{pdb_id}_{author_chain_id}"
        for mmcif_chain_id in mmcif_chain_ids:
            pdb_mmcif_chain_id = f"{pdb_id}_{mmcif_chain_id}"
            assert pdb_mmcif_chain_id not in pdb_chain_ids_mapping
            pdb_chain_ids_mapping[pdb_mmcif_chain_id] = pdb_author_chain_id
        pdb_author_chain_ids.add(pdb_author_chain_id)
    # Sometimes CAMEO targets are in format: "pdb_id [author_chain_id]"
    # The following loop will allow them to be included in the validation set.
    for pdb_author_chain_id in list(pdb_author_chain_ids):
        if pdb_author_chain_id not in pdb_chain_ids_mapping:
            pdb_chain_ids_mapping[pdb_author_chain_id] = pdb_author_chain_id
    return pdb_chain_ids_mapping


def _parse_cameo_target(cameo_target_id: str) -> str:
    pdb_id, mmcif_chain_id = cameo_target_id.split()
    pdb_id = pdb_id.lower()
    mmcif_chain_id = mmcif_chain_id.strip("[]")
    pdb_mmcif_chain_id = f"{pdb_id}_{mmcif_chain_id}"
    return pdb_mmcif_chain_id


def _filter_by_alignments(
    mmcif_chains_df: pd.DataFrame,
    alignments_super_index: Dict[str, dict],
) -> pd.DataFrame:
    selector = mmcif_chains_df["pdb_chain_id"].isin(alignments_super_index)
    mmcif_chains_df = mmcif_chains_df[selector].copy()
    return mmcif_chains_df


def _assert_pdb_alignments(
    mmcif_chains_df: pd.DataFrame,
    alignments_super_index: Dict[str, dict],
) -> None:
    pdb_chain_ids = set(mmcif_chains_df["pdb_chain_id"])
    alignments_super_index_ids = set(alignments_super_index.keys())
    if not pdb_chain_ids.issubset(alignments_super_index_ids):
        diff = pdb_chain_ids - alignments_super_index_ids
        raise RuntimeError(
            f"`mmcif_chains_df` has {len(diff)} ids"
            " not present in `alignments_super_index`."
            " To filter them out, set `filter_by_alignments` flag."
        )


def _get_pdb_release_dates(
    pdb_mmcif_chains_df: pd.DataFrame,
) -> Dict[str, datetime.datetime]:
    return {
        pdb_id: datetime_from_string(release_date, "%Y-%m-%d")
        for pdb_id, release_date in zip(
            pdb_mmcif_chains_df["pdb_id"].values,
            pdb_mmcif_chains_df["release_date"].values,
        )
    }


def _compute_pdb_cluster_size(mmcif_chains_df: pd.DataFrame) -> pd.DataFrame:
    vc = mmcif_chains_df["pdb_cluster_id"].value_counts()
    if -1 in vc:
        # -1 means 'unassigned to any cluster'
        vc[-1] = 0
    vc = vc.rename("pdb_cluster_size")
    vc = vc.reset_index()
    vc = vc.rename(columns={"index": "pdb_cluster_id"})
    mmcif_chains_df = mmcif_chains_df.merge(vc, on="pdb_cluster_id", how="left")
    return mmcif_chains_df


def _filter_self_distillation_samples(
    sample_keys: List[str],
    targets_super_index: Dict[str, dict],
) -> List[str]:
    sample_keys_filtered = []
    for sample_key in sample_keys:
        if len(targets_super_index[sample_key]["files"]) > 0:
            sample_keys_filtered.append(sample_key)
    return sample_keys_filtered
