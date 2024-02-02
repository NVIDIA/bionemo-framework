# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from functools import partial

from nemo.utils import logging

from bionemo.data.diffdock.confidence_dataset import ListDataset, diffdock_confidence_dataset
from bionemo.data.diffdock.docking_dataset import diffdock_build_dataset
from bionemo.model.molecule.diffdock.infer import DiffDockModelInference
from bionemo.model.molecule.diffdock.utils.diffusion import t_to_sigma as t_to_sigma_compl


class Singleton(type):
    """
    Ensures single instance of a class.

    Example Usage:
        class MySingleton(metaclass=Singleton)
            pass
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

    def reset_instances(cls):
        cls._instances = {}


class DataManager(metaclass=Singleton):
    def __init__(self, cfg) -> None:
        self.cfg = cfg

        if cfg.model.confidence_mode:
            # Initialize the datasets for confidence model training
            self.train_ds = diffdock_confidence_dataset(cfg.model.train_ds, mode="train")
            self.validation_ds = diffdock_confidence_dataset(cfg.model.validation_ds, mode="validation")
            self.test_ds = diffdock_confidence_dataset(cfg.model.test_ds, mode="test")
        else:
            # Initialize the datasets for score model training
            self.t_to_sigma = partial(t_to_sigma_compl, cfg=cfg.model)

            self.train_ds = diffdock_build_dataset(
                cfg.model.train_ds, self.t_to_sigma, _num_conformers=True, mode="train"
            )
            self.validation_ds = diffdock_build_dataset(
                cfg.model.validation_ds, self.t_to_sigma, _num_conformers=False, mode="validation"
            )
            self.test_ds = diffdock_build_dataset(
                cfg.model.test_ds, self.t_to_sigma, _num_conformers=False, mode="test"
            )

        self.datasets_ready = False
        if not cfg.do_preprocessing:
            try:
                self.load_datasets()
            except Exception:
                logging.warning(
                    "Data preprocessing is not done, set 'do_preprocessing' to True for preprocessing datasets"
                )

    def preprocess(self):
        if self.cfg.model.confidence_mode:
            # preprocess datasets for confidence model training
            # 1. preprocess complex graphs
            self.train_ds.build_complex_graphs()
            self.validation_ds.build_complex_graphs()
            self.test_ds.build_complex_graphs()

            # 2. Use a trained score model to generate ligand poses
            score_model = DiffDockModelInference(self.cfg.score_infer)
            score_model.eval()

            self.train_ds.build_confidence_dataset(score_model)
            self.validation_ds.build_confidence_dataset(score_model)
            self.test_ds.build_confidence_dataset(score_model)
        else:
            # preprocess complex graph for score model training
            self.train_ds.build_complex_graphs()
            self.validation_ds.build_complex_graphs()
            self.test_ds.build_complex_graphs()

        self.load_datasets()

    def load_datasets(self):
        if self.cfg.model.confidence_mode:
            # load confidence dataset(complex graphs and ligand poses) for confidence model training
            self.train_ds.load_confidence_dataset()
            self.validation_ds.load_confidence_dataset()
            self.test_ds.load_confidence_dataset()

            self.train_ds = ListDataset(self.train_ds)
            self.validation_ds = ListDataset(self.validation_ds)
            self.test_ds = ListDataset(self.test_ds)

        else:
            # load complex graph dataset for score model training
            self.train_ds.load_complex_graphs()
            self.validation_ds.load_complex_graphs()
            self.test_ds.load_complex_graphs()
        self.datasets_ready = True
