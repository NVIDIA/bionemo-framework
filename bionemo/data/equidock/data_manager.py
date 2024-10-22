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

from bionemo.data.equidock.db5_data import nemo_get_dataset


class Singleton(type):
    """
    Python metaclass to insures single instance of a class.

    Example Usage:
        class MySingleton(metaclass=Singleton)
            pass
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class DataManager(metaclass=Singleton):
    def __init__(self, cfg, **kwargs) -> None:
        self.train_ds = nemo_get_dataset(cfg.model.train_ds, "train")
        self.validation_ds = nemo_get_dataset(cfg.model.validation_ds, "val")
        self.test_ds = nemo_get_dataset(cfg.model.test_ds, "test")

    def swap(self, cfg):
        """
        Override DataManager using a new cfg file
        """
        self.train_ds = nemo_get_dataset(cfg.model.train_ds, "train")
        self.validation_ds = nemo_get_dataset(cfg.model.validation_ds, "val")
        self.test_ds = nemo_get_dataset(cfg.model.test_ds, "test")
