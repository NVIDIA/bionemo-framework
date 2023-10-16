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

import os
from copy import deepcopy
from typing import Dict, List, Optional, Union

from nemo.core import Dataset
from omegaconf import DictConfig, OmegaConf

from bionemo.data.dataset_builder_utils import build_typed_dataset
from bionemo.data.preprocess.molecule.uspto50k_preprocess import USPTO50KPreprocess


def megamolbart_build_train_valid_test_datasets(
    cfg: DictConfig, train_valid_test_num_samples: Dict[str, Optional[int]]
) -> List[Dataset]:
    """
    Build train, validation and test for pretraining of MegaMolBartModel.
    Args:
        cfg: config of data components
        train_valid_test_num_samples: dict that specifies split-specific size of loaded dataset
    Returns:
        list of dataset for splits
    """
    cfg = deepcopy(cfg)

    # setting
    use_upsampling: bool = cfg.get('use_upsampling', True)
    data_impl: str = cfg.get('data_impl', None)
    assert data_impl is not None, 'Config "cfg" should contain field "cfg.data_impl"'
    dataset_path: str = cfg.get('dataset_path', None)
    assert dataset_path is not None, 'Config "cfg" should contain field "cfg.dataset_path"'

    assert all(
        split in ['train', 'val', 'test'] for split in train_valid_test_num_samples.keys()
    ), 'Incorrect key in train_valid_test_num_samples!'

    datasets = []
    # Build individual datasets.
    for split in train_valid_test_num_samples.keys():
        num_samples = train_valid_test_num_samples[split]
        print(f'{split}:{num_samples}')
        if num_samples is None or num_samples > 0:
            ds_name: Optional[Union[str, List[Union[int, str]]]] = cfg.dataset.get(split, None)
            assert ds_name is not None, (
                f'Config "cfg" should contain field "cfg.dataset.{split}" with name or list of '
                f'names corresponding to the data files used to construct the dataset'
            )
            filepath: str = os.path.join(dataset_path, split, ds_name)
            dataset = build_typed_dataset(
                dataset_paths=filepath,
                data_impl=data_impl,
                use_upsampling=use_upsampling,
                cfg=cfg,
                num_samples=num_samples,
            )
        else:
            dataset = None
        datasets.append(dataset)

    return datasets


def megamolbart_retro_build_train_valid_test_datasets(
    cfg: OmegaConf, train_valid_test_num_samples: Dict[str, Optional[int]]
) -> List[Optional[Dataset]]:
    """
    Build train, validation and test reaction dataset for MegaMolBartRetro model.
    Args:
        cfg: config of data components
        train_valid_test_num_samples: dict that specifies split-specific size of loaded dataset
    Returns:
        list of dataset for splits
    """
    cfg = deepcopy(cfg)
    use_upsampling = cfg.get('use_upsampling', True)
    data_preprocessor = USPTO50KPreprocess(max_smiles_length=cfg.max_seq_length, data_dir=cfg.dataset_path)
    assert all(
        split in data_preprocessor.splits for split in train_valid_test_num_samples.keys()
    ), 'Incorrect key in train_valid_test_num_samples!'

    data_impl = cfg.get('data_impl', None)
    assert data_impl is not None, 'Argument data_impl must be specified!'

    datasets = []

    for split in train_valid_test_num_samples.keys():
        num_samples = train_valid_test_num_samples[split]
        print(f'{split}:{num_samples}')
        if num_samples is None or num_samples > 0:
            filepath = os.path.join(data_preprocessor.get_split_dir(split), data_preprocessor.data_file)
            dataset = build_typed_dataset(
                dataset_paths=filepath,
                data_impl=data_impl,
                use_upsampling=use_upsampling,
                cfg=cfg,
                num_samples=num_samples,
            )
        else:
            dataset = None
        datasets.append(dataset)

    return datasets
