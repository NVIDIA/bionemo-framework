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
import pytest

from bionemo.contrib.data.molecule.diffdock.datamodule import ScoreModelWDS


@pytest.fixture(scope="module")
def get_path(request):
    dir_test = os.path.dirname(request.module.__file__)
    dir_data = f"{dir_test}/test_data"
    return dir_test, dir_data


@pytest.fixture(scope="module")
def get_diffdock_score_model_heterodata(get_path, tmp_path_factory):
    _, dir_data = get_path
    dir_heterodata = f"{dir_data}/molecule/diffdock/heterodata"
    suffix_heterodata = "heterodata.pyd"
    prefix_dir_tars_wds = tmp_path_factory.mktemp(
        "diffdock_score_model_tars_wds").as_posix()
    names_subset_train = set(["6t88", "6vs3", "6wtn", "6yqv", "7amc", "7bmi",
                              "7cuo", "7d5c", "7din", "7fha", "7jnb", "7k0v",
                              "7kb1", "7km8", "7l7c", "7lcu", "7msr", "7my1",
                              "7n6f", "7np6"])
    names_subset_val = set(["7nr6", "7oeo", "7oli", "7oso", "7p5t", "7q5i",
                            "7qhl", "7rh3", "7rzl", "7sgv"])
    names_subset_test = set(["7sne", "7t2i", "7tbu", "7tsf", "7umv", "7up3",
                             "7uq3", "7wpw", "7xek", "7xij"])
    return (dir_heterodata, suffix_heterodata, prefix_dir_tars_wds,
            names_subset_train, names_subset_val, names_subset_test)


@pytest.fixture(scope="module")
def create_ScoreModelWDS(get_diffdock_score_model_heterodata):
    (dir_heterodata, suffix_heterodata, prefix_dir_tars_wds,
            names_subset_train, names_subset_val, names_subset_test) =\
        get_diffdock_score_model_heterodata
    local_batch_size = 2
    global_batch_size = 2
    n_workers_dataloader = 2
    n_tars_wds = 4
    seed_rng_shfl = 822782392
    data_module = ScoreModelWDS(dir_heterodata, suffix_heterodata,
                                prefix_dir_tars_wds, names_subset_train,
                                names_subset_val, local_batch_size,
                                global_batch_size, n_workers_dataloader,
                                n_tars_wds=n_tars_wds,
                                names_subset_test=names_subset_test,
                                seed_rng_shfl=seed_rng_shfl)
    return data_module

create_another_ScoreModelWDS = create_ScoreModelWDS
