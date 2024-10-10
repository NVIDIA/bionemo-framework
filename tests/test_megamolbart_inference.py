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


import logging
from copy import deepcopy
from pathlib import Path
from typing import Generator, List

import pytest

from bionemo.model.molecule.megamolbart import MegaMolBARTInference
from bionemo.utils.hydra import load_model_config
from bionemo.utils.tests import (
    distributed_model_parallel_state,
)

from .inference_shared_test_code import (
    get_config_dir,
    get_expected_vals_file,
    get_inference_class,
    run_seqs_to_embedding,
    run_seqs_to_hiddens_with_goldens,
)
from .molecule_inference_shared_test_code import (
    SMIS_FOR_TEST,
    run_beam_search,
    run_beam_search_product,
    run_hidden_to_smis,
    run_interpolate,
    run_sample_not_beam,
)


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


@pytest.fixture()
def _smis() -> List[str]:
    return deepcopy(SMIS_FOR_TEST)


@pytest.fixture(scope="module")
def megamolbart_inferer(bionemo_home: Path) -> Generator[MegaMolBARTInference, None, None]:
    model_name = "megamolbart"
    cfg_path = get_config_dir(bionemo_home, model_name)
    cfg = load_model_config(config_name="infer", config_path=cfg_path)
    with distributed_model_parallel_state():
        inferer = get_inference_class(model_name)(
            cfg=cfg, inference_batch_size_for_warmup=2
        )  # Change to 1 to debug the failure
        yield inferer  # Yield so cleanup happens after the test


@pytest.fixture(scope="module")
def megamolbart_expected_vals_path(bionemo_home: Path) -> Path:
    return get_expected_vals_file(bionemo_home, "megamolbart")


@pytest.mark.needs_fork
@pytest.mark.needs_gpu
def test_smis_to_hiddens_with_goldens_megamolbart(
    megamolbart_inferer: MegaMolBARTInference, _smis: List[str], megamolbart_expected_vals_path: Path
):
    run_seqs_to_hiddens_with_goldens(
        megamolbart_inferer,
        _smis,
        megamolbart_expected_vals_path,
        megamolbart_inferer.model.cfg.encoder.hidden_size,
        megamolbart_inferer.model.cfg.encoder.arch,
        megamolbart_inferer._tokenize,
    )


@pytest.mark.needs_fork
@pytest.mark.needs_gpu
def test_smis_to_embedding_megamolbart(megamolbart_inferer: MegaMolBARTInference, _smis: List[str]):
    run_seqs_to_embedding(megamolbart_inferer, _smis, megamolbart_inferer.model.cfg.encoder.hidden_size)


@pytest.mark.needs_fork
@pytest.mark.needs_gpu
def test_hidden_to_smis_megamolbart(megamolbart_inferer: MegaMolBARTInference, _smis: List[str]):
    run_hidden_to_smis(megamolbart_inferer, _smis)


@pytest.mark.needs_fork
@pytest.mark.needs_gpu
@pytest.mark.parametrize("sampling_method", ["topkp-perturbate", "greedy-perturbate"])
def test_sample_megamolbart(megamolbart_inferer: MegaMolBARTInference, _smis: List[str], sampling_method: str):
    run_sample_not_beam(megamolbart_inferer, _smis, sampling_method)


@pytest.mark.needs_fork
@pytest.mark.needs_gpu
@pytest.mark.parametrize("beam_search_method", ["beam-search-perturbate", "beam-search-single-sample"])
def test_beam_search_megamolbart(megamolbart_inferer: MegaMolBARTInference, _smis: List[str], beam_search_method: str):
    run_beam_search(megamolbart_inferer, _smis, beam_search_method)


@pytest.mark.needs_fork
@pytest.mark.needs_gpu
def test_beam_search_product_megamolbart(megamolbart_inferer: MegaMolBARTInference, _smis: List[str]):
    run_beam_search_product(megamolbart_inferer, _smis)


@pytest.mark.needs_fork
@pytest.mark.needs_gpu
@pytest.mark.parametrize(
    "sampling_method",
    [
        "greedy-perturbate",
        "topkp-perturbate",
        "beam-search-perturbate",
        "beam-search-perturbate-sample",
        "beam-search-single-sample",
    ],
)
def test_interpolate_megamolbart(megamolbart_inferer: MegaMolBARTInference, _smis: List[str], sampling_method: str):
    run_interpolate(megamolbart_inferer, _smis, sampling_method)
