# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging
from copy import deepcopy
from pathlib import Path
from typing import Generator, List

import pytest

from bionemo.model.molecule.molmim.infer import MolMIMInference
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
def molmim_inferer(bionemo_home: Path) -> Generator[MolMIMInference, None, None]:
    model_name = "molmim"
    cfg_path = get_config_dir(bionemo_home, model_name)
    cfg = load_model_config(config_name="infer", config_path=cfg_path)
    with distributed_model_parallel_state():
        inferer = get_inference_class(model_name)(cfg=cfg, inference_batch_size_for_warmup=2)
        yield inferer  # Yield so cleanup happens after the test


@pytest.fixture(scope="module")
def molmim_expected_vals_path(bionemo_home: Path) -> Path:
    return get_expected_vals_file(bionemo_home, "molmim")


@pytest.mark.needs_fork
@pytest.mark.needs_gpu
def test_smis_to_hiddens_with_goldens_molmim(
    molmim_inferer: MolMIMInference, _smis: List[str], molmim_expected_vals_path: Path
):
    run_seqs_to_hiddens_with_goldens(
        molmim_inferer,
        _smis,
        molmim_expected_vals_path,
        molmim_inferer.model.cfg.encoder.hidden_size,
        molmim_inferer.model.cfg.encoder.arch,
        tokenize_fn=molmim_inferer._tokenize,
    )


@pytest.mark.needs_fork
@pytest.mark.needs_gpu
def test_smis_to_embedding_molmim(molmim_inferer: MolMIMInference, _smis: List[str]):
    run_seqs_to_embedding(molmim_inferer, _smis, molmim_inferer.model.cfg.encoder.hidden_size)


@pytest.mark.needs_fork
@pytest.mark.needs_gpu
def test_hidden_to_smis_molmim(molmim_inferer: MolMIMInference, _smis: List[str]):
    run_hidden_to_smis(molmim_inferer, _smis)


@pytest.mark.needs_fork
@pytest.mark.needs_gpu
@pytest.mark.parametrize("sampling_method", ["topkp-perturbate", "greedy-perturbate"])
def test_sample_molmim(molmim_inferer: MolMIMInference, _smis: List[str], sampling_method: str):
    run_sample_not_beam(molmim_inferer, _smis, sampling_method)


@pytest.mark.needs_fork
@pytest.mark.needs_gpu
@pytest.mark.parametrize("beam_search_method", ["beam-search-perturbate", "beam-search-single-sample"])
def test_beam_search_molmim(molmim_inferer: MolMIMInference, _smis: List[str], beam_search_method: str):
    run_beam_search(molmim_inferer, _smis, beam_search_method)


@pytest.mark.needs_fork
@pytest.mark.needs_gpu
def test_beam_search_product_molmim(molmim_inferer: MolMIMInference, _smis: List[str]):
    run_beam_search_product(molmim_inferer, _smis)


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
def test_interpolate_molmim(molmim_inferer: MolMIMInference, _smis: List[str], sampling_method: str):
    run_interpolate(molmim_inferer, _smis, sampling_method)
