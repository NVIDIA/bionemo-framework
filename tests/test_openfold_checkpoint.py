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


# This test suit checks if we can load original, public weights into BioNeMo.
# Particularly, if we turn on initial-training optimizations like weights fusing,
# is it still possible to load it. We also include sanity-checking for
# scenarios that should fail.
#
# OpenFold in BioNeMo has some layers renamed so we always have to map them when
# we load .pt checkpoints. This is reflected in
# bionemo.model.protein.openfold.checkpoint_utils.remap_layers_names

import os

import omegaconf
import pytest
import torch

from bionemo.model.protein.openfold.checkpoint_utils import load_pt_checkpoint
from bionemo.model.protein.openfold.openfold_model import AlphaFold
from bionemo.model.protein.openfold.optim_hub import OptimHub
from bionemo.utils.tests import check_model_exists


BIONEMO_HOME = os.getenv("BIONEMO_HOME")
CHECKPOINT_PATH = os.path.join(BIONEMO_HOME, "models/protein/openfold/initial_training.pt")


@pytest.fixture(scope="module")
def base_config(bionemo_home):
    # base config is enough. It contains architectural details for initial-training without
    # training hyperparameters
    return omegaconf.OmegaConf.load(os.path.join(bionemo_home, "examples/protein/openfold/conf/base_config.yaml"))


@pytest.fixture(scope="function")
def alphafold_model(base_config, request):
    if hasattr(request, "param"):
        OptimHub.enable_multiple(request.param)
    return AlphaFold(base_config.model, None)


@pytest.mark.xfail(reason="FIXME: Missing checkpoint")
@pytest.mark.needs_checkpoint
def test_model_exists():
    check_model_exists(CHECKPOINT_PATH)


@pytest.mark.needs_checkpoint
@pytest.mark.skip_if_no_file(CHECKPOINT_PATH)
def test_load_openfold_fail_direct(alphafold_model):
    """
    We should be NOT able to load pt checkpoint directly as some layers have different names.
    """
    with pytest.raises(RuntimeError):
        alphafold_model.load_state_dict(torch.load(CHECKPOINT_PATH))


@pytest.mark.needs_checkpoint
@pytest.mark.skip_if_no_file(CHECKPOINT_PATH)
@pytest.mark.parametrize(
    "alphafold_model",
    [[], ["mha_fused_gemm"], ["mha_fused_gemm", "mha_triton"], ["mha_fused_gemm", "mha_triton", "layernorm_inductor"]],
    indirect=["alphafold_model"],
)
def test_load_openfold_mapping(alphafold_model):
    """
    We should be able to load pt checkpoint if GEMM fusing from MLPerf is turned on.
    """
    load_pt_checkpoint(alphafold_model, CHECKPOINT_PATH)
