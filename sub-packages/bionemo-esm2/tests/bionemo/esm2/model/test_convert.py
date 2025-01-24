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


import pytest
from nemo.lightning import io

from bionemo.esm2.model.convert import HFESM2Importer  # noqa: F401
from bionemo.esm2.model.model import ESM2Config
from bionemo.esm2.testing.compare import assert_model_equivalence
from bionemo.llm.model.biobert.lightning import biobert_lightning_module
from bionemo.testing import megatron_parallel_state_utils


# pytestmark = pytest.mark.xfail(
#     reason="These tests are failing due to a bug in nemo global state when run in the same process as previous "
#     "checkpoint save/load scripts."
# )


def test_nemo2_conversion_equivalent_8m(tmp_path):
    model_tag = "facebook/esm2_t6_8M_UR50D"
    module = biobert_lightning_module(config=ESM2Config())
    io.import_ckpt(module, f"hf://{model_tag}", tmp_path / "nemo_checkpoint")
    with megatron_parallel_state_utils.distributed_model_parallel_state():
        assert_model_equivalence(tmp_path / "nemo_checkpoint", model_tag)


def test_nemo2_conversion_equivalent_8m_bf16(tmp_path):
    model_tag = "facebook/esm2_t6_8M_UR50D"
    module = biobert_lightning_module(config=ESM2Config())
    io.import_ckpt(module, f"hf://{model_tag}", tmp_path / "nemo_checkpoint")
    with megatron_parallel_state_utils.distributed_model_parallel_state(precision="bf16"):
        assert_model_equivalence(tmp_path / "nemo_checkpoint", model_tag, precision="bf16")


@pytest.mark.slow
def test_nemo2_conversion_equivalent_650m(tmp_path):
    model_tag = "facebook/esm2_t33_650M_UR50D"
    module = biobert_lightning_module(config=ESM2Config())
    io.import_ckpt(module, f"hf://{model_tag}", tmp_path / "nemo_checkpoint")
    with megatron_parallel_state_utils.distributed_model_parallel_state():
        assert_model_equivalence(tmp_path / "nemo_checkpoint", model_tag, atol=1e-4, rtol=1e-4)
