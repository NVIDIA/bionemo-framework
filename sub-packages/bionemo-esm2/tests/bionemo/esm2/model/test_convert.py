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
import torch
from nemo.lightning import io
from transformers import AutoModelForMaskedLM

from bionemo.core.data.load import load
from bionemo.esm2.model.convert import HFESM2Exporter, HFESM2Importer  # noqa: F401
from bionemo.esm2.model.model import ESM2Config
from bionemo.esm2.testing.compare import assert_esm2_equivalence
from bionemo.llm.model.biobert.lightning import biobert_lightning_module
from bionemo.testing import megatron_parallel_state_utils


# pytestmark = pytest.mark.xfail(
#     reason="These tests are failing due to a bug in nemo global state when run in the same process as previous "
#     "checkpoint save/load scripts."
# )


def test_nemo2_conversion_equivalent_8m(tmp_path):
    model_tag = "facebook/esm2_t6_8M_UR50D"
    module = biobert_lightning_module(config=ESM2Config(), post_process=True)
    io.import_ckpt(module, f"hf://{model_tag}", tmp_path / "nemo_checkpoint")
    with megatron_parallel_state_utils.distributed_model_parallel_state():
        assert_esm2_equivalence(tmp_path / "nemo_checkpoint", model_tag)


def test_nemo2_export_8m_weights_equivalent(tmp_path):
    ckpt_path = load("esm2/8m:2.0")
    with megatron_parallel_state_utils.distributed_model_parallel_state():
        output_path = io.export_ckpt(ckpt_path, "hf", tmp_path / "hf_checkpoint")

    hf_model_from_nemo = AutoModelForMaskedLM.from_pretrained(output_path)
    hf_model_from_hf = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")

    del hf_model_from_nemo.esm.contact_head
    del hf_model_from_hf.esm.contact_head

    for key in hf_model_from_nemo.state_dict().keys():
        torch.testing.assert_close(
            hf_model_from_nemo.state_dict()[key],
            hf_model_from_hf.state_dict()[key],
            atol=1e-2,
            rtol=1e-2,
            msg=lambda msg: f"{key}: {msg}",
        )


def test_nemo2_export_golden_values(tmp_path):
    ckpt_path = load("esm2/8m:2.0")
    with megatron_parallel_state_utils.distributed_model_parallel_state():
        output_path = io.export_ckpt(ckpt_path, "hf", tmp_path / "hf_checkpoint")
        assert_esm2_equivalence(ckpt_path, output_path, precision="bf16")


def test_nemo2_conversion_equivalent_8m_bf16(tmp_path):
    model_tag = "facebook/esm2_t6_8M_UR50D"
    module = biobert_lightning_module(config=ESM2Config())
    io.import_ckpt(module, f"hf://{model_tag}", tmp_path / "nemo_checkpoint")
    with megatron_parallel_state_utils.distributed_model_parallel_state():
        assert_esm2_equivalence(tmp_path / "nemo_checkpoint", model_tag, precision="bf16")


@pytest.mark.slow
def test_nemo2_conversion_equivalent_650m(tmp_path):
    model_tag = "facebook/esm2_t33_650M_UR50D"
    module = biobert_lightning_module(config=ESM2Config())
    io.import_ckpt(module, f"hf://{model_tag}", tmp_path / "nemo_checkpoint")
    with megatron_parallel_state_utils.distributed_model_parallel_state():
        assert_esm2_equivalence(tmp_path / "nemo_checkpoint", model_tag, atol=1e-4, rtol=1e-4)
