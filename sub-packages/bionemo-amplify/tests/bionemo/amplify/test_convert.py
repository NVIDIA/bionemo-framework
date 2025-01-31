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


from nemo.lightning import io

from bionemo.amplify.convert import HFAMPLIFYImporter  # noqa: F401
from bionemo.amplify.model import AMPLIFYConfig
from bionemo.amplify.tokenizer import BioNeMoAMPLIFYTokenizer
from bionemo.core.utils.dtypes import PrecisionTypes
from bionemo.esm2.testing.compare import get_input_tensors, load_and_evaluate_hf_model
from bionemo.llm.model.biobert.lightning import biobert_lightning_module


def assert_amplify_equivalence(ckpt_path: str, model_tag: str, precision: PrecisionTypes) -> None:
    tokenizer = BioNeMoAMPLIFYTokenizer()

    input_ids, attention_mask = get_input_tensors(tokenizer)
    load_and_evaluate_hf_model(model_tag, precision, input_ids, attention_mask)


def test_convert_smoke_test(tmp_path):
    model_tag = "chandar-lab/AMPLIFY_120M"
    module = biobert_lightning_module(config=AMPLIFYConfig())
    io.import_ckpt(module, f"hf://{model_tag}", tmp_path / "nemo_checkpoint")
