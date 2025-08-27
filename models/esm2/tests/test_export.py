# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


def test_export_hf_checkpoint(tmp_path):
    from esm.export import export_hf_checkpoint
    from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

    export_hf_checkpoint("esm2_t6_8M_UR50D", tmp_path)

    model_for_masked_lm, loading_info = AutoModelForMaskedLM.from_pretrained(
        tmp_path / "esm2_t6_8M_UR50D", trust_remote_code=True, output_loading_info=True
    )

    assert not loading_info["missing_keys"]
    assert not loading_info["unexpected_keys"]
    assert not loading_info["mismatched_keys"]
    assert not loading_info["error_msgs"]

    model, loading_info = AutoModel.from_pretrained(
        tmp_path / "esm2_t6_8M_UR50D", trust_remote_code=True, output_loading_info=True
    )

    assert not loading_info["mismatched_keys"]
    assert not loading_info["error_msgs"]

    tokenizer = AutoTokenizer.from_pretrained(tmp_path / "esm2_t6_8M_UR50D")

    assert model_for_masked_lm is not None
    assert model is not None
    assert tokenizer is not None

    # Test that required files (LICENSE, README.md) are present in the exported directory
    export_dir = tmp_path / "esm2_t6_8M_UR50D"
    assert (export_dir / "LICENSE").is_file(), "LICENSE file is missing in the export directory"
    assert (export_dir / "README.md").is_file(), "README.md file is missing in the export directory"
