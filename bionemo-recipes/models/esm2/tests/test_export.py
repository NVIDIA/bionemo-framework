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


from pathlib import Path
import pytest
import tempfile
import torch
from transformers import AutoModelForMaskedLM


def test_export_hf_checkpoint(tmp_path):
    from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

    from esm.export import export_hf_checkpoint

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
    readme_path = export_dir / "README.md"
    assert readme_path.is_file(), "README.md file is missing in the export directory"
    with open(readme_path, "r") as f:
        readme_contents = f.read()
    assert "**Number of model parameters:** 7.5 x 10^6" in readme_contents, (
        f"README.md does not contain the expected parameter count line: {readme_contents}"
    )
    assert (
        "Hugging Face 07/29/2025 via [https://huggingface.co/nvidia/esm2_t6_8M_UR50D]"
        "(https://huggingface.co/nvidia/esm2_t6_8M_UR50D)"
    ) in readme_contents, f"README.md does not contain the expected Hugging Face link line: {readme_contents}"
    assert "**Benchmark Score:** 0.48" in readme_contents, (
        f"README.md does not contain the expected CAMEO score line: {readme_contents}"
    )
    assert "**Benchmark Score:** 0.37" in readme_contents, (
        f"README.md does not contain the expected CASP14 score line: {readme_contents}"
    )

@pytest.mark.parametrize("model_name", ["esm2_t6_8M_UR50D"])
def test_export_te_checkpoint_to_hf(model_name):
    """Test the export function that saves TE checkpoint as HF format."""
    from esm.export import export_hf_checkpoint, export_te_checkpoint

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        model_hf_original = AutoModelForMaskedLM.from_pretrained(f"facebook/{model_name}")

        # Use export_hf_checkpoint to create TE checkpoint
        te_checkpoint_path = temp_path / "te_checkpoint"
        export_hf_checkpoint(model_name, te_checkpoint_path)
        te_model_path = te_checkpoint_path / model_name

        hf_export_path = temp_path / "hf_export"
        export_te_checkpoint(str(te_model_path), str(hf_export_path))

        model_hf_exported = AutoModelForMaskedLM.from_pretrained(str(hf_export_path))

        original_state_dict = model_hf_original.state_dict()
        exported_state_dict = model_hf_exported.state_dict()

        # assert original_state_dict.keys() == exported_state_dict.keys()
        original_keys = {k for k in original_state_dict.keys() if "contact_head" not in k}
        exported_keys = {k for k in exported_state_dict.keys() if "contact_head" not in k}
        assert original_keys == exported_keys

        for key in original_state_dict.keys():
            if not key.endswith("_extra_state") and not key.endswith("inv_freq") and "contact_head" not in key:
                torch.testing.assert_close(original_state_dict[key], exported_state_dict[key], atol=1e-5, rtol=1e-5)