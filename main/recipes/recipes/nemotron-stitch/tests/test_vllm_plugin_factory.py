# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""build_mm_plugin factory surface. Framework imports stay lazy."""

from pathlib import Path


def test_build_mm_plugin_is_framework_free_and_idempotent_shape():
    import os
    import subprocess
    import sys

    import nemotron_stitch

    # Fresh process: the suite has already imported frameworks elsewhere.
    src_root = Path(nemotron_stitch.__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": f"{src_root}:{os.environ.get('PYTHONPATH', '')}"}
    code = (
        "import sys; import nemotron_stitch.vllm.plugin; "
        "assert 'vllm' not in sys.modules and 'transformers' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


def test_encode_mode_validates_its_callback_set():
    import pytest

    from nemotron_stitch.vllm.plugin import build_mm_plugin

    # Encode mode requires a complete callback set; it is
    # validated per parameter. Class-building cases live in
    # tests/test_vllm_encode_plugin.py, which needs vLLM installed.
    with pytest.raises(ValueError, match="encode_model_cls"):
        build_mm_plugin(
            modality="generic",
            architecture="GenericModel",
            placeholder_text="<generic>",
            mode="encode",
        )


def test_projected_mode_validates_its_required_set():
    import pytest

    from nemotron_stitch.vllm.plugin import SentinelAttrs, build_mm_plugin

    with pytest.raises(ValueError, match="base_model_cls"):
        build_mm_plugin(
            modality="generic",
            architecture="GenericModel",
            placeholder_text="<generic>",
            sentinels=SentinelAttrs(start="a", placeholder="b", end="c"),
            num_tokens_attr="generic_num_tokens",
        )


def test_projected_mode_validates_geometry_at_factory_call():
    import pytest

    from nemotron_stitch.vllm.plugin import SentinelAttrs, build_mm_plugin

    with pytest.raises(ValueError, match="unknown projected geometry"):
        build_mm_plugin(
            modality="generic",
            architecture="GenericModel",
            placeholder_text="<generic>",
            sentinels=SentinelAttrs(start="a", placeholder="b", end="c"),
            num_tokens_attr="generic_num_tokens",
            geometry="ragged",
            # Present so the required-set proof passes first; the classes are
            # only imported inside register(), which never runs here.
            base_model_cls="some.module:Model",
            base_processing_info_cls="some.module:Info",
            base_processor_cls="some.module:Processor",
            base_dummy_inputs_cls="some.module:Dummy",
        )


def test_check_projected_token_count():
    import pytest

    from nemotron_stitch.vllm.plugin import check_projected_token_count

    # Fixed: exact match or a transport-bug error.
    check_projected_token_count("toy", 4, 4, geometry="fixed")
    with pytest.raises(ValueError, match="has 3 tokens, expected 4"):
        check_projected_token_count("toy", 3, 4, geometry="fixed")

    # Variable: the payload sizes the request up to the configured maximum.
    check_projected_token_count("toy", 1, 4, geometry="variable")
    check_projected_token_count("toy", 4, 4, geometry="variable")
    with pytest.raises(ValueError, match="outside the configured maximum 4"):
        check_projected_token_count("toy", 5, 4, geometry="variable")
    with pytest.raises(ValueError, match="outside the configured maximum 4"):
        check_projected_token_count("toy", 0, 4, geometry="variable")

    with pytest.raises(ValueError, match="unknown projected geometry"):
        check_projected_token_count("toy", 4, 4, geometry="ragged")
