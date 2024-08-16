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


import functools

import transformers
from nemo.lightning.io import IOMixin


class BioNeMoAutoTokenizer(transformers.AutoTokenizer, IOMixin):  # noqa D101
    def __init__(self, pretrained_model_name, use_fast=True):
        """A wrapper to make AutoTokenizer serializable.

        Args:
            pretrained_model_name: A string, the *model id* of a predefined tokenizer hosted on huggingface
            use_fast: Use a [fast Rust-based tokenizer](https://huggingface.co/docs/tokenizers/index)
            if it is supported for a given model. Defaults to True.
        """
        other = self.from_pretrained(pretrained_model_name, use_fast=use_fast)
        for attr in dir(other):
            if not attr.startswith("_"):
                setattr(self, attr, getattr(other, attr))


@functools.cache
def get_tokenizer() -> BioNeMoAutoTokenizer:
    """Get the tokenizer for the ESM2 model."""
    return BioNeMoAutoTokenizer(pretrained_model_name="facebook/esm2_t33_650M_UR50D", use_fast=True)
