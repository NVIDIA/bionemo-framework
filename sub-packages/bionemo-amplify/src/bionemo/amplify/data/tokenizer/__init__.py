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
from importlib.resources import files

from transformers import AutoTokenizer
from nemo.lightning.io import IOMixin


class BioNeMoAMPLIFYTokenizer(AutoTokenizer, IOMixin):  # noqa D101
    def __init__(self):
        """A wrapper to make AutoTokenizer serializable.
        """
        other = AutoTokenizer.from_pretrained(str(files("bionemo.amplify.data.tokenizer")), use_fast=True)
        for attr in dir(other):
            if not attr.startswith("_"):
                setattr(self, attr, getattr(other, attr))
        #In case PreTrainedTokenizer is inherited and special token IDs are not in dir
        if hasattr(other, "mask_token_id"):
            setattr(self, "mask_token_id", getattr(other, "mask_token_id"))
        if hasattr(other, "pad_token_id"):
            setattr(self, "pad_token_id", getattr(other, "pad_token_id"))


@functools.cache
def get_tokenizer() -> BioNeMoAMPLIFYTokenizer:
    """Get the tokenizer for the AMPLIFY model."""
    return BioNeMoAMPLIFYTokenizer()
