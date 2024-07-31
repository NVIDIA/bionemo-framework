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


import enum

import torch


class SpecialToken(enum.IntEnum):
    CLS = 0
    PAD = 1
    EOS = 2
    UNK = 3
    MASK = 32


# The original ESM tokenizer implementation supports different multi-character tokens like <mask> and <cls>, but this
# makes the actual tokenization logic more complex. We simplify the implementation by re-mapping these to
# single-character tokens prior to performing the mask.
_character_map = {aa: i + 4 for i, aa in enumerate("LAGVSERTIDPKQNFYMHWCXBUZO.-")}
_character_map["^"] = SpecialToken.CLS
_character_map["_"] = SpecialToken.PAD
_character_map["$"] = SpecialToken.EOS
_character_map["?"] = SpecialToken.UNK
_character_map["#"] = SpecialToken.MASK


def tokenize(sequence: str) -> torch.Tensor:
    """Tokenize a protein sequence into integers.

    Note that this function does not prepend the special token for the beginning of a sequence or end of the sequence,
    nor does it fill the output with padding tokens.

    Args:
        sequence: A protein sequence.

    Returns:
        A tensor of integers representing the protein sequence.
    """

    # Replace these special tokens with single-character tokens.
    sequence = (
        sequence.replace(" ", "")
        .replace("<cls>", "^")
        .replace("<pad>", "_")
        .replace("<eos>", "$")
        .replace("<unk>", "?")  # Not sure where this is used -- 'X' is typically used for unknown amino acids.
        .replace("<mask>", "#")
    )

    return torch.tensor([_character_map[aa] for aa in sequence], dtype=torch.int64)
