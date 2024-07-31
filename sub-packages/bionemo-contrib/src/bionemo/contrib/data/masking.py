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


from dataclasses import dataclass

import torch

from bionemo.contrib.data.esm2 import tokenizer


@dataclass(frozen=True)
class BertMaskConfig:
    """Configuration for masking tokens in a BERT-style model.

    Attributes:
        mask_prob: Probability of masking a token.
        mask_token_prob: Probability of replacing a masked token with the mask token.
        random_token_prob: Probability of replacing a masked token with a random token.
    """

    mask_prob: float = 0.15
    mask_token_prob: float = 0.8
    random_token_prob: float = 0.1


def apply_bert_pretraining_mask(
    tokenized_sequence: torch.Tensor,
    random_seed: int,
    mask_config: BertMaskConfig = BertMaskConfig(),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Applies the pretraining mask to a tokenized sequence.

    Args:
        tokenized_sequence: Tokenized protein sequence.
        random_seed: Random seed for reproducibility.

    Returns:
        masked_sequence:
            The tokenized sequence with some tokens masked.
        labels:
            A tensor the same shape as `masked_sequence` containing labels for the masked tokens, with -1 for non-masked
            tokens.
        loss_mask:
            A boolean tensor the same shape as `masked_sequence`, where 'True' indicates which tokens should be included
            in the loss.
    """

    if mask_config.random_token_prob + mask_config.mask_token_prob > 1.0:
        raise ValueError("Sum of random_token_prob and mask_token_prob must be less than or equal to 1.0.")

    # Set the seed so that __getitem__(idx) is always deterministic.
    # This is required by Megatron-LM's parallel strategies.
    torch.manual_seed(random_seed)

    mask_stop_1 = mask_config.mask_prob * mask_config.mask_token_prob
    mask_stop_2 = mask_config.mask_prob * (mask_config.mask_token_prob + mask_config.random_token_prob)

    random_draws = torch.rand(tokenized_sequence.shape)
    loss_mask = (
        random_draws < mask_config.mask_prob
    )  # The token is masked in some capacity; propagated to loss function.
    mask_token_mask = random_draws < mask_stop_1  # The token is masked and replaced with the mask token.
    random_token_mask = (random_draws >= mask_stop_1) & (random_draws < mask_stop_2)  # Replaced with a random token.

    # Mask the tokens. For the random mask, we sample from the range [4, 24) to avoid non-standard amino acid tokens.
    masked_sequence = tokenized_sequence.clone()
    masked_sequence[mask_token_mask] = tokenizer.SpecialToken.MASK
    masked_sequence[random_token_mask] = torch.randint(low=4, high=24, size=(random_token_mask.sum().item(),))

    # Create the labels for the masked tokens.
    labels = tokenized_sequence.clone()
    labels[~loss_mask] = -1  # Ignore loss for non-masked tokens.

    return masked_sequence, labels, loss_mask
