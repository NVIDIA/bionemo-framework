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


from typing import Dict

import torch


def get_random_microbatch(
    microbatch_size: int, max_sequence_length: int, vocab_size: int, seed: int
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Generate random microbatches for testing."""
    generator = torch.Generator(device=torch.cuda.current_device()).manual_seed(seed)
    labels = torch.randint(
        low=0,
        high=vocab_size,
        size=(microbatch_size, max_sequence_length),
        generator=generator,
        device=torch.cuda.current_device(),
    )  # [b s]
    loss_mask = torch.randint(
        low=1,
        high=1 + 1,
        size=(microbatch_size, max_sequence_length),
        dtype=torch.long,
        device=torch.cuda.current_device(),
        generator=generator,
    )  # [b s]
    token_logits = torch.rand(
        microbatch_size, max_sequence_length, vocab_size, device=torch.cuda.current_device(), generator=generator
    )  # [b s v]
    labels[loss_mask == 0] = -100  # propagate masking to labels
    microbatch_output = {
        "batch": {"labels": labels, "loss_mask": loss_mask},
        "forward_out": {"token_logits": token_logits},
    }
    return microbatch_output
