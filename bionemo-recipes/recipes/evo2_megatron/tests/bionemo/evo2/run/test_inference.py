# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Arc Institute. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Michael Poli. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Stanford University. All rights reserved
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

"""Integration tests for Evo2 inference using MBridge.

These tests verify that the Evo2 model can generate text autoregressively
using the MCore inference infrastructure with HyenaInferenceContext.
"""

import torch

from bionemo.evo2.models.evo2_provider import HyenaInferenceContext


class TestHyenaInferenceContext:
    """Unit tests for the Hyena-specific inference context."""

    def test_context_initialization(self):
        """Test that HyenaInferenceContext can be initialized."""
        context = HyenaInferenceContext(max_batch_size=1, max_sequence_length=8192)
        assert context is not None
        assert context.max_batch_size == 1
        assert context.max_sequence_length == 8192

    def test_context_reset(self):
        """Test that context reset works without error."""
        context = HyenaInferenceContext(max_batch_size=1, max_sequence_length=8192)
        # Add some fake filter state (simulating what hyena layers do)
        context.filter_state_dict_layer_0 = {"key": torch.zeros(10)}
        context.filter_state_dict_layer_1 = {"key": torch.ones(10)}

        # Verify the state was added
        assert hasattr(context, "filter_state_dict_layer_0")
        assert hasattr(context, "filter_state_dict_layer_1")

        # Reset should remove all filter_state_dict attributes
        context.reset()

        assert not hasattr(context, "filter_state_dict_layer_0")
        assert not hasattr(context, "filter_state_dict_layer_1")

    def test_context_materialize_logits_setting(self):
        """Test that materialize_only_last_token_logits can be configured."""
        context = HyenaInferenceContext(max_batch_size=1, max_sequence_length=8192)

        # Default should be True for efficiency
        # We can set it to False if we need full sequence logits
        context.materialize_only_last_token_logits = False
        assert context.materialize_only_last_token_logits is False

        context.materialize_only_last_token_logits = True
        assert context.materialize_only_last_token_logits is True

    def test_context_multiple_batches(self):
        """Test context with different batch sizes."""
        for batch_size in [1, 2, 4]:
            context = HyenaInferenceContext(max_batch_size=batch_size, max_sequence_length=4096)
            assert context.max_batch_size == batch_size
            context.reset()  # Should not error

    def test_context_different_sequence_lengths(self):
        """Test context with different max sequence lengths."""
        for seq_len in [1024, 8192, 16384]:
            context = HyenaInferenceContext(max_batch_size=1, max_sequence_length=seq_len)
            assert context.max_sequence_length == seq_len
            context.reset()
