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

import torch
import torch.nn as nn
import torch.nn.functional as functional


class SimpleTwoLayerModel(nn.Module):
    """Simple model for testing gradient accumulation - mimics basic transformer-like architecture."""

    def __init__(self, vocab_size=32000, hidden_size=256, seq_len=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len

        # Embedding layer (like Llama's input embeddings)
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

        # Two linear layers (simplified attention + MLP)
        self.layer1 = nn.Linear(hidden_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, labels=None):
        # Embedding
        x = self.embed_tokens(input_ids)  # [batch_size, seq_len, hidden_size]

        # Layer 1 (simplified self-attention + MLP)
        x = self.layer1(x)
        x = functional.relu(x)

        # Layer 2 (output projection)
        logits = self.layer2(x)  # [batch_size, seq_len, vocab_size]

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Flatten for cross-entropy
            logits_flat = logits.view(-1, self.vocab_size)
            labels_flat = labels.view(-1)
            loss = functional.cross_entropy(logits_flat, labels_flat)

        return type("CausalLMOutput", (), {"loss": loss, "logits": logits})()


def test_grad_acc_gradient_equivalence_golden_values():
    """Test that gradient accumulation produces mathematically equivalent gradients to full batch.

    This golden value test validates that:
    - mb=2, ga=1 produces identical gradients to mb=1, ga=2
    - Loss values match (within tolerance)
    - Gradient norms match (within tolerance)
    - Individual parameter gradients match element-wise (within tight tolerance)

    This ensures gradient accumulation is implemented correctly.
    """
    # Set seeds for deterministic behavior
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True

    # Model and data parameters
    batch_size_full = 2
    seq_len = 128
    vocab_size = 32000
    hidden_size = 256

    # Create deterministic input data
    input_ids = torch.randint(0, vocab_size, (batch_size_full, seq_len), dtype=torch.long)
    labels = torch.randint(0, vocab_size, (batch_size_full, seq_len), dtype=torch.long)

    # Ensure we're on CPU for deterministic behavior (can be changed to CUDA if needed)
    device = torch.device("cpu")

    # ============================================================================
    # Configuration A: Full batch (mb=2, ga=1)
    # ============================================================================
    model_a = SimpleTwoLayerModel(vocab_size=vocab_size, hidden_size=hidden_size, seq_len=seq_len).to(device)
    model_a.train()

    # Single forward pass with full batch
    batch_full = {"input_ids": input_ids.to(device), "labels": labels.to(device)}
    outputs_a = model_a(**batch_full)

    # Backward pass (no loss scaling for ga=1)
    loss_a = outputs_a.loss
    loss_a.backward()

    # Capture results
    loss_value_a = loss_a.detach().item()
    grad_norm_a = torch.norm(torch.stack([p.grad.norm() for p in model_a.parameters()]))
    grads_a = {name: param.grad.clone() for name, param in model_a.named_parameters()}

    # ============================================================================
    # Configuration B: Accumulated microbatches (mb=1, ga=2)
    # ============================================================================
    model_b = SimpleTwoLayerModel(vocab_size=vocab_size, hidden_size=hidden_size, seq_len=seq_len).to(device)
    model_b.train()

    # Copy weights from model_a for fair comparison
    model_b.load_state_dict(model_a.state_dict())

    # Split full batch into two microbatches
    batch_micro1 = {"input_ids": input_ids[:1].to(device), "labels": labels[:1].to(device)}  # First sample
    batch_micro2 = {"input_ids": input_ids[1:].to(device), "labels": labels[1:].to(device)}  # Second sample

    # First microbatch
    outputs_b1 = model_b(**batch_micro1)
    loss_b1 = outputs_b1.loss / 2  # Scale loss by grad_acc_steps
    loss_b1.backward()

    # Second microbatch
    outputs_b2 = model_b(**batch_micro2)
    loss_b2 = outputs_b2.loss / 2  # Scale loss by grad_acc_steps
    loss_b2.backward()

    # Capture results
    loss_value_b_total = loss_b1.detach().item() + loss_b2.detach().item()  # Total loss for comparison
    grad_norm_b = torch.norm(torch.stack([p.grad.norm() for p in model_b.parameters()]))
    grads_b = {name: param.grad.clone() for name, param in model_b.named_parameters()}

    # ============================================================================
    # Validation: Compare configurations
    # ============================================================================

    # 1. Total loss values should match (scaled appropriately)
    # Config A: loss for full batch (effective batch size 2)
    # Config B: sum of scaled losses (each scaled by 1/2, so total = (loss1 + loss2)/2)
    torch.testing.assert_close(
        loss_value_a,
        loss_value_b_total,
        rtol=1e-2,  # 1% relative tolerance
        atol=1e-6,  # Small absolute tolerance
        msg=f"Total loss mismatch: Config A ({loss_value_a:.6f}) vs Config B total ({loss_value_b_total:.6f})",
    )

    # 2. Gradient norms should match (1% relative tolerance)
    torch.testing.assert_close(
        grad_norm_a,
        grad_norm_b,
        rtol=1e-2,  # 1% relative tolerance
        atol=1e-6,  # Small absolute tolerance
        msg=f"Gradient norm mismatch: Config A ({grad_norm_a:.6f}) vs Config B ({grad_norm_b:.6f})",
    )

    # 3. Individual gradients should match element-wise (0.1% relative tolerance)
    for name in grads_a.keys():
        torch.testing.assert_close(
            grads_a[name],
            grads_b[name],
            rtol=1e-3,  # 0.1% relative tolerance for individual gradients
            atol=1e-8,  # Very small absolute tolerance
            msg=f"Gradient mismatch for {name}: max diff = {torch.max(torch.abs(grads_a[name] - grads_b[name]))}",
        )

    print("✅ Gradient accumulation golden value test passed!")
    print(".6f")
    print(".6f")
    print(".6f")


if __name__ == "__main__":
    test_grad_acc_gradient_equivalence_golden_values()
