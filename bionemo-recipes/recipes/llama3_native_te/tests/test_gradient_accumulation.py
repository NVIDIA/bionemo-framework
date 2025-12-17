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
    """Minimal model for testing gradient accumulation."""

    def __init__(self, vocab_size=32000, hidden_size=256, seq_len=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layer1 = nn.Linear(hidden_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, labels=None):
        x = self.embed_tokens(input_ids)
        x = self.layer1(x)
        x = functional.relu(x)
        logits = self.layer2(x)

        loss = None
        if labels is not None:
            logits_flat = logits.view(-1, self.vocab_size)
            labels_flat = labels.view(-1)
            loss = functional.cross_entropy(logits_flat, labels_flat)

        return type("CausalLMOutput", (), {"loss": loss, "logits": logits})()


def test_grad_acc_gradient_equivalence_golden_values():
    """Test that gradient accumulation produces mathematically equivalent gradients.

    Validates that mb=2, ga=1 produces identical gradients to mb=1, ga=2.
    """
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True

    batch_size_full = 2
    seq_len = 128
    vocab_size = 32000
    hidden_size = 256
    device = torch.device("cpu")

    input_ids = torch.randint(0, vocab_size, (batch_size_full, seq_len), dtype=torch.long)
    labels = torch.randint(0, vocab_size, (batch_size_full, seq_len), dtype=torch.long)

    # Configuration A: Full batch (mb=2, ga=1)
    model_a = SimpleTwoLayerModel(vocab_size=vocab_size, hidden_size=hidden_size, seq_len=seq_len).to(device)
    model_a.train()

    batch_full = {"input_ids": input_ids.to(device), "labels": labels.to(device)}
    outputs_a = model_a(**batch_full)
    loss_a = outputs_a.loss
    loss_a.backward()

    loss_value_a = loss_a.detach().item()
    grad_norm_a = torch.norm(torch.stack([p.grad.norm() for p in model_a.parameters()]))
    grads_a = {name: param.grad.clone() for name, param in model_a.named_parameters()}

    # Configuration B: Accumulated microbatches (mb=1, ga=2)
    model_b = SimpleTwoLayerModel(vocab_size=vocab_size, hidden_size=hidden_size, seq_len=seq_len).to(device)
    model_b.train()
    model_b.load_state_dict(model_a.state_dict())

    batch_micro1 = {"input_ids": input_ids[:1].to(device), "labels": labels[:1].to(device)}
    batch_micro2 = {"input_ids": input_ids[1:].to(device), "labels": labels[1:].to(device)}

    outputs_b1 = model_b(**batch_micro1)
    loss_b1 = outputs_b1.loss / 2
    loss_b1.backward()

    outputs_b2 = model_b(**batch_micro2)
    loss_b2 = outputs_b2.loss / 2
    loss_b2.backward()

    loss_value_b_total = loss_b1.detach().item() + loss_b2.detach().item()
    grad_norm_b = torch.norm(torch.stack([p.grad.norm() for p in model_b.parameters()]))
    grads_b = {name: param.grad.clone() for name, param in model_b.named_parameters()}

    # Verify loss values match
    torch.testing.assert_close(
        loss_value_a,
        loss_value_b_total,
        rtol=1e-2,
        atol=1e-6,
        msg=f"Loss mismatch: {loss_value_a:.6f} vs {loss_value_b_total:.6f}",
    )

    # Verify gradient norms match
    torch.testing.assert_close(
        grad_norm_a,
        grad_norm_b,
        rtol=1e-2,
        atol=1e-6,
        msg=f"Gradient norm mismatch: {grad_norm_a:.6f} vs {grad_norm_b:.6f}",
    )

    # Verify individual parameter gradients match
    for name in grads_a.keys():
        torch.testing.assert_close(
            grads_a[name],
            grads_b[name],
            rtol=1e-3,
            atol=1e-8,
            msg=f"Gradient mismatch for {name}",
        )
