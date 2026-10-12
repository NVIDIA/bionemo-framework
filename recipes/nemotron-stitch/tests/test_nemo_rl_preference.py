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

"""DPO carries encoder features in the stock preference batch coordinates."""

from __future__ import annotations

import pytest
import torch


def test_preference_features_follow_both_branches_and_pair_order(monkeypatch):
    pytest.importorskip("nemo_rl")
    from nemo_rl.data import processors
    from nemo_rl.data.collate_fn import preference_collate_fn

    from nemotron_stitch.nemo_rl import data

    encoder_preference_processor = data.encoder_preference_processor

    def preprocess(row, _spec, _tokenizer, _max_length, idx):
        def branch(answer):
            return [
                {"role": "user", "content": "prompt", "token_ids": torch.tensor([1, 32, 32])},
                {"role": "assistant", "content": answer, "token_ids": torch.tensor([answer])},
            ]

        return {
            "message_log_chosen": branch(2),
            "message_log_rejected": branch(3),
            "length_chosen": 4,
            "length_rejected": 4,
            "loss_multiplier": 1.0,
            "idx": idx,
        }

    monkeypatch.setattr(processors, "preference_preprocessor", preprocess)
    monkeypatch.setattr(data, "resolve_callback", lambda _name: lambda row, _adapter: row["features"])

    class Tokenizer:
        pad_token_id = 0

        def convert_tokens_to_ids(self, _token):
            return 32

    tokenizer = Tokenizer()
    rows = [
        {
            "adapter": "encoder",
            "encoder_loader": "test.loader",
            "encoder_placeholder_token": "<placeholder>",
            "features": torch.full((2, 3), value),
        }
        for value in (5.0, 7.0)
    ]
    processed = [encoder_preference_processor(row, None, tokenizer, 16, idx) for idx, row in enumerate(rows)]
    batch = preference_collate_fn(processed, tokenizer, 8, True)
    features = batch["mm_features__encoder"].as_tensor()
    assert batch["pair_index"].tolist() == [0, 0, 1, 1]
    assert batch["is_chosen"].tolist() == [True, False, True, False]
    assert features[:, 0].tolist() == [5.0, 5.0, 5.0, 5.0, 7.0, 7.0, 7.0, 7.0]
    assert batch["input_ids"].eq(32).sum(dim=1).tolist() == [2, 2, 2, 2]

    rows[0]["features"] = torch.ones(3, 3)
    with pytest.raises(ValueError, match="soft-token slots"):
        encoder_preference_processor(rows[0], None, tokenizer, 16, 0)
