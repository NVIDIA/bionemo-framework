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

import gzip
import io
import json

import pytest
import requests
from nemo.collections.llm.gpt.model.hyena import HyenaModel
from torch import Tensor
from torch.nn import functional as F

from bionemo.evo2.models.mamba import MambaModel
from bionemo.evo2.run.predict import BasePredictor, _to_cpu


def head_jsonl_gz(url: str, n: int, *, timeout: float = 30.0):
    """
    Stream a .jsonl.gz from `url` and return the first `n` JSON objects.
    Only downloads enough compressed bytes to produce N lines, then closes.
    """
    if n <= 0:
        return []

    # Ask server to send raw bytes; we'll handle gzip ourselves.
    headers = {"Accept-Encoding": "identity", "User-Agent": "pytest-jsonl-head/1.0"}
    with requests.get(url, stream=True, headers=headers, timeout=timeout) as r:
        r.raise_for_status()
        # Feed the raw compressed stream into gzip
        gz = gzip.GzipFile(fileobj=r.raw)  # bytes -> decompressor
        text = io.TextIOWrapper(gz, encoding="utf-8", newline="")  # bytes -> str lines

        out = []
        for i, line in enumerate(text, 1):
            out.append(json.loads(line))
            if i >= n:
                break
        # Close early so the server stops sending more bytes
        r.close()
        return out


@pytest.mark.network  # optional marker for CI
def test_take_first_5_records():
    url = "https://huggingface.co/datasets/arcinstitute/opengenome2/resolve/main/json/pretraining_or_both_phases/organelle/data_organelle_test_chunk1.jsonl.gz"
    first5 = head_jsonl_gz(url, 5)
    assert len(first5) == 5
    # Example structure checks you can tailor:
    assert isinstance(first5[0], dict)


class LogitsPredictor(BasePredictor):
    def __init__(
        self, *args, output_log_prob_seqs: bool = True, include_tokens_with_logprob_seqs: bool = True, **kwargs
    ):
        super().__init__(
            *args,
            output_log_prob_seqs=output_log_prob_seqs,
            include_tokens_with_logprob_seqs=include_tokens_with_logprob_seqs,
            **kwargs,
        )

    def predict_step(self, batch, batch_idx: int | None = None) -> Tensor | dict[str, Tensor] | None:
        result: dict[str, Tensor] = super().predict_step(batch, batch_idx, to_cpu=False)
        shifted_token_logits = result["token_logits"][:, :-1]
        shifted_pad_mask = result["pad_mask"][:, 1:]
        shifted_tokens = result["tokens"][:, 1:]
        lm_loss_full = (
            F.cross_entropy(shifted_token_logits[shifted_pad_mask], shifted_tokens[shifted_pad_mask], reduction="none")
            * shifted_pad_mask
        )
        n_tokens = shifted_pad_mask.sum()
        nll_sum = lm_loss_full.sum()
        return _to_cpu({"nll_sum": nll_sum, "n_tokens": n_tokens})


class MambaLogitsPredictor(LogitsPredictor, MambaModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class HyenaLogitsPredictor(LogitsPredictor, HyenaModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
