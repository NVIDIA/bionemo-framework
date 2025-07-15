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

"""
To run (only in eos right now):
    pytest -svv sub-packages/bionemo-evo2/tests/bionemo/evo2/test_evo2_mamba_batch_generate.py

Todo: 
1. add dfw checkpoint
2. refactor this into test_evo2.py; parameterize the test_batch there for model type.
"""

import pytest
from pathlib import Path

import nemo.lightning as nl
import torch
from megatron.core.inference.common_inference_params import CommonInferenceParams
from nemo.collections.llm import generate


@pytest.fixture
def sequences():
    with (Path(__file__).parent / "data" / "prompts.csv").open(newline="") as f:
        from csv import DictReader

        reader = DictReader(f)
        return [row["Sequence"] for row in reader]


def calculate_sequence_identity(seq1: str, seq2: str) -> float | None:
    """Calculate sequence identity between two sequences through direct comparison."""
    if not seq1 or not seq2:
        return None

    min_length = min(len(seq1), len(seq2))
    matches = sum(a == b for a, b in zip(seq1[:min_length], seq2[:min_length]))

    return (matches / min_length) * 100


def mid_point_split(*, seq, num_tokens):
    mid_point = 2 * (len(seq) // 4)
    prompt = seq[:mid_point]
    target = seq[mid_point : mid_point + num_tokens]
    return prompt, target

def test_evo2_mamba(sequences: list[str]):
    trainer = nl.Trainer(
        accelerator="gpu",
        devices=1,
        strategy=nl.MegatronStrategy(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            pipeline_dtype=torch.bfloat16,
            ckpt_load_optimizer=False,
            ckpt_save_optimizer=False,
            ckpt_async_save=False,
            save_ckpt_format="torch_dist",
            ckpt_load_strictness="log_all",
        ),
        log_every_n_steps=1,
        limit_val_batches=10,
        num_sanity_val_steps=0,
        plugins=nl.MegatronMixedPrecision(
            precision="bf16-mixed",
            params_dtype=torch.bfloat16,
        ),
    )

    match_percents = []
    expected_matchpercents = [96.8, 29.7, 76.6, 71.6]
    num_tokens = 500

    seq_prompts = [mid_point_split(seq=seq, num_tokens=num_tokens) for seq in sequences]
    # Tyler's checkpoint in eos
    ckpt_dir = (
        "checkpoints/pretrain_hybrid_mamba_8b--val_loss=1.0444-epoch=0-consumed_samples=73993728.0-last/"
    )

    results = generate(
        path=ckpt_dir,
        prompts=[sq[0] for sq in seq_prompts],
        trainer=trainer,
        inference_params=CommonInferenceParams(
            temperature=1.0,
            top_k=1,
            top_p=0.0,
            return_log_probs=False,
            num_tokens_to_generate=num_tokens,
        ),
        text_only=True,
        random_seed=42,
    )

    for i, (result, (prompt, target)) in enumerate(zip(results, seq_prompts)):
        gen_seq = result
        match_percent = calculate_sequence_identity(target, gen_seq)
        match_percents.append(match_percent)
    print("Target:", target)
    print("Generated:", gen_seq)

    assert len(match_percents) == len(expected_matchpercents)
    matchperc_print = [f"{mp:.1f}%" for mp in match_percents]
    matchperc_print_expected = [f"{ep:.1f}%" for ep in expected_matchpercents]
    assert all(
        mp >= 0.90 * ep for mp, ep in zip(match_percents, expected_matchpercents)
    ), (
        f"Expected at least 90% of {matchperc_print_expected=}, got {matchperc_print=}"
    )

