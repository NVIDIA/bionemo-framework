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

import logging
import os
from pathlib import Path

import nemo.lightning as nl
import pytest
import torch
from megatron.core.inference.common_inference_params import CommonInferenceParams


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Capture all levels in the logger itself


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


def get_trainer(pipeline_parallel=1):
    fp8 = True
    full_fp8 = False
    return nl.Trainer(
        accelerator="gpu",
        devices=pipeline_parallel,
        strategy=nl.MegatronStrategy(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=pipeline_parallel,
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
            # Only use FP8 in this plugin when using full FP8 precision and FP8.
            #   Otherwise use vortex_style_fp8 in the model config.
            fp8="hybrid" if fp8 and full_fp8 else None,
            fp8_amax_history_len=16 if fp8 and full_fp8 else 1,
            fp8_amax_compute_algo="max" if fp8 and full_fp8 else "most_recent",
        ),
    )


def get_model_and_tokenizer(ckpt_dir_or_name: Path | str):
    trainer = get_trainer()
    from bionemo.core.data.load import load

    if isinstance(ckpt_dir_or_name, Path):
        ckpt_dir: Path = ckpt_dir_or_name
    else:
        ckpt_dir: Path = load(ckpt_dir_or_name)
    from nemo.collections.llm import inference

    inference_wrapped_model, mcore_tokenizer = inference.setup_model_and_tokenizer(
        path=ckpt_dir,
        trainer=trainer,
        params_dtype=torch.bfloat16,
        inference_batch_times_seqlen_threshold=8192,  # TODO
        inference_max_seq_length=8192,  # TODO
        recompute_granularity=None,
        recompute_num_layers=None,
        recompute_method=None,
    )
    return inference_wrapped_model, mcore_tokenizer


# TODO: add evo2_mamba/7b-8k:0.1 to NGC and remove this skipif.
@pytest.mark.skipif(
    os.environ.get("BIONEMO_DATA_SOURCE") != "pbss",
    reason="This test will fail because the checkpoint is not on NGC yet. Run with `BIONEMO_DATA_SOURCE=pbss`.",
)
def test_evo2_mamba(
    sequences: list[str],
    ckpt_name: str = "evo2_mamba/7b-8k:0.1",
    expected_matchpercents: list[float] = [99.2, 51.0, 73.0, 82.6],
):
    assert len(sequences) > 0
    try:
        inference_wrapped_model, mcore_tokenizer = get_model_and_tokenizer(ckpt_name)
    except ValueError as e:
        if "does not have an NGC URL." in str(e):
            raise AssertionError(
                "Please rerun test with `BIONEMO_DATA_SOURCE=pbss pytest ...` "
                + "as the requested checkpoint is not yet uploaded to NGC."
                + "Original error: "
                + str(e)
            )

    match_percents = []
    num_tokens = 500
    seq_prompts = [mid_point_split(seq=seq, num_tokens=num_tokens) for seq in sequences]
    from nemo.collections.llm.inference import generate

    results = generate(
        model=inference_wrapped_model,
        max_batch_size=1,  # vortex only supports batch size 1
        tokenizer=mcore_tokenizer,
        prompts=[sq[0] for sq in seq_prompts],
        random_seed=42,
        inference_params=CommonInferenceParams(
            temperature=1.0,
            top_k=1,
            top_p=0.0,
            return_log_probs=False,
            num_tokens_to_generate=num_tokens,
        ),
    )

    for i, (result, (prompt, target)) in enumerate(zip(results, seq_prompts)):
        gen_seq = result.generated_text
        logger.info(f"{ckpt_name} {torch.distributed.get_rank()=} {gen_seq=}")
        logger.info(f"{ckpt_name} {torch.distributed.get_rank()=} {target=}")
        match_percent = calculate_sequence_identity(target, gen_seq)
        logger.info(
            f"{ckpt_name} {torch.distributed.get_rank()=} {match_percent=} expected: {expected_matchpercents[i]}"
        )
        match_percents.append(match_percent)

    assert len(match_percents) == len(expected_matchpercents)
    matchperc_print = [f"{mp:.1f}%" for mp in match_percents]
    matchperc_print_expected = [f"{ep:.1f}%" for ep in expected_matchpercents]
    assert all(mp >= 0.90 * ep for mp, ep in zip(match_percents, expected_matchpercents)), (
        f"Expected at least 90% of {matchperc_print_expected=}, got {matchperc_print=}"
    )
