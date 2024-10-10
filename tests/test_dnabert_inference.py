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


import logging
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Generator, List

import pytest
import torch

from bionemo.model.dna.dnabert.infer import DNABERTInference
from bionemo.tokenizer.dna_tokenizer import KmerTokenizer
from bionemo.utils.hydra import load_model_config
from bionemo.utils.tests import (
    Deterministic,
    distributed_model_parallel_state,
)

from .inference_shared_test_code import (
    get_config_dir,
    get_expected_vals_file,
    run_seqs_to_embedding,
    run_seqs_to_hiddens_with_goldens,  # Loads golden values or updates them (.pt file)
)


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


@pytest.fixture()
def dna() -> List[str]:
    import random

    random.seed(1337)
    dnas_for_test = ["".join([random.choice(["A", "C", "G", "T"]) for _ in range(512)]) for _ in range(5)]
    return deepcopy(dnas_for_test)


def gv_path_and_pbss_download(
    base_dir: Path,
    pbss_bucket: str,
    pbss_key_prefix: str,
    dnabert_expected_vals_path: Path,
) -> Path:
    # Remote filename
    gv_filename = "20240708-inference_test_golden_values.pt"

    # Local filenmae
    gv_file = dnabert_expected_vals_path

    if not gv_file.exists():
        print("GV file not found locally, downloading from PBSS")
        gv_file.parent.mkdir(parents=True, exist_ok=True)
        ret_code = subprocess.check_call(
            [
                "aws",
                "s3",
                "cp",
                f"s3://{pbss_bucket}/{pbss_key_prefix}/{gv_filename}",
                str(dnabert_expected_vals_path.absolute()),  # local filename
                "--endpoint-url",
                "https://pbss.s8k.io",
            ]
        )
        if ret_code != 0:
            raise ValueError("PBSS download failed! Check logs for details.")

    assert gv_file.is_file()

    return gv_file


@pytest.fixture(scope="module")
def dnabert_expected_vals(bionemo_home: Path) -> Path:
    dnabert_expected_vals_path = get_expected_vals_file(bionemo_home, "dnabert")
    return gv_path_and_pbss_download(
        base_dir=bionemo_home / "tests" / "data" / "dnabert",
        pbss_bucket="bionemo-ci",
        pbss_key_prefix="test-data/dnabert",
        dnabert_expected_vals_path=dnabert_expected_vals_path,
    )


@pytest.fixture(scope="module")
def dnabert_inferer(bionemo_home: Path) -> Generator[DNABERTInference, None, None]:
    model_name = "dnabert"
    cfg_path = get_config_dir(bionemo_home, model_name)
    cfg = load_model_config(config_name="infer", config_path=cfg_path)
    with Deterministic(), distributed_model_parallel_state():
        inferer = DNABERTInference(cfg=cfg, inference_batch_size_for_warmup=1)
        yield inferer  # Yield so cleanup happens after the test


def test_inferer_unk_tokens_from_bad_token(dnabert_inferer: DNABERTInference):
    # DNABERT returns UNK tokens if it does not exist in the vocabulary
    [unk_id1] = dnabert_inferer.tokenize(["FFFFF"])
    [unk_id2] = dnabert_inferer.tokenize(["ZZZZZ"])
    assert unk_id1 == unk_id2, "expect random tokens to return all unknowns"
    assert (
        unk_id1 == [dnabert_inferer.tokenizer.vocab[dnabert_inferer.tokenizer.unk_token]] * 3
    ), "Expected unknown token for invalid input sequence."

    # ... warmup
    actual1 = dnabert_inferer.seq_to_hiddens(["AAAAAATATATATAAAAAA"])
    actual2 = dnabert_inferer.seq_to_hiddens(["AAAAAATATATATAAAAAA"])
    assert all(
        torch.equal(x, y) for x, y in zip(actual1, actual2)
    ), "expected equal length lists of unknown tokens to have the same output."


def test_dnabert_inference_tokenization(dnabert_inferer: DNABERTInference, dna: List[str]):
    dna_tokenization = dnabert_inferer.tokenize(
        dna
    )  # we dont actually use the padding token in DNABERT, so padding is not returned by tokenize.
    kmer_tok: KmerTokenizer = dnabert_inferer.tokenizer
    # Should have the same shape as the input

    # Note expected length
    # expected = len(sequence) - kmer_tok.k + 1
    assert len(dna) == len(dna_tokenization)
    for i, seq in enumerate(dna):
        assert len(seq) - kmer_tok.k + 1 == len(
            dna_tokenization[i]
        ), f"Expected {len(seq) - kmer_tok.k + 1} tokens, got {len(dna_tokenization[i])} for sequence {seq}"


@pytest.mark.needs_fork
@pytest.mark.needs_gpu
def test_dna_to_hiddens_with_goldens_dnabert(
    dnabert_inferer: DNABERTInference, dna: List[List[str]], dnabert_expected_vals: Path
):
    # Do our warmup with the same batch size as what we will do for our test.
    dnabert_inferer.seq_to_hiddens(dna)
    # NOTE: if expected values need to be updated, they must also be updated on pbss.
    run_seqs_to_hiddens_with_goldens(
        dnabert_inferer,
        dna,
        dnabert_expected_vals,
        dnabert_inferer.model.cfg.hidden_size,
        "bert",
        dnabert_inferer._tokenize,
    )


@pytest.mark.needs_fork
@pytest.mark.needs_gpu
def test_dna_to_embedding_dnabert(dnabert_inferer: DNABERTInference, dna: List[str]):
    run_seqs_to_embedding(dnabert_inferer, dna, dnabert_inferer.model.cfg.hidden_size)
