# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pytest
from omegaconf import DictConfig
from pytriton.client import ModelClient
from pytriton.triton import Triton

from bionemo.model.protein.esm1nv.infer import ESM1nvInference
from bionemo.triton.client_encode import send_seqs_for_inference
from bionemo.triton.embeddings import triton_embedding_infer_fn
from bionemo.triton.hiddens import triton_hidden_infer_fn
from bionemo.triton.inference_wrapper import complete_model_name
from bionemo.triton.serve_bionemo_model import bind_embedding, bind_hidden
from bionemo.triton.types_constants import (
    EMBEDDINGS,
    HIDDENS,
    MASK,
    SEQUENCES,
)
from bionemo.triton.utils import (
    encode_str_batch,
    load_model_config,
    load_model_for_inference,
)


SEQS = ['MSLKRKNIALIPAAGIGVRFGADKPKQYVEIGSKTVLEHVL', 'MIQSQINRNIRLDLADAILLSKAKKDLSFAEIADGTGLA']

MODEL_NAME = "esm1nv"

NAME_EMBEDDINGS = complete_model_name(MODEL_NAME, EMBEDDINGS)
NAME_HIDDENS = complete_model_name(MODEL_NAME, HIDDENS)


@pytest.fixture(scope='module')
def cfg(bionemo_home: Path) -> DictConfig:
    return load_model_config(
        config_path=bionemo_home / "examples" / "protein" / MODEL_NAME / "conf",
        config_name="infer.yaml",
        logger=None,
    )


@pytest.fixture(scope='module')
def model(cfg: DictConfig) -> ESM1nvInference:
    # TODO [mgreaves] replace with this in !553
    # model = load_model_for_inference(cfg, interactive=False)
    return load_model_for_inference(cfg)


@pytest.fixture(scope='module')
def server(cfg: DictConfig, model: ESM1nvInference) -> Triton:
    triton = Triton()
    bind_embedding(triton, cfg, model, nav=False, triton_model_name=NAME_EMBEDDINGS)
    bind_hidden(triton, cfg, model, nav=False, triton_model_name=NAME_HIDDENS)
    triton.run()
    yield triton
    triton.stop()


def _validate_embeddings(result: Dict[str, np.ndarray], key: str, expected_shape: Tuple[int]) -> None:
    assert isinstance(result, dict), f"Expecting dict-like but found {type(result)=}"
    assert key in result, f"Expecting {key} but only found {result.keys()=}"
    embeddings = result[key]
    assert embeddings.shape == expected_shape


@pytest.mark.needs_checkpoint
@pytest.mark.needs_gpu
# TODO [mgreaves] parameterize later: need expected shapes too
# @pytest.mark.parametrize('seqs', [SEQS])
def test_seq_to_embedding_triton(server: Triton) -> None:
    with ModelClient("grpc://localhost:8001", NAME_EMBEDDINGS, inference_timeout_s=60 * 5) as client:
        result = send_seqs_for_inference(client, SEQUENCES, SEQS)
    _validate_embeddings(result, EMBEDDINGS, (2, 768))


@pytest.mark.needs_checkpoint
@pytest.mark.needs_gpu
def test_seq_to_embedding_direct(model: ESM1nvInference) -> None:
    infer_fn = triton_embedding_infer_fn(model)
    result = infer_fn([{SEQUENCES: encode_str_batch(SEQS)}])[0]
    _validate_embeddings(result, EMBEDDINGS, (2, 768))


@pytest.mark.needs_checkpoint
@pytest.mark.needs_gpu
# TODO [mgreaves] parameterize later: need expected shapes too
# @pytest.mark.parametrize('seqs', [SEQS])
def test_seq_to_hidden_triton(server) -> None:
    with ModelClient("grpc://localhost:8001", NAME_HIDDENS, inference_timeout_s=60 * 5) as client:
        result = send_seqs_for_inference(client, SEQUENCES, SEQS)
    _validate_embeddings(result, HIDDENS, (2, 43, 768))
    _validate_embeddings(result, MASK, (2, 43))


@pytest.mark.needs_checkpoint
@pytest.mark.needs_gpu
# TODO [mgreaves] parameterize later: need expected shapes too
# @pytest.mark.parametrize('seqs', [SEQS])
def test_seq_to_hidden_direct(model: ESM1nvInference) -> None:
    infer_fn = triton_hidden_infer_fn(model)
    result = infer_fn([{SEQUENCES: encode_str_batch(SEQS)}])[0]
    _validate_embeddings(result, HIDDENS, (2, 43, 768))
    _validate_embeddings(result, MASK, (2, 43))
