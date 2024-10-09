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


from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest
from omegaconf import DictConfig
from pytriton.client import ModelClient
from pytriton.triton import Triton
from rdkit import Chem

from bionemo.model.molecule.megamolbart.infer import MegaMolBARTInference
from bionemo.triton.client_decode import send_masked_embeddings_for_inference
from bionemo.triton.client_encode import send_seqs_for_inference
from bionemo.triton.decodes import triton_decode_infer_fn
from bionemo.triton.embeddings import triton_embedding_infer_fn
from bionemo.triton.hiddens import triton_hidden_infer_fn
from bionemo.triton.inference_wrapper import complete_model_name
from bionemo.triton.samplings import triton_sampling_infer_fn
from bionemo.triton.serve_bionemo_model import bind_decode, bind_embedding, bind_hidden, bind_sampling
from bionemo.triton.types_constants import (
    DECODES,
    EMBEDDINGS,
    GENERATED,
    HIDDENS,
    MASK,
    SAMPLINGS,
    SEQUENCES,
)
from bionemo.triton.utils import (
    decode_str_batch,
    encode_str_batch,
    load_model_for_inference,
)
from bionemo.utils.hydra import load_model_config
from bionemo.utils.tests import teardown_apex_megatron_cuda


SMILES = ["c1ccc2ccccc2c1", "COc1cc2nc(N3CCN(C(=O)c4ccco4)CC3)nc(N)c2cc1OC"]

MODEL_NAME = "megamolbart"

NAME_EMBEDDINGS = complete_model_name(MODEL_NAME, EMBEDDINGS)
NAME_HIDDENS = complete_model_name(MODEL_NAME, HIDDENS)
NAME_SAMPLINGS = complete_model_name(MODEL_NAME, SAMPLINGS)
NAME_DECODES = complete_model_name(MODEL_NAME, DECODES)


@pytest.fixture(scope="module")
def cfg(bionemo_home: Path) -> DictConfig:
    return load_model_config(
        config_path=str(bionemo_home / "examples" / "molecule" / MODEL_NAME / "conf"),
        config_name="infer.yaml",
        logger=None,
    )


@pytest.fixture(scope="module")
def model(cfg: DictConfig) -> MegaMolBARTInference:
    # TODO [mgreaves] replace with this in !553
    # model = load_model_for_inference(cfg, interactive=False)
    yield load_model_for_inference(cfg, inference_batch_size_for_warmup=len(SMILES))
    teardown_apex_megatron_cuda()


@pytest.fixture(scope="module")
def server(cfg: DictConfig, model: MegaMolBARTInference) -> Triton:
    triton = Triton()
    bind_embedding(triton, cfg, model, nav=False, triton_model_name=NAME_EMBEDDINGS)
    bind_hidden(triton, cfg, model, nav=False, triton_model_name=NAME_HIDDENS)
    bind_decode(triton, cfg, model, nav=False, triton_model_name=NAME_DECODES)
    bind_sampling(triton, cfg, model, nav=False, triton_model_name=NAME_SAMPLINGS)
    triton.run()
    yield triton
    triton.stop()


def canonicalize_smiles(smiles: str) -> str:
    """Canonicalize input SMILES"""
    mol = Chem.MolFromSmiles(smiles)
    canon_smiles = Chem.MolToSmiles(mol, canonical=True)
    return canon_smiles


def _validate_embeddings(result: Dict[str, np.ndarray], key: str, expected_shape: Tuple[int]) -> None:
    assert isinstance(result, dict), f"Expecting dict-like but found {type(result)=}"
    assert key in result, f"Expecting {key} but only found {result.keys()=}"
    embeddings = result[key]
    assert embeddings.shape == expected_shape


def _validate_to_seqs(smis: List[str]) -> None:
    assert len(smis) > 0, "Expecting non-empty decoded output"
    for i, x in enumerate(smis):
        assert len(x) > 0, f"Expecting non-empty decoded output #{i+1}"

    assert len(smis) == len(SMILES), f"Expecting to decode {len(SMILES)} but actually decoded {len(smis)}"

    for i, (result, expected) in enumerate(zip(smis, SMILES)):
        actual = canonicalize_smiles(result)
        assert expected == actual, f"Failure on decoding SMILE input #{i+1}, {expected=} vs. {actual=}"


def _validate_generated(generated: np.ndarray, expected_shape: Tuple[int]) -> None:
    assert generated.shape == expected_shape

    for i in range(len(generated)):
        gen = generated[i]
        assert len(gen) >= 1, f"Expecting batch #{i+1} of generated samples to be non-empty"
        for j in range(len(gen)):
            try:
                sample = gen[j].decode("utf8")
            except Exception as e:
                raise ValueError(f"Error decoding generated sample ({i},{j}): {gen[j]}") from e
            assert len(sample) >= 1, f"Expecting sample ({i},{j}) to be non-empty"


#### Triton-based


@pytest.mark.needs_checkpoint
@pytest.mark.needs_gpu
# TODO [mgreaves] parameterize later: need expected shapes too
# @pytest.mark.parametrize('seqs', [SEQS])
def test_seq_to_embedding_triton(server):
    with ModelClient("grpc://localhost:8001", NAME_EMBEDDINGS, inference_timeout_s=60 * 5) as client:
        result = send_seqs_for_inference(client, SEQUENCES, SMILES)

    _validate_embeddings(result, EMBEDDINGS, (2, 512))


@pytest.mark.needs_checkpoint
@pytest.mark.needs_gpu
# TODO [mgreaves] parameterize later: need expected shapes too
# @pytest.mark.parametrize('seqs', [SEQS])
def test_seq_to_hidden_triton(server) -> None:
    with ModelClient("grpc://localhost:8001", NAME_HIDDENS, inference_timeout_s=60 * 5) as client:
        result = send_seqs_for_inference(client, SEQUENCES, SMILES)

    _validate_embeddings(result, HIDDENS, (2, 45, 512))
    _validate_embeddings(result, MASK, (2, 45))


@pytest.mark.needs_checkpoint
@pytest.mark.needs_gpu
# TODO [mgreaves] parameterize later: need expected shapes too
# @pytest.mark.parametrize('seqs', [SEQS])
def test_hidden_to_seqs_triton(server) -> None:
    with ModelClient("grpc://localhost:8001", NAME_HIDDENS, inference_timeout_s=60 * 5) as client:
        result = send_seqs_for_inference(client, SEQUENCES, SMILES)

    _validate_embeddings(result, HIDDENS, (2, 45, 512))
    _validate_embeddings(result, MASK, (2, 45))
    hidden_states = result[HIDDENS]
    masks = result[MASK]

    with ModelClient("grpc://localhost:8001", NAME_DECODES, inference_timeout_s=60 * 5) as client:
        smis = send_masked_embeddings_for_inference(client, HIDDENS, hidden_states, masks, output_name=SEQUENCES)

    _validate_to_seqs(smis)


@pytest.mark.needs_checkpoint
@pytest.mark.needs_gpu
def test_samplings_triton(server) -> None:
    with ModelClient("grpc://localhost:8001", NAME_SAMPLINGS, inference_timeout_s=60 * 5) as client:
        result = send_seqs_for_inference(client, SEQUENCES, SMILES)

    assert GENERATED in result, f"Expecting {GENERATED} but found {result.keys()=}"
    generated: np.ndarray = result[GENERATED]
    _validate_generated(generated, expected_shape=(2, 1))


#### Direct inference-based


@pytest.mark.needs_checkpoint
@pytest.mark.needs_gpu
# TODO [mgreaves] parameterize later: need expected shapes too
# @pytest.mark.parametrize('seqs', [SEQS])
def test_seq_to_embedding_direct(model):
    infer_fn = triton_embedding_infer_fn(model)
    result = infer_fn([{SEQUENCES: encode_str_batch(SMILES)}])[0]

    _validate_embeddings(result, EMBEDDINGS, (2, 512))


@pytest.mark.needs_checkpoint
@pytest.mark.needs_gpu
# TODO [mgreaves] parameterize later: need expected shapes too
# @pytest.mark.parametrize('seqs', [SEQS])
def test_seq_to_hidden_direct(model) -> None:
    infer_fn = triton_hidden_infer_fn(model)
    result = infer_fn([{SEQUENCES: encode_str_batch(SMILES)}])[0]

    _validate_embeddings(result, HIDDENS, (2, 45, 512))
    _validate_embeddings(result, MASK, (2, 45))


@pytest.mark.needs_checkpoint
@pytest.mark.needs_gpu
# TODO [mgreaves] parameterize later: need expected shapes too
# @pytest.mark.parametrize('seqs', [SEQS])
def test_hidden_to_seqs_direct(model) -> None:
    hidden_infer_fn = triton_hidden_infer_fn(model)
    result = hidden_infer_fn([{SEQUENCES: encode_str_batch(SMILES)}])[0]

    _validate_embeddings(result, HIDDENS, (2, 45, 512))
    _validate_embeddings(result, MASK, (2, 45))
    hidden_states = result[HIDDENS]
    masks = result[MASK]

    decode_infer_fn = triton_decode_infer_fn(model)
    result = decode_infer_fn([{HIDDENS: hidden_states, MASK: masks}])[0]

    assert SEQUENCES in result, f"Expecting {SEQUENCES} but found {result.keys()=}"
    smis = decode_str_batch(result[SEQUENCES])
    _validate_to_seqs(smis)


@pytest.mark.needs_checkpoint
@pytest.mark.needs_gpu
def test_samplings_direct(model) -> None:
    infer_fn = triton_sampling_infer_fn(model)
    result = infer_fn([{SEQUENCES: encode_str_batch(SMILES)}])[0]
    assert GENERATED in result, f"Expecting {GENERATED} but found {result.keys()=}"
    generated = result[GENERATED]
    _validate_generated(generated, expected_shape=(2, 1))
