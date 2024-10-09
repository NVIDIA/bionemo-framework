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
from typing import Dict, Tuple

import numpy as np
import pytest
from omegaconf import DictConfig
from pytriton.client import ModelClient
from pytriton.triton import Triton

from bionemo.model.singlecell.geneformer.infer import GeneformerInference
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
    load_model_for_inference,
)
from bionemo.utils.hydra import load_model_config
from bionemo.utils.tests import teardown_apex_megatron_cuda


# Each "cell" is an ordered list of gene names
SEQS = [
    ["SCYGR3", "NBPF3", "SCYGR6", "RNA5SP114", "MIR561", "GULP1", "RNU6-298P", "TEFM", "PFN1P10"],
    ["RNU2-38P", "RNVU1-24", "SLC25A34", "SLC25A34-AS1", "TMEM82", "FBLIM1", "SUZ12P1", "CRLF3"],
]

MODEL_NAME = "geneformer"

NAME_EMBEDDINGS = complete_model_name(MODEL_NAME, EMBEDDINGS)
NAME_HIDDENS = complete_model_name(MODEL_NAME, HIDDENS)


@pytest.fixture(scope="module")
def cfg(bionemo_home: Path) -> DictConfig:
    return load_model_config(
        config_path=str(bionemo_home / "examples" / "singlecell" / MODEL_NAME / "conf"),
        config_name="infer.yaml",
        logger=None,
    )


@pytest.fixture(scope="module")
def model(cfg: DictConfig) -> GeneformerInference:
    # TODO [mgreaves] replace with this in !553
    # model = load_model_for_inference(cfg, interactive=False)
    yield load_model_for_inference(cfg, inference_batch_size_for_warmup=len(SEQS))
    teardown_apex_megatron_cuda()


@pytest.fixture(scope="module")
def server(cfg: DictConfig, model: GeneformerInference) -> Triton:
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
@pytest.mark.xfail(reason="Need to support list of gene names format in encode_str_batch.")
def test_seq_to_embedding_triton(server: Triton) -> None:
    with ModelClient("grpc://localhost:8001", NAME_EMBEDDINGS, inference_timeout_s=60 * 5) as client:
        result = send_seqs_for_inference(client, SEQUENCES, SEQS)
    _validate_embeddings(result, EMBEDDINGS, (2, 256))


@pytest.mark.needs_checkpoint
@pytest.mark.needs_gpu
@pytest.mark.xfail(reason="Need to support list of gene names format in encode_str_batch.")
def test_seq_to_embedding_direct(model: GeneformerInference) -> None:
    infer_fn = triton_embedding_infer_fn(model)
    result = infer_fn([{SEQUENCES: encode_str_batch(SEQS)}])[0]
    _validate_embeddings(result, EMBEDDINGS, (2, 256))


@pytest.mark.needs_checkpoint
@pytest.mark.needs_gpu
# TODO [mgreaves] parameterize later: need expected shapes too
# @pytest.mark.parametrize('seqs', [SEQS])
@pytest.mark.xfail(reason="Need to support list of gene names format in encode_str_batch.")
def test_seq_to_hidden_triton(server) -> None:
    with ModelClient("grpc://localhost:8001", NAME_HIDDENS, inference_timeout_s=60 * 5) as client:
        result = send_seqs_for_inference(client, SEQUENCES, SEQS)
    _validate_embeddings(result, HIDDENS, (2, 10, 256))
    _validate_embeddings(result, MASK, (2, 10))


@pytest.mark.needs_checkpoint
@pytest.mark.needs_gpu
# TODO [mgreaves] parameterize later: need expected shapes too
# @pytest.mark.parametrize('seqs', [SEQS])
@pytest.mark.xfail(reason="Need to support list of gene names format in encode_str_batch.")
def test_seq_to_hidden_direct(model: GeneformerInference) -> None:
    infer_fn = triton_hidden_infer_fn(model)
    result = infer_fn([{SEQUENCES: encode_str_batch(SEQS)}])[0]
    _validate_embeddings(result, HIDDENS, (2, 10, 256))
    _validate_embeddings(result, MASK, (2, 10))
