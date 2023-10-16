# Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

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

from concurrent import futures

import grpc
import megamolbart_pb2_grpc
import numpy as np
import pytest
from megamolbart_pb2 import InputSpec
from megamolbart_pb2_grpc import GenerativeSamplerStub
from rdkit import Chem

from bionemo.model.molecule.megamolbart.grpc.service import InferenceService


SMILES = ['c1ccc2ccccc2c1', 'COc1cc2nc(N3CCN(C(=O)c4ccco4)CC3)nc(N)c2cc1OC']

##########


def canonicalize_smiles(smiles: str) -> str:
    """Canonicalize input SMILES"""
    mol = Chem.MolFromSmiles(smiles)
    canon_smiles = Chem.MolToSmiles(mol, canonical=True)
    return canon_smiles


@pytest.fixture(scope='module')
def grpc_server():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    megamolbart_pb2_grpc.add_GenerativeSamplerServicer_to_server(InferenceService(), server)
    server.add_insecure_port(f'[::]:{50051}')
    server.start()
    yield server
    server.stop(grace=None)


@pytest.fixture(scope='module')
def grpc_stub():
    channel = grpc.insecure_channel('localhost:50051')
    return GenerativeSamplerStub(channel)


def get_hidden_states(smis, grpc_stub):
    spec = InputSpec(smis=smis)
    resp = grpc_stub.SmilesToHidden(spec)

    hidden_states = np.array(list(resp.hidden_states))
    hidden_states = np.reshape(hidden_states, tuple(resp.dim))
    masks = np.array(list(resp.masks))
    masks = np.reshape(masks, tuple(resp.dim[:2]))

    return hidden_states, masks


@pytest.mark.needs_checkpoint
@pytest.mark.needs_gpu
@pytest.mark.parametrize('smis', [SMILES])
def test_smis_to_embedding(grpc_server, grpc_stub, smis):
    spec = InputSpec(smis=smis)
    resp = grpc_stub.SmilesToEmbedding(spec)
    embeddings = np.array(list(resp.embeddings))
    embeddings = np.reshape(embeddings, tuple(resp.dim))
    assert embeddings.shape == (2, 512)


@pytest.mark.needs_checkpoint
@pytest.mark.needs_gpu
@pytest.mark.parametrize('smis', [SMILES])
def test_smis_to_hidden(grpc_server, grpc_stub, smis):
    hidden_states, masks = get_hidden_states(smis, grpc_stub)
    assert hidden_states.shape == (2, 45, 512)
    assert masks.shape == (2, 45)


@pytest.mark.needs_checkpoint
@pytest.mark.needs_gpu
@pytest.mark.parametrize('smis', [SMILES])
def test_hidden_to_smis(grpc_server, grpc_stub, smis):
    hidden_states, masks = get_hidden_states(smis, grpc_stub)
    dim = hidden_states.shape
    spec = InputSpec(hidden_states=hidden_states.flatten().tolist(), dim=dim, masks=masks.flatten().tolist())
    resp = grpc_stub.HiddenToSmis(spec)

    resp_canon_smiles = [canonicalize_smiles(x) for x in resp.smis]
    assert resp_canon_smiles == smis
