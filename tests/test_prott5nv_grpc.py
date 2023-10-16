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
import numpy as np
import prott5_pb2_grpc
import pytest
from prott5_pb2 import InputSpec
from prott5_pb2_grpc import GenerativeSamplerStub

from bionemo.model.protein.prott5nv.grpc.service import InferenceService


SEQS = ['MSLKRKNIALIPAAGIGVRFGADKPKQYVEIGSKTVLEHVL', 'MIQSQINRNIRLDLADAILLSKAKKDLSFAEIADGTGLA']

##########


@pytest.fixture(scope='module')
def grpc_server():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    prott5_pb2_grpc.add_GenerativeSamplerServicer_to_server(InferenceService(), server)
    server.add_insecure_port(f'[::]:{50051}')
    server.start()
    yield server
    server.stop(grace=None)


@pytest.fixture(scope='module')
def grpc_stub():
    channel = grpc.insecure_channel('localhost:50051')
    return GenerativeSamplerStub(channel)


@pytest.mark.needs_checkpoint
@pytest.mark.needs_gpu
@pytest.mark.parametrize('seqs', [SEQS])
def test_seq_to_embedding(grpc_server, grpc_stub, seqs):
    spec = InputSpec(seqs=seqs)
    resp = grpc_stub.SeqToEmbedding(spec)
    embeddings = np.array(list(resp.embeddings))
    embeddings = np.reshape(embeddings, tuple(resp.dim))
    assert embeddings.shape == (2, 41, 768)
