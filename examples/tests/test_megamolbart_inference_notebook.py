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

import os
from concurrent import futures

import grpc
import megamolbart_pb2_grpc
import pytest
from testbook import testbook

from bionemo.model.molecule.megamolbart.grpc.service import InferenceService


os.environ['PROJECT_MOUNT'] = os.environ.get('PROJECT_MOUNT', '/workspace/bionemo')
NOTEBOOK_PATH = os.path.join(os.environ['PROJECT_MOUNT'], 'examples/molecule/megamolbart/nbs/Inference.ipynb')

##########


@pytest.fixture(scope='module')
def grpc_server():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    megamolbart_pb2_grpc.add_GenerativeSamplerServicer_to_server(InferenceService(), server)
    server.add_insecure_port(f'[::]:{50051}')
    server.start()
    yield server
    server.stop(grace=None)


@pytest.fixture(scope='module')
def tb(grpc_server, notebook_path=NOTEBOOK_PATH):
    with testbook(notebook_path, execute=False) as tb:
        notebook_dir = os.path.dirname(notebook_path)
        path_str = """import sys; sys.path.insert(0, '""" + notebook_dir + """')"""
        tb.inject(path_str)
        tb.execute()
        yield tb


@pytest.mark.needs_checkpoint
@pytest.mark.needs_gpu
def test_ipynb_smis_to_hidden(tb):
    output = tb.cell_output_text(5)
    assert output == '(torch.Size([2, 45, 512]), torch.Size([2, 45]))'


@pytest.mark.needs_checkpoint
@pytest.mark.needs_gpu
def test_ipynb_smis_to_embedding(tb):
    output = tb.cell_output_text(7)
    assert output == 'torch.Size([2, 512])'


@pytest.mark.needs_checkpoint
@pytest.mark.needs_gpu
def test_ipynb_hidden_to_smis(tb):
    output = tb.cell_output_text(9)
    assert output == "['c1ccc2ccccc2c1', 'COc1cc2nc(N3CCN(C(=O)c4ccco4)CC3)nc(N)c2cc1OC']"
