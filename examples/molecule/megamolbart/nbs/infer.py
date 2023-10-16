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

import logging

import grpc
import torch
from megamolbart_pb2 import InputSpec
from megamolbart_pb2_grpc import GenerativeSamplerStub


log = logging.getLogger(__name__)


class InferenceWrapper:
    def __init__(self):
        channel = grpc.insecure_channel('localhost:50051')
        self.stub = GenerativeSamplerStub(channel)

    def smis_to_embedding(self, smis):
        spec = InputSpec(smis=smis)
        resp = self.stub.SmilesToEmbedding(spec)

        embeddings = torch.FloatTensor(list(resp.embeddings))
        embeddings = torch.reshape(embeddings, tuple(resp.dim)).cuda()

        return embeddings

    def smis_to_hidden(self, smis):
        spec = InputSpec(smis=smis)
        resp = self.stub.SmilesToHidden(spec)

        hidden_states = torch.FloatTensor(list(resp.hidden_states))
        hidden_states = torch.reshape(hidden_states, tuple(resp.dim)).cuda()
        masks = torch.BoolTensor(list(resp.masks))
        masks = torch.reshape(masks, tuple(resp.dim[:2])).cuda()

        return hidden_states, masks

    def hidden_to_smis(self, hidden_states, masks):
        dim = hidden_states.shape
        spec = InputSpec(hidden_states=hidden_states.flatten().tolist(), dim=dim, masks=masks.flatten().tolist())

        resp = self.stub.HiddenToSmis(spec)
        return resp.smis
