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
from prott5_pb2 import InputSpec
from prott5_pb2_grpc import GenerativeSamplerStub


log = logging.getLogger(__name__)


class ProtT5nvInferenceWrapper:
    def __init__(self):
        channel = grpc.insecure_channel('localhost:50051')
        self.stub = GenerativeSamplerStub(channel)

    def seq_to_hiddens(self, seqs):
        spec = InputSpec(seqs=seqs)
        resp = self.stub.SeqToEmbedding(spec)

        hidden_states = torch.FloatTensor(list(resp.embeddings))
        hidden_states = torch.reshape(hidden_states, tuple(resp.dim)).cuda()
        max_seq_length = hidden_states.size()[1]
        enc_mask = []

        for s in seqs:
            enc_mask.append(torch.cat([torch.ones(1, len(s)), torch.zeros(1, max_seq_length - len(s))], dim=1))
        enc_mask = torch.cat(enc_mask, dim=0).cuda()

        return hidden_states, enc_mask

    def hiddens_to_embedding(self, hidden_states, enc_mask):
        # compute average on active hiddens
        lengths = enc_mask.sum(dim=1, keepdim=True)
        if (lengths == 0).any():
            raise ValueError("Empty input is not supported (no token was proveded in one or more of the inputs)")

        embeddings = torch.sum(hidden_states * enc_mask.unsqueeze(-1), dim=1) / lengths

        return embeddings

    def seq_to_embedding(self, seqs):
        hidden_states, enc_mask = self.seq_to_hiddens(seqs)
        embeddings = self.hiddens_to_embedding(hidden_states, enc_mask)
        return embeddings
