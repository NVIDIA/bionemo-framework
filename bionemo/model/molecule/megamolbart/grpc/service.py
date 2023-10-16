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
from concurrent import futures

import grpc
import megamolbart_pb2_grpc
import torch
from hydra import compose, initialize
from megamolbart_pb2 import OutputSpec

from bionemo.model.molecule.megamolbart import MegaMolBARTInference


logger = logging.getLogger(__name__)


class InferenceService(megamolbart_pb2_grpc.GenerativeSampler):
    def __init__(self):
        if not hasattr(self, '_inferer'):
            with initialize(config_path="../../../../../examples/molecule/megamolbart/conf"):
                inf_cfg = compose(config_name="infer")
                self._inferer = MegaMolBARTInference(cfg=inf_cfg)

    def SmilesToEmbedding(self, spec, context):
        embeddings = self._inferer.seq_to_embeddings(spec.smis)
        output = OutputSpec(embeddings=embeddings.flatten().tolist(), dim=embeddings.shape)
        return output

    def SmilesToHidden(self, spec, context):
        hidden_states, pad_masks = self._inferer.seq_to_hiddens(spec.smis)
        output = OutputSpec(
            hidden_states=hidden_states.flatten().tolist(), dim=hidden_states.shape, masks=pad_masks.flatten().tolist()
        )
        return output

    def HiddenToSmis(self, spec, context):
        pad_mask = torch.BoolTensor(list(spec.masks))
        pad_mask = torch.reshape(pad_mask, tuple(spec.dim[:2])).cuda()

        hidden_states = torch.FloatTensor(list(spec.hidden_states))
        hidden_states = torch.reshape(hidden_states, tuple(spec.dim)).cuda()

        smis = self._inferer.hiddens_to_seq(hidden_states, pad_mask)
        output = OutputSpec(smis=smis)
        return output


def main():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    megamolbart_pb2_grpc.add_GenerativeSamplerServicer_to_server(InferenceService(), server)
    server.add_insecure_port(f'[::]:{50051}')
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    main()
