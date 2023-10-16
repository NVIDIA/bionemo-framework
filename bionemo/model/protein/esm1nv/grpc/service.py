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

import argparse
import logging
import os
from concurrent import futures

import esm1nv_pb2_grpc
import grpc
from esm1nv_pb2 import OutputSpec
from hydra import compose, initialize

from bionemo.model.protein.esm1nv import ESM1nvInference


logger = logging.getLogger(__name__)

MODEL_MAPPING = {
    "esm1nv": ("../../../../../examples/protein/esm1nv/conf", "/model/protein/esm1nv/esm1nv.nemo"),
    "esm2nv_650M": ("../../../../../examples/protein/esm2nv/conf", "/model/protein/esm2nv/esm2nv_650M_converted.nemo"),
    "esm2nv_3B": ("../../../../../examples/protein/esm2nv/conf", "/model/protein/esm2nv/esm2nv_3B_converted.nemo"),
}


class InferenceService(esm1nv_pb2_grpc.GenerativeSampler):
    def __init__(self, config_path=None, config_name=None, model=None):
        """
        Initialize an InferenceService for gRPC client, which prepares
        an inference class `ESM1nvInference` with the specified .nemo checkpoints.

        Parameters:
        ----------
        config_path: str, optional
            The path to load the model's configuration from.
            By default None.
        config_name: str, optional
            The name of the configuration to load.
            By default None.
        model: str, optional
            The name of the model to load. If specified, the default
            converted model will be loaded. Note that the model cannot
            be set when using config_path and config_name.
            By default None.
        """
        if model:
            if (config_path is not None) or (config_name is not None):
                raise ValueError("If 'model' is specified, 'config_path' and 'config_name' must be None.")
            config_path = MODEL_MAPPING[model][0]
        if config_path is None:
            config_path = "../../../../../examples/protein/esm1nv/conf"
        if config_name is None:
            config_name = "infer"
        if not hasattr(self, '_inferer'):
            with initialize(config_path=config_path):
                inf_cfg = compose(config_name=config_name)
                if model:
                    inf_cfg.model.downstream_task.restore_from_path = MODEL_MAPPING[model][1]
                self._inferer = ESM1nvInference(cfg=inf_cfg)

    def SeqToEmbedding(self, spec, context):
        embeddings, masks = self._inferer.seq_to_hiddens(spec.seqs)
        embeddings[~masks.bool()] = 0
        embeddings = embeddings[:, masks.sum(axis=0) != 0, :]
        output = OutputSpec(embeddings=embeddings.flatten().tolist(), dim=embeddings.shape)
        return output


def OptionalRelativePath(arg):
    if arg is None:
        return arg
    if os.path.isabs(arg):
        raise argparse.ArgumentTypeError(f"'{arg}' is not a relative path. Relative path is required.")
    return arg


def main():
    parser = argparse.ArgumentParser(description="Start a GRPC inference server for ESM")
    parser.add_argument('--config-path', help='Path to config file directory', default=None, type=OptionalRelativePath)
    parser.add_argument('--config-name', help='Path to config file', default=None)
    parser.add_argument(
        '--model',
        help='Name of the ESM model to load',
        choices=[None, "esm1nv", "esm2nv_650M", "esm2nv_3B"],
        default=None,
    )

    args = parser.parse_args()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    esm1nv_pb2_grpc.add_GenerativeSamplerServicer_to_server(
        InferenceService(args.config_path, args.config_name, args.model), server
    )
    server.add_insecure_port(f'[::]:{50051}')
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    main()
