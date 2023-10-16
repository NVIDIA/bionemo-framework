# Copyright (c) 2023, NVIDIA CORPORATION.
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
import numpy as np
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.model_utils import import_class_by_path
from omegaconf.omegaconf import OmegaConf
from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton

from bionemo.model.utils import initialize_distributed_parallel_state


@hydra_runner(config_path="conf", config_name="infer")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    infer_class = import_class_by_path(cfg.infer_target)

    initialize_distributed_parallel_state(
        local_rank=0,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        pipeline_model_parallel_split_rank=0,
    )

    MODEL = infer_class(cfg)
    MODEL.freeze()

    @batch
    def _infer_fn(sequences: np.ndarray):
        sequences = np.char.decode(sequences.astype("bytes"), "utf-8")
        sequences = sequences.squeeze(1).tolist()

        embedding = MODEL.seq_to_embeddings(sequences)

        response = {
            "embedding": embedding.cpu().numpy(),
        }

        return response

    with Triton() as triton:
        logging.info("Loading model")
        triton.bind(
            model_name="bionemo_model",
            infer_func=_infer_fn,
            inputs=[
                Tensor(name="sequences", dtype=bytes, shape=(1,)),
            ],
            outputs=[
                Tensor(name="embedding", dtype=np.float32, shape=(-1,)),
            ],
            config=ModelConfig(max_batch_size=10),
        )
        logging.info("Serving model")
        triton.serve()


if __name__ == "__main__":
    main()
