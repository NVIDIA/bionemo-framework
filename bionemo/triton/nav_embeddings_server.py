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


from typing import Sequence

import model_navigator
from nemo.core.config import hydra_runner
from nemo.utils import logging
from omegaconf.omegaconf import OmegaConf
from pytriton.triton import Triton

from bionemo.triton.embeddings import nav_triton_embedding_infer_fn
from bionemo.triton.types_constants import BIONEMO_MODEL, EMBEDDINGS, SEQUENCES
from bionemo.triton.utils import load_navigated_model_for_inference, register_str_embedding_infer_fn


__all_: Sequence[str] = ()


@hydra_runner(config_path="conf", config_name="infer")
def entrypoint(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")

    model, runner = load_navigated_model_for_inference(cfg, strategy=model_navigator.MaxThroughputStrategy())

    with Triton() as triton:
        register_str_embedding_infer_fn(
            triton,
            nav_triton_embedding_infer_fn(model, runner),
            triton_model_name=BIONEMO_MODEL,
            in_name=SEQUENCES,
            out=EMBEDDINGS,
            max_batch_size=10,
            verbose=True,
        )
        logging.info("Serving model")
        triton.serve()


if __name__ == "__main__":
    entrypoint()
