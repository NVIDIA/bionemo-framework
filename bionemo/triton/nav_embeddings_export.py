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

from typing import Any, Callable, Dict, List, Sequence, Tuple

import model_navigator
import numpy as np
import torch
from model_navigator.api.config import TensorRTProfile
from model_navigator.package.package import Package
from nemo.core.config import hydra_runner
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from bionemo.model.core.infer import M
from bionemo.model.loading import setup_inference
from bionemo.triton.types_constants import EMBEDDINGS
from bionemo.triton.utils import NavWrapper, model_navigator_filepath


__all__: Sequence[str] = (
    "load_and_save_nav_optimized_model",
    "tokenize_batches",
    "navigator_optimize_embedding_model",
    "configure_trt_profile",
    "dynamic_axes_for_bionemo_model",
    "verification_function",
    "INPUT_NAMES",
)

INPUT_NAMES: Sequence[str] = (
    "tokens",
    "mask",
)


def load_and_save_nav_optimized_model(cfg: DictConfig) -> str:
    nav_path = model_navigator_filepath(cfg)

    inferer, _, dataloader = setup_inference(cfg)

    tokenized_batches = tokenize_batches(inferer, dataloader)

    package = navigator_optimize_embedding_model(
        inferer, tokenized_batches, batch_size=cfg.model.data.batch_size, seq_length=cfg.model.seq_length
    )

    model_navigator.package.save(package, nav_path, override=True)

    return nav_path


def tokenize_batches(inferer: M, dataloader: DataLoader) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    batches = []
    for batch in dataloader:
        tokens_enc, enc_mask = inferer.tokenize(batch["sequence"])
        batches.append((tokens_enc, enc_mask))
    return batches


def navigator_optimize_embedding_model(
    model: M,
    tokenized_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    *,
    batch_size: int,
    seq_length: int,
) -> Package:
    return model_navigator.torch.optimize(
        model=NavWrapper(model),
        dataloader=tokenized_batches,
        input_names=INPUT_NAMES,
        output_names=(EMBEDDINGS,),
        custom_configs=[
            model_navigator.TensorRTConfig(trt_profile=configure_trt_profile(batch_size, seq_length, INPUT_NAMES)),
            model_navigator.OnnxConfig(dynamic_axes=dynamic_axes_for_bionemo_model(INPUT_NAMES)),
        ],
        verify_func=verification_function(),
    )


def configure_trt_profile(batch_size: int, seq_length: int, input_tensor_names: Sequence[str]) -> TensorRTProfile:
    trt_profile = TensorRTProfile()
    for name in input_tensor_names:
        trt_profile.add(
            name,
            (1, 1),
            (batch_size, seq_length // 2),
            (batch_size * 2, seq_length),
        )
    return trt_profile


def dynamic_axes_for_bionemo_model(input_tensor_names: Sequence[str]) -> Dict[str, Dict[int, str]]:
    return {name: {0: "batchsize", 1: "seqlen"} for name in input_tensor_names}


def verification_function(
    *, atol: float = 1.0e-3, rtol: float = 1.0e-3
) -> Callable[[List[Dict[str, Any]], List[Dict[str, Any]]], bool]:
    """FIXME: returned function always returns true -- Define verify function that compares outputs of the torch model and the optimized model."""

    def verify_func(ys_runner: List[Dict[str, Any]], ys_expected: List[Dict[str, Any]]) -> bool:
        for y_runner, y_expected in zip(ys_runner, ys_expected):
            for k in y_runner:
                print(f"KEY: {k}")
                exp_range = np.percentile(y_expected[k], (0, 10, 50, 90, 100))
                runner_range = np.percentile(y_runner[k], (0, 10, 50, 90, 100))
                adiff = np.abs(y_runner[k] - y_expected[k]).max()
                rdiff = np.abs(1 - y_runner[k] / y_expected[k]).max()
                print(f"expected:  {exp_range}")
                print(f"runner:    {runner_range}")
                print(f"abs. diff: {adiff}")
                print(f"rel. diff: {rdiff}")
        return True

    return verify_func


@hydra_runner(config_path="conf", config_name="infer")
def entrypoint(cfg) -> None:
    nav_path = load_and_save_nav_optimized_model(cfg)
    print(f"Exported to {nav_path}")


if __name__ == "__main__":
    entrypoint()
