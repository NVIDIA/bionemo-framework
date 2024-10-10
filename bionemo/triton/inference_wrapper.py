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


from dataclasses import dataclass
from typing import List, Sequence, Tuple, Union

import click
import numpy as np
import torch
from pytriton.client import ModelClient

from bionemo.triton.client_decode import send_masked_embeddings_for_inference
from bionemo.triton.client_encode import send_seqs_for_inference
from bionemo.triton.serve_bionemo_model import main
from bionemo.triton.types_constants import (
    BIONEMO_MODEL,
    DECODES,
    EMBEDDINGS,
    GENERATED,
    HIDDENS,
    MASK,
    SAMPLINGS,
    SEQUENCES,
)


__all__: Sequence[str] = (
    "InferenceWrapper",
    "new_inference_wrapper",
    "complete_model_name",
)


@dataclass
class InferenceWrapper:
    embeddings_client: ModelClient
    hiddens_client: ModelClient
    samplings_client: ModelClient
    decodes_client: ModelClient

    def seqs_to_embedding(self, seqs: List[str]) -> torch.Tensor:
        result = send_seqs_for_inference(self.embeddings_client, SEQUENCES, seqs)
        embeddings = torch.Tensor(result[EMBEDDINGS])
        return embeddings

    def seqs_to_hidden(self, seqs: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        result = send_seqs_for_inference(self.hiddens_client, SEQUENCES, seqs)
        if HIDDENS not in result:
            raise ValueError(f"Expecting {HIDDENS} but only found {result.keys()=}")
        if MASK not in result:
            raise ValueError(f"Expecting {MASK} but only found {result.keys()=}")
        hiddens = torch.Tensor(result[HIDDENS])
        masks = torch.Tensor(result[MASK]).to(dtype=bool)
        return hiddens, masks

    def hidden_to_seqs(self, hidden_states: torch.Tensor, masks: torch.Tensor) -> List[str]:
        seqs = send_masked_embeddings_for_inference(
            self.decodes_client,
            # DECODES,
            HIDDENS,
            hidden_states.detach().cpu().numpy(),
            masks.detach().cpu().numpy(),
            output_name=SEQUENCES,
        )
        return seqs

    def sample_seqs(self, seqs: List[str]) -> List[List[str]]:
        result = send_seqs_for_inference(self.samplings_client, SEQUENCES, seqs)
        generated: np.ndarray = result[GENERATED]
        sequences = np.char.decode(generated.astype("bytes"), "utf-8")
        return sequences.tolist()

    def hiddens_to_embedding(self, hidden_states: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        lengths = masks.sum(dim=1, keepdim=True)
        if (lengths == 0).any():
            raise ValueError("Empty input is not supported. No token was proveded in one or more of the inputs!")
        embeddings = torch.sum(hidden_states * masks.unsqueeze(-1), dim=1) / lengths
        return embeddings

    def __enter__(self) -> "InferenceWrapper":
        return self

    def __exit__(self, *args, **kwargs) -> None:
        self._close()

    def _close(self) -> None:
        for c in [self.decodes_client, self.embeddings_client, self.samplings_client, self.hiddens_client]:
            try:
                c.close()
            except:  # noqa
                pass

    def __del__(self) -> None:
        self._close()


def new_inference_wrapper(triton_url: str, base_model_name: str = BIONEMO_MODEL) -> InferenceWrapper:
    return InferenceWrapper(
        samplings_client=ModelClient(triton_url, complete_model_name(base_model_name, SAMPLINGS)),
        decodes_client=ModelClient(triton_url, complete_model_name(base_model_name, DECODES)),
        embeddings_client=ModelClient(triton_url, complete_model_name(base_model_name, EMBEDDINGS)),
        hiddens_client=ModelClient(triton_url, complete_model_name(base_model_name, HIDDENS)),
    )


def complete_model_name(
    base_model_name: str, inference_type: Union[MASK, EMBEDDINGS, HIDDENS, DECODES, SEQUENCES, SAMPLINGS, GENERATED]
) -> str:
    return f"{base_model_name}_{inference_type}"


@click.command()
@click.option("--config-path", required=True, help="Path to Hydra config directory where configuration date lives.")
@click.option(
    "--config-name",
    default="infer.yaml",
    show_default=True,
    required=True,
    help="Name of YAML config file in --config-path to load from.",
)
@click.option(
    "--nav", is_flag=True, help="If present, load runtime optimized with model navigator. Requires export beforehand."
)
@click.option(
    "--model-name-base",
    required=True,
    default=BIONEMO_MODEL,
    help="Common prefix to use to group all of the loaded model's bound inference functions. "
    f"This program binds {SAMPLINGS}, {EMBEDDINGS}, {DECODES}, and {HIDDENS} inference functions for the model. "
    "It uses the format '<--model-name-base>_<one of those names>' to uniquely identify each inference function.",
)
def entrypoint(config_path: str, config_name: str, nav: bool, model_name_base: str) -> None:  # pragma: no cover
    main(
        config_path=config_path,
        config_name=config_name,
        nav=nav,
        embedding=complete_model_name(model_name_base, EMBEDDINGS),
        sampling=complete_model_name(model_name_base, SAMPLINGS),
        decode=complete_model_name(model_name_base, DECODES),
        hidden=complete_model_name(model_name_base, HIDDENS),
        allow_override_name=False,
    )


if __name__ == "__main__":  # pragma: no cover
    entrypoint()
