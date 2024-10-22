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

import json
from contextlib import ExitStack
from typing import Callable, Dict, Optional, Sequence

import click
import numpy as np
from pytriton.client import ModelClient

from bionemo.triton.types_constants import SEQUENCES
from bionemo.triton.utils import encode_str_batch


__all__: Sequence[str] = ("send_seqs_for_inference",)


def send_seqs_for_inference(client: ModelClient, input_name: str, seqs: Sequence[str]) -> Dict[str, np.ndarray]:
    """Sends a batch of protein sequences for inference to Triton and waits for the result."""
    if len(seqs) == 0:
        raise ValueError("Must supply at least one protein sequence for inference.")
    # sequences = np.char.encode(np.array([[s] for s in seqs]), encoding='utf-8')
    sequences = encode_str_batch(seqs)
    result_dict = client.infer_batch(**{input_name: sequences})
    return result_dict


@click.command("")
@click.option(
    "--triton-url",
    type=str,
    default="http://localhost:8000",
    help="Url to Triton server (e.g. grpc://localhost:8001). "
    "HTTP protocol with default port is used if parameter is not provided",
    show_default=True,
)
@click.option("--model", type=str, default="bionemo_model", help="Name of model in Triton.", show_default=True)
@click.option("--sequences", "-s", type=str, multiple=True, help="Protein sequence(s) to send for inference.")
@click.option(
    "--output",
    type=str,
    required=False,
    help="If provided, writes inference responses to this file as JSON. Otherwise, writes output to STDOUT.",
)
@click.option(
    "--in-name",
    type=str,
    default=SEQUENCES,
    show_default=True,
    help="Name of the Triton model's single character-encoded input tensor.",
)
def entrypoint(
    triton_url: str,
    model: str,
    sequences: Sequence[str],
    output: Optional[str],
    in_name: str,
) -> None:  # pragma: no cover
    print(f"Triton server:      {triton_url}")
    print(f"Model name:         {model}")
    print(f"Protein sequences:  {sequences}")
    print(f"Output to file?:    {output}")
    print(f"Input tensor name:  {in_name}")
    print("-" * 90)

    with ExitStack() as stack:
        handle_output = _output_handler(stack, output)

        client = stack.enter_context(ModelClient(triton_url, model))
        print(f"Sending inference request for {len(sequences)} sequences")
        result_dict = send_seqs_for_inference(client, in_name, sequences)

        handle_output(result_dict)

    print("Complete!")


def _output_handler(
    stack: ExitStack, output: Optional[str]
) -> Callable[[Dict[str, np.ndarray]], None]:  # pragma: no cover
    if output is None:
        print("Inference results:")
        _sep = "-" * 10
        print(_sep)

        def handle(result_dict: Dict[str, np.ndarray]) -> None:
            for output_name, output_batch in result_dict.items():
                print(output_name)
                print(output_batch)
                print(_sep)

    else:
        wt = stack.enter_context(open(output, "wt"))
        # csv.writer(wt, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        print(f"Writing results to {output}")

        def handle(result_dict: Dict[str, np.ndarray]) -> None:
            json.dump({k: v.tolist() for k, v in result_dict.items()}, wt, indent=2)

    return handle


if __name__ == "__main__":  # pragma: no cover
    entrypoint()
