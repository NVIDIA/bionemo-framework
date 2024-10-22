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
from typing import List, Optional, Sequence

import click
import numpy as np
from pytriton.client import ModelClient

from bionemo.triton.types_constants import HIDDENS, MASK, SEQUENCES, NamedArrays
from bionemo.triton.utils import decode_str_batch


__all__: Sequence[str] = ("send_masked_embeddings_for_inference",)


def send_masked_embeddings_for_inference(
    client: ModelClient, input_name: str, embeddings: np.ndarray, mask: np.ndarray, output_name: Optional[str]
) -> List[str]:
    """Sends a batch of protein sequences for inference to Triton and waits for the result."""
    result_dict: NamedArrays = client.infer_batch(**{input_name: embeddings, MASK: mask})

    if output_name is not None:
        sequences: np.ndarray = result_dict[output_name]
    elif len(result_dict) == 1:
        sequences = list(result_dict.values())[0]
    else:
        raise ValueError(
            f"Found more than one key in decoded output ({result_dict.keys()=}) and no {output_name=} specified!"
        )

    seqs = decode_str_batch(sequences)
    return seqs


@click.command()
@click.option(
    "--triton-url",
    type=str,
    default="http://localhost:8000",
    help="Url to Triton server (e.g. grpc://localhost:8001). "
    "HTTP protocol with default port is used if parameter is not provided",
    show_default=True,
)
@click.option("--model", type=str, default="bionemo_model", help="Name of model in Triton.", show_default=True)
@click.option(
    "--input",
    type=str,
    required=True,
    help=f"JSON file containing named input tensors. Must contain the embeddings under the --in-name key. If the '{MASK}' binary array is not present in this input object, then a mask of 1s will be used. Otherwise, it must be under the '{MASK}' key.",
)
@click.option(
    "--in-name",
    type=str,
    default=HIDDENS,
    show_default=True,
    help="Name of the non-mask, dense embeddings input tensor for the registered Triton model. Must match what is in the --input.",
)
@click.option(
    "--output",
    type=str,
    required=False,
    help="If provided, writes the decoded input sequences to this file as JSON. Otherwise, writes output to STDOUT.",
)
def entrypoint(
    triton_url: str,
    model: str,
    input: str,
    in_name: str,
    output: Optional[str],
) -> None:  # pragma: no cover
    print(f"Triton server:      {triton_url}")
    print(f"Model name:         {model}")
    print(f"Input JSON file:    {input}")
    print(f"Input tensor name:  {in_name}")
    print(f"Output to file?:    {output}")
    print("-" * 90)

    with open(input, "rt") as rt:
        request = json.load(rt)

    if in_name not in request:
        raise ValueError(
            f"Expecting --in-name={in_name} to be in the --input named array JSON object, but only found input keys: {request.keys()}"
        )
    embeddings = np.array(request[in_name], dtype=np.float32)

    if MASK in request:
        mask: np.ndarray = np.array(request[MASK])
    else:
        mask = np.ones(embeddings.shape[0 : len(embeddings.shape) - 1], dtype=bool)

    with ModelClient(triton_url, model) as client:
        seqs = send_masked_embeddings_for_inference(client, in_name, embeddings, mask, output_name=SEQUENCES)

    if output is None:
        print(f"Decoded {len(seqs)} sequences:")
        for s in seqs:
            print(s)
    else:
        print(f"Writing {len(seqs)} decoded sequences as a JSON list.")
        with open(output, "wt") as wt:
            json.dump(seqs, wt, indent=2)

    print("Complete!")


if __name__ == "__main__":  # pragma: no cover
    entrypoint()
