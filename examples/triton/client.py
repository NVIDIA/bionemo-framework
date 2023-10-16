#!/usr/bin/env python3
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
"""Client for seeded molecule generation with MegaMolBART."""
import argparse
import pickle

import numpy as np
from pytriton.client import ModelClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default="localhost",
        help=(
            "Url to Triton server (ex. grpc://localhost:8001)."
            "HTTP protocol with default port is used if parameter is not provided"
        ),
        required=False,
    )
    parser.add_argument("--sequences", type=str, nargs='+', required=True)
    parser.add_argument("--output_path", type=str, required=False)
    args = parser.parse_args()
    print(f"Input sequences: {args.sequences}")
    sequences = np.array([[seq] for seq in args.sequences])
    sequences = np.char.encode(sequences, "utf-8")

    with ModelClient(args.url, "bionemo_model") as client:
        print("Sending request")
        result_dict = client.infer_batch(sequences)

    if args.output_path is None:
        print("Inference results:")
        for output_name, output_batch in result_dict.items():
            print(output_name)
            print(output_batch)
    else:
        with open(args.output_path, 'wb') as f:
            pickle.dump(result_dict, f)
        print(f"Results written to {args.output_path}")


if __name__ == "__main__":
    main()
