# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
from pathlib import Path

from esm.export import export_hf_checkpoint


ESM_TAGS = [
    "esm2_t6_8M_UR50D",
    "esm2_t12_35M_UR50D",
    "esm2_t30_150M_UR50D",
    "esm2_t33_650M_UR50D",
    "esm2_t36_3B_UR50D",
    "esm2_t48_15B_UR50D",
]


def main():
    """Export the ESM2 models from Hugging Face to the Transformer Engine format."""
    parser = argparse.ArgumentParser(description="Convert ESM2 models from Hugging Face to Transformer Engine format")

    parser.add_argument(
        "--model",
        type=str,
        choices=ESM_TAGS,
        help="Specific model tag to convert. If not provided, all models will be converted.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./checkpoint_export",
        help="Output directory path for the converted model. Defaults to './checkpoint_export'",
    )

    args = parser.parse_args()

    if args.model:
        if args.model not in ESM_TAGS:
            print(f"Error: '{args.model}' is not a valid model tag.\nAvailable models: {', '.join(ESM_TAGS)}")
            return
        export_hf_checkpoint(args.model, Path(args.output_path))
    else:
        for tag in ESM_TAGS:
            print(f"Converting {tag}...")
            export_hf_checkpoint(tag, Path(args.output_path))


if __name__ == "__main__":
    main()
