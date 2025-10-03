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

from esm.export import export_hf_checkpoint, export_te_checkpoint


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

    subparsers = parser.add_subparsers(dest="conversion_type", required=True, help="Type of conversion to perform")

    hf_to_te_parser = subparsers.add_parser("hf-to-te", help="Convert from HuggingFace to Transformer Engine format")
    hf_to_te_parser.add_argument(
        "--model",
        type=str,
        choices=ESM_TAGS,
        help="Specific model tag to convert. If not provided, all models will be converted.",
    )
    hf_to_te_parser.add_argument(
        "--output-path",
        type=str,
        default="./hf_to_te_checkpoint_export",
        help="Output directory path for the converted model. Defaults to './hf_to_te_checkpoint_export'",
    )

    te_to_hf_parser = subparsers.add_parser("te-to-hf", help="Convert from Transformer Engine to HuggingFace format")
    te_to_hf_parser.add_argument(
        "--checkpoint-path", type=str, required=True, help="Path to the Transformer Engine checkpoint to convert"
    )
    te_to_hf_parser.add_argument(
        "--output-path",
        type=str,
        default="./te_to_hf_checkpoint_export",
        help="Output directory path for the converted model. Defaults to './te_to_hf_checkpoint_export'",
    )
    
    args = parser.parse_args()

    if args.conversion_type == "hf-to-te":
        if args.model:
            print(f"Converting {args.model} from HuggingFace to Transformer Engine format")
            export_hf_checkpoint(args.model, Path(args.output_path))
        else:
            for tag in ESM_TAGS:
                print(f"Converting {tag} from HuggingFace to Transformer Engine format")
                export_hf_checkpoint(tag, Path(args.output_path))
    else:
        print(f"Converting {args.checkpoint_path} from Transformer Engine to HuggingFace format")
        export_te_checkpoint(args.checkpoint_path, Path(args.output_path))


if __name__ == "__main__":
    main()
