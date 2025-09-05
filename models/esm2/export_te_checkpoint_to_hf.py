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

from esm.export import export_te_checkpoint


def main(te_checkpoint_dir: str, output_dir: str):
    """Export Transformer Engine checkpoints back to original Facebook HuggingFace ESM-2 format."""
    print(f"Converting {te_checkpoint_dir} from TE format back to HuggingFace Facebook ESM-2 format...")

    if not Path(te_checkpoint_dir).exists():
        raise FileNotFoundError(f"TE checkpoint {te_checkpoint_dir} not found")

    try:
        export_te_checkpoint(te_checkpoint_dir, output_dir)
        print(f"Successfully exported {te_checkpoint_dir} to HuggingFace Facebook ESM-2 format at {output_dir}")

    except Exception as e:
        print(f"Error converting {te_checkpoint_dir}: {e}")
        raise e

    print("Export completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert ESM2 models from Transformer Engine format back to HuggingFace Facebook ESM-2 format hosted on Hugging Face"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Path to the TE checkpoint.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./hf_checkpoints",
        help="Base output directory for the converted models. Each checkpoint will be saved in a subdirectory named after the checkpoint. If not provided, uses './hf_checkpoints'.",
    )
    args = parser.parse_args()

    main(te_checkpoint_dir=args.model, output_dir=args.output_path)
