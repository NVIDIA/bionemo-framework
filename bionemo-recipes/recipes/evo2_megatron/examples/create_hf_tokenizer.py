# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Arc Institute. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Michael Poli. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Stanford University. All rights reserved
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
import os
import shutil
from pathlib import Path

from transformers import AutoTokenizer

from bionemo.evo2.data.tokenizer import AsciiOrdTokenizer


def setup_and_save_tokenizer(output_dir: Path, vocab_size_limit: int):
    """Initializes the custom tokenizer, saves it, and configures it to be loadable via AutoTokenizer.from_pretrained()."""
    print("1. Initializing tokenizer...")
    # Initialize your tokenizer (choose 256 or 512)
    tokenizer = AsciiOrdTokenizer(vocab_size_limit=vocab_size_limit)

    # 2. Save the standard tokenizer files (vocab.json, etc.)
    print(f"2. Saving to '{output_dir}'...")
    tokenizer.save_pretrained(output_dir)

    # 3. CRITICAL: Copy the Python source code to the output directory.
    # AutoTokenizer needs the actual file 'ascii_tokenizer.py' present to import the class.
    import bionemo.evo2.data.tokenizer

    src_file = bionemo.evo2.data.tokenizer.__file__
    dst_file = os.path.join(output_dir, "ascii_tokenizer.py")
    shutil.copy(src_file, dst_file)
    print(f"   - Copied {src_file} to {output_dir}")

    # 4. CRITICAL: Update tokenizer_config.json with 'auto_map' and custom args.
    config_path = os.path.join(output_dir, "tokenizer_config.json")

    with open(config_path, "r") as f:
        config = json.load(f)

    # Add the auto_map to tell AutoTokenizer where to find the class
    # Format: "AutoClass": ["module_name.ClassName", "fast_tokenizer_class_or_None"]
    config["auto_map"] = {"AutoTokenizer": ["ascii_tokenizer.AsciiOrdTokenizer", None]}

    # Explicitly save custom arguments so they are restored correctly
    # (PreTrainedTokenizer sometimes misses args not in the base config)
    config["vocab_size_limit"] = tokenizer.vocab_size_limit

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print("   - Updated tokenizer_config.json with auto_map")
    print("Done. Setup complete.")


def test_loading(model_path="ascii_tokenizer_model"):
    """Verifies that the tokenizer can be loaded using AutoTokenizer."""
    print(f"\n--- Testing AutoTokenizer.from_pretrained('{model_path}') ---")

    # Note: trust_remote_code=True is REQUIRED for custom classes
    loaded_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print(f"Successfully loaded class: {type(loaded_tokenizer)}")
    print(f"Vocab Size Limit: {loaded_tokenizer.vocab_size_limit}")

    text = "Hello World!"
    tokens = loaded_tokenizer.encode(text)
    print(f"Test Encode '{text}': {tokens}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True, help="Path to save the tokenizer")
    parser.add_argument(
        "--vocab-size-limit", type=int, choices=[256, 512], default=512, help="Vocabulary size limit, 256 or 512"
    )
    args = parser.parse_args()
    setup_and_save_tokenizer(args.output_dir, args.vocab_size_limit)
    test_loading(args.output_dir)
