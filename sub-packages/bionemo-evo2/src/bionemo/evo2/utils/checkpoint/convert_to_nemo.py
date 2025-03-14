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


import argparse
import logging
from pathlib import Path

from nemo.collections.llm.gpt.model.hyena import (
    HYENA_MODEL_OPTIONS,
    HuggingFaceSavannaHyenaImporter,
    HyenaConfig,
    HyenaModel,
    PyTorchHyenaImporter,
)
from nemo.lightning import io, teardown


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the Evo2 un-sharded (MP1) model checkpoint file, or a Hugging Face model name. Any model "
        "from the Savanna Evo2 family is supported such as 'hf://arcinstitute/savanna_evo2_1b_base'.",
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory path for the converted model.")
    parser.add_argument(
        "--model-size",
        type=str,
        choices=sorted(HYENA_MODEL_OPTIONS.keys()),
        default="1b",
        help="Model architecture to use, choose between 1b, 7b, 40b, or test (a sub-model of 4 layers, "
        "less than 1B parameters). '*_arc_longcontext' models have GLU / FFN dimensions that support 1M "
        "context length when trained with TP>>8.",
    )
    parser.add_argument(
        "--strip-optimizer",
        action="store_true",
        help="Strip the optimizer state from the model checkpoint, this works on nemo2 format checkpoints.",
    )
    return parser.parse_args()


@io.model_importer(HyenaModel, "pytorch")
class HyenaOptimizerRemover(io.ModelConnector["HyenaModel", HyenaModel]):
    """Removes the optimizer state from a nemo2 format model checkpoint."""

    def __new__(cls, path: str, model_config=None):
        """Creates a new importer instance.

        Args:
            path: Path to the PyTorch model
            model_config: Optional model configuration

        Returns:
            PyTorchHyenaImporter instance
        """
        instance = super().__new__(cls, path)
        instance.model_config = model_config
        return instance

    def init(self) -> HyenaModel:
        """Initializes a new HyenaModel instance.

        Returns:
            HyenaModel: Initialized model
        """
        return HyenaModel(self.config, tokenizer=self.tokenizer)

    def get_source_model(self):
        """Returns the source model."""
        model, _ = self.nemo_load(self)
        return model

    def apply(self, output_path: Path, checkpoint_format: str = "torch_dist") -> Path:
        """Applies the model conversion from PyTorch to NeMo format.

        Args:
            output_path: Path to save the converted model
            checkpoint_format: Format for saving checkpoints

        Returns:
            Path: Path to the saved NeMo model
        """
        source = self.get_source_model()

        target = self.init()
        trainer = self.nemo_setup(target, ckpt_async_save=False, save_ckpt_format=checkpoint_format)
        source.to(self.config.params_dtype)
        target.to(self.config.params_dtype)
        self.convert_state(source, target)
        self.nemo_save(output_path, trainer)

        logging.info(f"Converted Hyena model to Nemo, model saved to {output_path}")

        teardown(trainer, target)
        del trainer, target

        return output_path

    def convert_state(self, source, target):
        """Converts the state dictionary from source format to target format.

        Args:
            source: Source model state
            target: Target model

        Returns:
            Result of applying state transforms
        """
        mapping = {k: k for k in source.module.state_dict().keys()}
        return io.apply_transforms(
            source,
            target,
            mapping=mapping,
        )

    @property
    def tokenizer(self):
        """Gets the tokenizer for the model.

        Returns:
            Tokenizer instance
        """
        from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer

        tokenizer = get_nmt_tokenizer(
            library=self.model_config.tokenizer_library,
        )

        return tokenizer

    @property
    def config(self) -> HyenaConfig:
        """Gets the model configuration.

        Returns:
            HyenaConfig: Model configuration
        """
        return self.model_config


def main():
    """Convert a PyTorch Evo2 model checkpoint to a NeMo model checkpoint."""
    args = parse_args()

    evo2_config = HYENA_MODEL_OPTIONS[args.model_size]()
    if args.strip_optimizer:
        importer = HyenaOptimizerRemover(args.model_path, model_config=evo2_config)
        assert not args.model_path.startswith("hf://"), "Strip optimizer only works on local nemo2 format checkpoints."
    elif args.model_path.startswith("hf://"):
        importer = HuggingFaceSavannaHyenaImporter(args.model_path.lstrip("hf://"), model_config=evo2_config)
    else:
        importer = PyTorchHyenaImporter(args.model_path, model_config=evo2_config)
    importer.apply(args.output_dir)


if __name__ == "__main__":
    main()
