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


import argparse
import logging
from pathlib import Path
from typing import Dict

import nemo.lightning as nl
import torch
from nemo.collections.llm.gpt.model.hyena import HyenaConfig, HyenaModel
from nemo.collections.llm.inference.base import _setup_trainer_and_restore_model
from nemo.lightning import io
from nemo.lightning.ckpt_utils import ckpt_to_context_subdir


# =================================================================================================
# NOTE: This version of the script does not require Vortex model/config class definitions.
# It builds and saves the state dictionary directly.
# =================================================================================================
logger = logging.getLogger(__name__)


class TransformFns:
    """Container for custom tensor transformation functions."""

    @staticmethod
    def calculate_log_poles(p: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        """Calculates log_poles for the Hyena 'H' layer from bionemo's p and gamma."""
        logp = -torch.exp(p)
        logp = (logp * torch.exp(gamma))[..., None]
        return logp

    @staticmethod
    def calculate_hyena_filter_h(h: torch.Tensor, decay: torch.Tensor) -> torch.Tensor:
        """Calculates the filter `h` for the Hyena 'D' layer."""
        # bionemo checkpoints should already have tensors truncated to the correct length.
        return (h * decay).unsqueeze(1)


class BioNeMoToVortexConverter:
    """Converter class for transforming a BioNeMo Hyena model's state_dict to the Vortex format."""

    def __init__(self, path: str, **kwargs):
        """Initializes the converter.

        Args:
            path: Path to the source .nemo checkpoint file.
            **kwargs: Additional config attributes to set on the model.config if you want to override the defaults.
        """
        self.source_path = Path(path)
        if not self.source_path.exists():
            raise FileNotFoundError(f"Source BioNeMo file not found at: {path}")

        # Load the source model config to access properties like layer structure
        logging.info("Loading source model configuration...")
        # this doesn't have weights or anything, just the config
        #  Hyena subclasses GPTModel
        self.model: HyenaModel = io.load_context(path=ckpt_to_context_subdir(path), subpath="model")
        self.source_config: HyenaConfig = self.model.config
        for key, value in kwargs.items():
            if hasattr(self.model.config, key):
                setattr(self.model.config, key, value)
            else:
                logging.warning(
                    f"Config attribute {key} not found in model.config, ignoring in setup_model_and_tokenizer"
                )
        logging.info("Source configuration loaded successfully.")

    def get_source_state_dict(self) -> Dict[str, torch.Tensor]:
        """Loads the state dictionary from the source .nemo file."""
        logging.info(f"Loading source model state_dict from {self.source_path}...")
        # Create PTL trainer.
        trainer = nl.Trainer(
            accelerator="gpu",
            devices=1,
            strategy=nl.MegatronStrategy(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                context_parallel_size=1,
                pipeline_dtype=torch.bfloat16,
                ckpt_load_optimizer=False,  # Needs to be false for a normal model checkpoint.
                ckpt_save_optimizer=False,
                ckpt_async_save=False,
                save_ckpt_format="torch_dist",
                ckpt_load_strictness="log_all",
            ),
            log_every_n_steps=1,
            limit_val_batches=10,
            num_sanity_val_steps=0,
            plugins=nl.MegatronMixedPrecision(
                precision="bf16-mixed",
                params_dtype=torch.bfloat16,
            ),
        )
        _setup_trainer_and_restore_model(path=self.source_path, trainer=trainer, model=self.model)
        state_dict = self.model.module.state_dict()
        del self.model.module
        logging.info("Source state_dict loaded successfully.")
        return state_dict

    def apply(self, output_path: Path) -> Path:
        """Applies the model conversion from BioNeMo to Vortex format and saves the resulting state dictionary.

        Args:
            output_path: Path to save the converted Vortex state_dict (.pt).

        Returns:
            Path to the saved Vortex state_dict file.
        """
        source_state_dict = self.get_source_state_dict()

        target_state_dict = self.convert_state_dict(source_state_dict)

        # Save the converted state dictionary
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(target_state_dict, str(output_path))

        logging.info(f"✅ Conversion complete. Vortex state_dict saved to: {output_path}")
        return output_path

    def convert_state_dict(self, source_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Converts the state dictionary from bionemo to vortex format by renaming and transforming tensors.

        Args:
            source_dict: The source state dictionary from the bionemo model.

        Returns:
            The converted state dictionary in vortex format.
        """
        logging.info("Starting state_dict conversion...")
        target_dict = {}

        # 1. Global mappings (embedding, final norm)
        if "embedding.word_embeddings.weight" in source_dict:
            target_dict["embedding_layer.weight"] = source_dict["embedding.word_embeddings.weight"]
            target_dict["unembed.weight"] = source_dict["embedding.word_embeddings.weight"].clone()
        if "decoder.final_norm.weight" in source_dict:
            target_dict["norm.scale"] = source_dict["decoder.final_norm.weight"]

        # 2. Per-layer mappings and transformations
        num_layers = self.source_config.num_layers
        for i in range(num_layers):
            symbol = self.source_config.hybrid_override_pattern[i]

            # --- Norms ---
            bionemo_pre_norm_key = (
                f"decoder.layers.{i}.input_layernorm.weight" if symbol == "*" else f"decoder.layers.{i}.norm.weight"
            )
            if bionemo_pre_norm_key in source_dict:
                target_dict[f"blocks.{i}.pre_norm.scale"] = source_dict[bionemo_pre_norm_key]
            if f"decoder.layers.{i}.pre_mlp_layernorm.weight" in source_dict:
                target_dict[f"blocks.{i}.post_norm.scale"] = source_dict[
                    f"decoder.layers.{i}.pre_mlp_layernorm.weight"
                ]

            # --- MLP ---
            fc1_key = f"decoder.layers.{i}.mlp.linear_fc1.weight"
            if fc1_key in source_dict:
                gate_proj, up_proj = torch.chunk(source_dict[fc1_key], 2, dim=0)
                target_dict[f"blocks.{i}.mlp.l1.weight"] = gate_proj
                target_dict[f"blocks.{i}.mlp.l2.weight"] = up_proj

            fc2_key = f"decoder.layers.{i}.mlp.linear_fc2.weight"
            if fc2_key in source_dict:
                target_dict[f"blocks.{i}.mlp.l3.weight"] = source_dict[fc2_key]

            # --- Mixer (Attention or Hyena) ---
            if symbol == "*":  # Attention Layer
                target_dict[f"blocks.{i}.inner_mha_cls.Wqkv.weight"] = source_dict[
                    f"decoder.layers.{i}.self_attention.linear_qkv.weight"
                ]
                target_dict[f"blocks.{i}.inner_mha_cls.out_proj.weight"] = source_dict[
                    f"decoder.layers.{i}.self_attention.linear_proj.weight"
                ]
                target_dict[f"blocks.{i}.inner_mha_cls.out_proj.bias"] = source_dict[
                    f"decoder.layers.{i}.self_attention.linear_proj.bias"
                ]
            else:  # Hyena Layers ('S', 'D', 'H')
                target_dict[f"blocks.{i}.projections.weight"] = source_dict[
                    f"decoder.layers.{i}.mixer.dense_projection.weight"
                ]
                target_dict[f"blocks.{i}.out_filter_dense.weight"] = source_dict[
                    f"decoder.layers.{i}.mixer.dense.weight"
                ]
                target_dict[f"blocks.{i}.out_filter_dense.bias"] = source_dict[f"decoder.layers.{i}.mixer.dense.bias"]
                target_dict[f"blocks.{i}.filter.short_filter_weight"] = source_dict[
                    f"decoder.layers.{i}.mixer.hyena_proj_conv.short_conv_weight"
                ].unsqueeze(1)

                if symbol == "S":  # Short Convolution Hyena
                    target_dict[f"blocks.{i}.filter.h"] = source_dict[
                        f"decoder.layers.{i}.mixer.mixer.short_conv.short_conv_weight"
                    ]
                elif symbol == "D":  # Medium (Diagonal) Hyena
                    target_dict[f"blocks.{i}.filter.D"] = source_dict[f"decoder.layers.{i}.mixer.mixer.conv_bias"]
                    h = source_dict[f"decoder.layers.{i}.mixer.mixer.filter.h"]
                    decay = source_dict[f"decoder.layers.{i}.mixer.mixer.filter.decay"]
                    target_dict[f"blocks.{i}.filter.h"] = TransformFns.calculate_hyena_filter_h(h, decay)
                elif symbol == "H":  # Long (State-Space) Hyena
                    target_dict[f"blocks.{i}.filter.D"] = source_dict[f"decoder.layers.{i}.mixer.mixer.conv_bias"]
                    target_dict[f"blocks.{i}.filter.residues"] = source_dict[
                        f"decoder.layers.{i}.mixer.mixer.filter.R"
                    ]
                    p = source_dict[f"decoder.layers.{i}.mixer.mixer.filter.p"]
                    gamma = source_dict[f"decoder.layers.{i}.mixer.mixer.filter.gamma"]
                    target_dict[f"blocks.{i}.filter.log_poles"] = TransformFns.calculate_log_poles(p, gamma)

        logging.info(
            f"Conversion logic complete. Processed {len(target_dict.keys())} tensors for the target state_dict."
        )
        return target_dict


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to the BioNeMo checkpoint file. Can download standard ones "
        "with `download_bionemo_data` and see `download_bionemo_data --list-resources | grep evo2` for options.",
    )
    parser.add_argument(
        "--output-ckpt", type=Path, required=True, help="Path to save the converted Vortex state_dict (.pt)."
    )
    return parser.parse_args()


if __name__ == "__main__":
    # ==============================================================================
    # Example Usage
    # ==============================================================================
    # Configure logging
    args = parse_args()
    logger.setLevel(logging.INFO)
    logger.info("Starting conversion from BioNeMo to Vortex format...")

    # Initialize converter
    converter = BioNeMoToVortexConverter(args.model_path)

    # Apply conversion
    converter.apply(args.output_ckpt)
