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
        """Convert a BioNeMo Hyena/GPT checkpoint to Vortex format.

        Note: All FP8 `*_extra_state` dicts for projection layers are copied verbatim.
        """
        logging.info("Starting state-dict conversion …")

        # 0. strip possible `model.` prefix
        if any(k.startswith("model.") for k in source_dict):
            source_dict = {k[6:]: v for k, v in source_dict.items()}

        tgt: Dict[str, torch.Tensor] = {}

        # 1. globals ----------------------------------------------------------------
        tgt["embedding_layer.weight"] = source_dict["embedding.word_embeddings.weight"]
        tgt["norm.scale"] = source_dict["decoder.final_norm.weight"]

        # helper to make RoPE if absent
        head_dim = self.source_config.hidden_size // self.source_config.num_attention_heads

        def synth_inv_freq(base: float = 10000, dtype=torch.float32):
            return 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=dtype) / head_dim))

        # 2. per-layer --------------------------------------------------------------
        cfg, pat = self.source_config, self.source_config.hybrid_override_pattern
        for i in range(cfg.num_layers):
            sym, lp = pat[i], f"decoder.layers.{i}"

            # ------ norms ------
            for cand in (
                f"{lp}.input_layernorm.weight",
                f"{lp}.mixer.dense_projection.layer_norm_weight",
                f"{lp}.norm.weight",
                f"{lp}.self_attention.linear_qkv.layer_norm_weight",
            ):
                if cand in source_dict:
                    tgt[f"blocks.{i}.pre_norm.scale"] = source_dict[cand]
                    break

            for cand in (
                f"{lp}.pre_mlp_layernorm.weight",
                f"{lp}.mlp.linear_fc1.layer_norm_weight",
            ):
                if cand in source_dict:
                    tgt[f"blocks.{i}.post_norm.scale"] = source_dict[cand]
                    break

            # ------ MLP ------
            gate, up = torch.chunk(source_dict[f"{lp}.mlp.linear_fc1.weight"], 2, dim=0)
            tgt[f"blocks.{i}.mlp.l1.weight"] = gate
            tgt[f"blocks.{i}.mlp.l2.weight"] = up
            tgt[f"blocks.{i}.mlp.l3.weight"] = source_dict[f"{lp}.mlp.linear_fc2.weight"]

            # ------ mixers ------
            if sym == "*":  # Attention
                sa = f"{lp}.self_attention"
                tgt[f"blocks.{i}.inner_mha_cls.Wqkv.weight"] = source_dict[f"{sa}.linear_qkv.weight"]
                tgt[f"blocks.{i}.inner_mha_cls.out_proj.weight"] = source_dict[f"{sa}.linear_proj.weight"]
                tgt[f"blocks.{i}.inner_mha_cls.out_proj.bias"] = source_dict[f"{sa}.linear_proj.bias"]

                rope_key = f"{sa}.rotary_emb.inv_freq"
                tgt[f"blocks.{i}.inner_mha_cls.rotary_emb.inv_freq"] = (
                    source_dict[rope_key]
                    if rope_key in source_dict
                    else synth_inv_freq(base=self.source_config.rotary_base)
                )

            else:  # Hyena (S/D/H)
                mix = f"{lp}.mixer"

                # projection weight + its original FP8 extra_state
                tgt[f"blocks.{i}.projections.weight"] = source_dict[f"{mix}.dense_projection.weight"]
                if f"{mix}.dense_projection._extra_state" in source_dict:
                    tgt[f"blocks.{i}.projections._extra_state"] = source_dict[f"{mix}.dense_projection._extra_state"]

                # dense after filter
                tgt[f"blocks.{i}.out_filter_dense.weight"] = source_dict[f"{mix}.dense.weight"]
                tgt[f"blocks.{i}.out_filter_dense.bias"] = source_dict[f"{mix}.dense.bias"]

                # short pre-filter
                tgt[f"blocks.{i}.filter.short_filter_weight"] = source_dict[
                    f"{mix}.hyena_proj_conv.short_conv_weight"
                ].unsqueeze(1)

                if sym == "S":  # short-conv only
                    tgt[f"blocks.{i}.filter.h"] = source_dict[f"{mix}.mixer.short_conv.short_conv_weight"]

                elif sym == "D":  # medium
                    tgt[f"blocks.{i}.filter.D"] = source_dict[f"{mix}.mixer.conv_bias"]
                    h, decay = source_dict[f"{mix}.mixer.filter.h"], source_dict[f"{mix}.mixer.filter.decay"]
                    tgt[f"blocks.{i}.filter.h"] = TransformFns.calculate_hyena_filter_h(h, decay)

                elif sym == "H":  # long
                    tgt[f"blocks.{i}.filter.D"] = source_dict[f"{mix}.mixer.conv_bias"]
                    tgt[f"blocks.{i}.filter.residues"] = source_dict[f"{mix}.mixer.filter.R"]
                    tgt[f"blocks.{i}.filter.log_poles"] = TransformFns.calculate_log_poles(
                        source_dict[f"{mix}.mixer.filter.p"], source_dict[f"{mix}.mixer.filter.gamma"]
                    )

                else:
                    raise ValueError(f"Unexpected symbol {sym!r} at layer {i}")

        # 3. drop BioNeMo extras we do not want --------------------------------------
        tgt = {k: v for k, v in tgt.items() if not k.endswith("mixer.dense._extra_state") and k != "unembed.weight"}
        # for k, v in tgt.items():
        #     if hasattr(v, "set_extra_state") or hasattr(v, '_extra_state') or not isinstance(v, torch.Tensor):
        #         raise ValueError(f"Extra state found in {k}")

        logging.info(f"Done - {len(tgt):,} tensors ready for Vortex.")
        return tgt


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert BioNeMo checkpoint to Vortex format",
        epilog="Example: evo2_convert_nemo2_to_vortex --model-path $(download_bionemo_data evo2/1b-8k-bf16:1.0) --output-ckpt evo2_1b_8k_bf16_1_0_vortex.pt",
    )
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


def main():
    """Main function to convert a BioNeMo checkpoint to Vortex format."""
    args = parse_args()
    logger.setLevel(logging.INFO)
    logger.info("Starting conversion from BioNeMo to Vortex format...")

    # Initialize converter
    converter = BioNeMoToVortexConverter(args.model_path)

    # Apply conversion
    converter.apply(args.output_ckpt)


if __name__ == "__main__":
    main()
