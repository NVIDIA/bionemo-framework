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

import tarfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    Any,
    Dict,
    Generic,
    Type,
)

import torch
import yaml
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.lightning import io, teardown

from bionemo.core.utils.dtypes import get_autocast_dtype
from bionemo.llm.model.biobert.lightning import BioBertLightningModule
from bionemo.llm.model.biobert.model import BioBertGenericConfig, MegatronBioBertModelT
from bionemo.llm.utils.weight_utils import nemo1_to_nemo2_biobert_key_mapping


class GenericBioBertNeMo1LightningModuleConnector(
    io.ModelConnector[Dict[str, torch.Tensor], BioBertLightningModule], Generic[MegatronBioBertModelT], ABC
):
    """A generic ModuleConnector for going between nemo1 and nemo2 checkpoints of BERT based models.

    Provide a single argument to input that is the path to the nemo1 checkpoint. Note that io.ModelConnector inherits
        from `pathlib.Path`. Call `object.apply(nemo2_output_path)` to convert the checkpoint pointed to by the
        input path to the output path.
    """

    @abstractmethod
    def get_config_class(self) -> Type[BioBertGenericConfig[MegatronBioBertModelT]]:
        """Return the class of the config so that it's easier to subclass."""
        raise NotImplementedError("Implement me")

    def init(self) -> BioBertLightningModule:
        """Initialize the lightning module (no model initialized in it yet)."""
        return BioBertLightningModule(
            self.config,
            self.tokenizer,
        )

    def apply(self, output_path: Path) -> Path:
        """Save this nemo1 checkpoint to the desired output path in nemo2 format."""
        nemo1_path = str(self)  # self is a Path object
        with tarfile.open(nemo1_path, "r") as old_ckpt:
            ckpt_file = old_ckpt.extractfile("./model_weights.ckpt")
            old_weights = torch.load(ckpt_file)
        target = self.init()
        trainer = self.nemo_setup(target)
        target.trainer = trainer
        self.convert_state(old_weights, target)
        self.nemo_save(output_path, trainer)

        print(f"Converted NeMo1, model at {self} saved to {output_path}")

        teardown(trainer, target)
        del trainer, target

        return output_path

    @staticmethod
    def is_te_mapping(model: BioBertLightningModule) -> bool:
        """Check for TE layers, for now infer this from the config."""
        # TODO come up with a more robust way of determining this.
        return "transformer_engine" in model.config.biobert_spec_option.value

    def convert_state(self, source: Dict[str, torch.Tensor], target: BioBertLightningModule) -> BioBertLightningModule:
        """Convert the input state_dict keys from nemo1 biobert to nemo2 biobert."""
        te_mapping = self.is_te_mapping(target)  # check for TE layers.
        new_state_dict_from_old = {}
        for k, v in source.items():
            new_key = nemo1_to_nemo2_biobert_key_mapping(k, new_model_prefix="", te_mapping=te_mapping)
            new_state_dict_from_old[new_key] = v
        for k, v in new_state_dict_from_old.items():
            if v.device == torch.device("meta"):
                raise ValueError(v)
        result = target.module.load_state_dict(new_state_dict_from_old, strict=not te_mapping, assign=True)
        if te_mapping:
            # Custom alternative to strict here, allow _extra_state keys but otherwise
            #  there are problems
            if result.unexpected_keys != [] or (
                result.missing_keys != [] and any("._extra_state" not in k for k in result.missing_keys)
            ):
                raise ValueError(f"There are mismatches other than _extra_state in loading: {result}")
        meta_tensors_keys = [
            k
            for (k, t) in target.module.state_dict().items()
            if isinstance(t, torch.Tensor) and t.device.type == "meta"
        ]
        if len(meta_tensors_keys) != 0:
            raise ValueError(
                f"The following tensors were left on device='meta', so weights did not get applied: {meta_tensors_keys}"
            )
        return target

    @property
    @abstractmethod
    def tokenizer(self) -> "AutoTokenizer":
        """Generic method to return a tokenizer, override this for your implemented nemo1 to nemo2 biobert converter."""
        raise NotImplementedError("Implement this method")

    def get_nemo1_config(self) -> Dict[str, Any]:
        """Return the nemo1 config from the checkpoint."""
        # First read from nemo file and get config settings
        nemo1_path = str(self)  # self inherits from PosixPath
        with tarfile.open(nemo1_path, "r") as old_ckpt:
            config_yaml = old_ckpt.extractfile("./model_config.yaml")
            if config_yaml is None:
                raise ValueError("Config cannot be None in nemo1 checkpoint")
            return yaml.safe_load(config_yaml.read())

    def get_config_overrides(self, autocast_dtype: torch.dtype) -> Dict[str, Any]:
        """Override this method in your child class if you need alternative overrides."""
        overrides = {
            "params_dtype": autocast_dtype,
            "pipeline_dtype": autocast_dtype,
            "autocast_dtype": autocast_dtype,
            "attention_dropout": 0.1,
            "fp32_residual_connection": False,
            "bias_activation_fusion": True,
            "bias_dropout_fusion": True,
            "apply_query_key_layer_scaling": False,
            "share_embeddings_and_output_weights": True,
            "fp16": autocast_dtype == torch.float16,
            "bf16": autocast_dtype == torch.bfloat16,
        }
        return overrides

    @property
    def config(self) -> BioBertGenericConfig[MegatronBioBertModelT]:
        """Convert and return the nemo2 config from the nemo1 config."""
        nemo1_settings = self.get_nemo1_config()
        cfg_class = self.get_config_class()
        autocast_dtype = get_autocast_dtype(nemo1_settings["precision"])
        overrides = self.get_config_overrides(autocast_dtype)
        output = cfg_class(
            **overrides,
            **{k: v for k, v in nemo1_settings.items() if k in dir(cfg_class) and k not in overrides},
        )
        return output
