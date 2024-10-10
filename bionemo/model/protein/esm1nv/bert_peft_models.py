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


from nemo.collections.nlp.modules.common.megatron.adapters.parallel_adapters import (
    AdapterName,
    LoraKQVAdapterConfig,
)
from nemo.core.classes.mixins import adapter_mixins
from nemo.utils import logging, model_utils
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from bionemo.model.protein.esm1nv.esm1nv_model import ESM2nvModel


try:
    from apex.transformer import tensor_parallel  # noqa: F401

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False

__all__ = ["MegatronBertPEFTModel", "BertLoRAModel"]


# NOTE: adapted from NeMo, may contains flags not relevant to bert
class MegatronBertPEFTModel(ESM2nvModel):  # ESM1nvModel
    """
    base class for all mixin based adapter models
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer)
        self.setup_complete = False
        self.base_keys = self.get_all_keys()
        self.init_peft_modules()
        self.adapter_keys = self.get_all_keys() - self.base_keys

    def init_peft_modules(self):
        """
        Randomly initialize the peft params and add them to the appropriate modules.
        """
        assert len(self.peft_name_keys) > 0, "peft_name_keys have not been set no PEFT modules will be added"
        assert len(self.name_key_to_cfg) > 0, "name_key_to_cfg has not been set no PEFT modules will be added"
        logging.info(f"Before adding PEFT params:\n{self.summarize()}")
        for _, module in self.named_modules():
            if isinstance(module, adapter_mixins.AdapterModuleMixin):
                for peft_key in self.peft_name_keys:
                    peft_cfg = self.name_key_to_cfg[peft_key]
                    if model_utils.import_class_by_path(peft_cfg._target_) in module.get_accepted_adapter_types():
                        module.add_adapter(
                            name=peft_key,
                            cfg=peft_cfg,
                        )
        logging.info(f"After adding PEFT params:\n{self.summarize()}")
        return True

    def get_all_keys(
        self,
    ):
        """
        Returns all the keys in the model
        """
        k = [n for n, p in self.named_parameters()]
        return set(k)

    def get_peft_state_dict(
        self,
    ):
        """
        Gets the keys associated with the adapters only.
        """
        state_dict = self.model.state_dict(prefix="model.")
        peft_state_dict = {}
        for k in self.adapter_keys:
            peft_state_dict[k] = state_dict[k]
        return peft_state_dict

    def state_dict(self, destination=None, prefix=None, keep_vars=False):
        if self.setup_complete:
            # Once setup is complete we no longer need to track the frozen part of the model. Only there adapter state dict keeps changing so state_dict only track these.
            return self.get_peft_state_dict()
        else:
            # we want all the params with the same keys as calling self.state_dict()
            # but we can't call self.state_dict() here as it would be a recursive call.
            # so we call self.model.state_dict(prefix="model.") which will return all the keys and params same as calling self.state_dict()
            return self.model.state_dict(prefix="model.")

    def load_state_dict(self, state_dict, strict: bool = True):
        if self.setup_complete:
            # at this stage only PEFT params will appear in the state_dict arg
            # so we only update those while the rest of the model is frozen.
            # setting strict=False will ignore the missing keys (which are not being updated anyway)
            # explicitly check if state_dict.keys matches all the expected self.adapter_keys since we don't have the
            # safety in strict=True anymore.
            assert set(state_dict.keys()) == self.adapter_keys
            super().load_state_dict(state_dict, strict=False)
        else:
            super().load_state_dict(state_dict, strict=True)

    def setup_optimizer_param_groups(self):
        """
        ModelPT override. Optimizer will get self._optimizer_param_groups.
        Makes two optimizer param groups, one for the frozen model params
        and one for the prompt-table/prompt-encoder params. The learning
        rate for the frozen model's params will always be zero effectively
        freezing the model's params but still allowing for the needed gradients
        to be passed around in pipeline parallel models. The prompt-encoder
        and/or prompt table will use the learning rate set by the user.
        """

        self.freeze()  # Freeze the entire model
        opt_params = []
        for _, module in self.named_modules():
            if isinstance(module, adapter_mixins.AdapterModuleMixin) and module.is_adapter_available():
                module.set_enabled_adapters(enabled=True)
                module.unfreeze_enabled_adapters()  # selectively unfreeze the adapter modules.
                opt_params += list(module.parameters())

        self._optimizer_param_groups = ({"params": opt_params},)
        logging.info(f"Optimizer groups set:\n{self.summarize()}")
        self.setup_complete = True  # tracks that setup is already complete


class BertLoRAModel(MegatronBertPEFTModel):
    """
    BertLoRAModel is a model that combines a base model (ESM2nvModel) with a low-rank adapters.
    The lora adapters will be added in `nemo/collections/nlp/modules/common/megatron/attention.py`
    The implementation is based on Hu et al. nemo/collections/nlp/modules/common/megatron/attention.py
    Note: NeMo's implementation of LoRa ignores the constant scaling factor `alpha` that was introduced
    in the original paper by Hu et al. This is because adjusting the learning rate value can
    effectively compensate for the scaling.

    A single low-rank feedfowrad layer is used in parallel with the KQV projection layer.
    TODO: Add support to also include an option to adda low-rank adapter in the output projection layer.
    """

    def __init__(
        self,
        cfg: DictConfig,
        trainer: Trainer,
    ):
        self.peft_name_keys = [
            AdapterName.LORA_KQV_ADAPTER,
        ]
        lora_cfg = cfg.peft.lora_tuning
        if cfg.get("kv_channels", None) is None:
            assert (
                cfg.hidden_size % cfg.num_attention_heads == 0
            ), "hidden_size must be divisible by num_attention_heads if kv_channels is None"
            kv_channels = cfg.hidden_size // cfg.num_attention_heads
        else:
            kv_channels = cfg.kv_channels
        projection_size = kv_channels * cfg.num_attention_heads

        adapter_cfg = LoraKQVAdapterConfig(
            in_features=cfg.hidden_size,
            out_features=3 * projection_size,
            dim=lora_cfg.adapter_dim,
            norm_position="none",
            norm_type="none",
            activation="identity",
            column_init_method=lora_cfg.get("column_init_method", "normal"),
            row_init_method=lora_cfg.get("row_init_method", "zero"),
            gather_output=False,
            dropout=lora_cfg.adapter_dropout,
        )

        self.name_key_to_cfg = {}
        for k in self.peft_name_keys:
            self.name_key_to_cfg[k] = adapter_cfg

        super().__init__(cfg, trainer)
