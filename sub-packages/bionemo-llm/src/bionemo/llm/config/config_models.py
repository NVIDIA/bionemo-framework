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


from abc import ABC, abstractmethod
from dataclasses import field
from typing import Any, Callable, Dict, Generic, List, Literal, Optional, Type, TypeVar

import pytorch_lightning as pl
import torch
from pydantic import BaseModel, ValidationError, field_serializer, field_validator, model_validator
from torch.nn import functional as F

from bionemo.core.utils import dtypes
from bionemo.llm.model.biobert.model import BioBertGenericConfig
from bionemo.llm.model.biobert.transformer_specs import BiobertSpecOption
from bionemo.llm.utils.logger_utils import WandbConfig


ModelConfigT = TypeVar("ModelConfigT", bound=BioBertGenericConfig)
DataModuleT = TypeVar("DataModuleT", bound=pl.LightningDataModule)

# To register a custom activation function, add it to this dictionary to pass validation and allow serialization.
CUSTOM_ACTIVATION_FNS: Dict[str, Callable[[torch.Tensor, Any], torch.Tensor]] = {}

# DO NOT use keys that already exist in torch.nn.functional, as the torch.nn.functional functions are selected first.
for key in CUSTOM_ACTIVATION_FNS:
    assert key not in dir(torch.nn.functional), f"Key {key} already exists in torch.nn.functional"

# It does not matter if values are duplicated as the key=>value mapping still does the right thing. Repeat values should be considered aliases.
REVERSE_CUSTOM_ACTIVATION_FNS: Dict[Callable[[torch.Tensor, Any], torch.Tensor], str] = {
    v: k for k, v in CUSTOM_ACTIVATION_FNS.items()
}


class DataConfig(BaseModel, Generic[DataModuleT], ABC):
    """Base class for all data configurations.

    This class is used to define the interface for all data configurations. It is used to define the data module that
    will be used in the training loop.
    """

    micro_batch_size: int = 8
    result_dir: str = "./results"
    num_dataset_workers: int = 0
    seq_length: int = 128

    @abstractmethod
    def construct_data_module(self, global_batch_size: int) -> DataModuleT:
        """Construct the data module from the configuration. Cannot be defined generically."""
        ...

    def model_validator(self, global_cfg: "MainConfig") -> "MainConfig":
        ''' Use custom implementation of this method to define the things inside global_config. 

        The following expression will always be true:

        global_cfg.data_config == self 
        '''
        return global_cfg


class ExposedModelConfig(BaseModel, Generic[ModelConfigT], ABC):
    """BioNeMo model configuration class, wraps TransformerConfig and friends.

    This class is used to define the interface for all model configurations. It is **Exposed** to guard against ill-typed
    or poorly defined fields in the underlying configuration objects. `ModelConfigT` declares the associated type of the
    underlying config (most commonly a BioBertGenericConfig, but could also be a TransformerConfig or something similar).
    Children should try to expose the minimal set of fields necessary for the user to configure the model while keeping
    the more esoteric configuration private to the underlying ModelConfigT.

    """

    # Restores weights from a pretrained checkpoint
    initial_ckpt_path: Optional[str] = None
    # Does not attempt to load keys with these prefixes (useful if you attached extra parameters and still want to load a set of weights)
    initial_ckpt_skip_keys_with_these_prefixes: List[str] = field(default_factory=list)

    # TODO validator on num_attention_heads, ffn_hidden_size, and hidden_size as these have knowable constraints.


    # Pydantic stuff to allow arbitrary types + validators + serializers
    class Config:
        arbitrary_types_allowed = True

    """ Use this class to hide fields that are not serializable by Pydantic that we do not want to expose. """

    def model_class(self) -> Type[ModelConfigT]:
        # How did this all work yesterday even?
        # so we cant do it this way because we are kinda losing the magic of generics.
        #  ideally _the generics_ have all the methods we want implemented on them already.
        raise NotImplementedError

    def model_validator(self, global_cfg: "MainConfig") -> "MainConfig":
        ''' Use custom implementation of this method to define the things inside global_config. 

        The following expression will always be true:

        global_cfg.bionemo_model_config == self 
        '''
        return global_cfg

    def exposed_to_internal_bionemo_model_config(self) -> ModelConfigT:
        """Converts the exposed dataclass to the underlying Transformer config.

        The underlying ModelConfigT may both be incomplete and unserializable. We use this transformation as a way to
        hide fields that are either not serializable by Pydantic or that we do not want to expose.

        This is a good candidate for refactoring.
        """
        cls: Type[ModelConfigT] = self.model_class()
        model_dict = {}
        for attr in self.model_fields:
            if attr not in model_dict and attr in cls.__dataclass_fields__:
                model_dict[attr] = getattr(self, attr)
        # Now set fp16 and bf16 based on the precision for the underlying TransformerConfig=>ParallelConfig
        #   the only constraint is that both must not be true.
        model_dict["bf16"] = self.pipeline_dtype == dtypes.precision_to_dtype["bf16-mixed"]
        model_dict["fp16"] = self.pipeline_dtype == dtypes.precision_to_dtype["16-mixed"]
        result = cls(**model_dict)

        return result

    # NOTE: See PrecisionTypes for a list of valid literals that may be deserialized.
    params_dtype: torch.dtype
    pipeline_dtype: torch.dtype
    autocast_dtype: torch.dtype

    num_layers: int = 6
    hidden_size: int = 256
    ffn_hidden_size: int = 512
    num_attention_heads: int = 4
    seq_length: int = 512
    fp32_residual_connection: bool = False
    hidden_dropout: float = 0.02
    init_method_std: float = 0.02
    kv_channels: Optional[int] = None
    apply_query_key_layer_scaling: bool = False
    make_vocab_size_divisible_by: int = 128
    masked_softmax_fusion: bool = True
    fp16_lm_cross_entropy: bool = False
    gradient_accumulation_fusion: bool = False
    layernorm_zero_centered_gamma: bool = False
    layernorm_epsilon: float = 1.0e-12
    activation_func: Callable[[torch.Tensor, Any], torch.Tensor] = F.gelu
    qk_layernorm: bool = False
    apply_residual_connection_post_layernorm: bool = False
    bias_activation_fusion: bool = True
    bias_dropout_fusion: bool = True
    get_attention_mask_from_fusion: bool = False
    attention_dropout: float = 0.1
    share_embeddings_and_output_weights: bool = True
    enable_autocast: bool = False
    nemo1_ckpt_path: Optional[str] = None
    biobert_spec_option: BiobertSpecOption = BiobertSpecOption.bert_layer_with_transformer_engine_spec

    @field_validator("activation_func", mode="before")
    @classmethod
    def validate_activation_func(cls, activation_func: str) -> Callable:
        """Validates the activation function, assumes this function exists in torch.nn.functional. For custom
        activation functions, use the CUSTOM_ACTIVATION_FUNCTIONS dictionary in the module.

        This method validates the provided activation function string and returns
        a callable function based on the validation context using the provided validator in the base class.

        Args:
            activation_func (str): The activation function to be validated.
            context (ValidationInfo): The context for validation.

        Returns:
            Callable: A callable function after validation.

        See Also:
            CUSTOM_ACTIVATION_FNS
        """
        func = getattr(torch.nn.functional, activation_func.lower(), None)
        if func is None and activation_func in CUSTOM_ACTIVATION_FNS:
            func = CUSTOM_ACTIVATION_FNS[activation_func]
            return func
        elif func is None:
            raise ValidationError(
                f"activation_func must be a valid function in `torch.nn.functional`, got {activation_func=}"
            )
        else:
            return func

    @field_serializer("activation_func")
    def serialize_activation_func(self, v: Callable[[torch.Tensor, Any], torch.Tensor]) -> str:
        func_name = v.__name__
        func = getattr(torch.nn.functional, func_name, None)
        if func is not None:
            return func_name
        elif func in REVERSE_CUSTOM_ACTIVATION_FNS:
            return REVERSE_CUSTOM_ACTIVATION_FNS[func]  # Get the serialization key
        else:
            raise ValueError(f"Unsupported activation function: {v}")

    @field_validator("params_dtype", "pipeline_dtype", "autocast_dtype", mode="before")
    @classmethod
    def precision_validator(cls, v: dtypes.PrecisionTypes) -> torch.dtype:
        return dtypes.get_autocast_dtype(v)

    @field_serializer("params_dtype", "pipeline_dtype", "autocast_dtype")
    def serialize_dtypes(self, v: torch.dtype) -> dtypes.PrecisionTypes:
        return dtypes.dtype_to_precision[v]


class ParallelConfig(BaseModel):
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    accumulate_grad_batches: int = 1
    ddp: Literal["megatron"] = "megatron"
    remove_unused_parameters: bool = True
    num_devices: int = 1
    num_nodes: int = 1

    @model_validator(mode="after")
    def validate_devices(self):
        # I think we can do a 2x2 split on 2 gpus for pipeline/tensor model parallel
        if self.num_devices < self.tensor_model_parallel_size * self.pipeline_model_parallel_size:
            raise ValidationError(
                "devices must be divisible by tensor_model_parallel_size * pipeline_model_parallel_size"
            )
        return self


class TrainingConfig(BaseModel):
    max_steps: int
    limit_val_batches: int
    val_check_interval: int
    # NOTE this matches whats used by nl.MegatronMixedPrecision which has a restricted set of precisions.
    precision: Literal["32", "bf16-mixed", "16-mixed"] = "bf16-mixed"
    accelerator: str = "gpu"


class OptimizerSchedulerConfig(BaseModel):
    # TODO validators on optimizer, interval, and monitor.
    lr: float = 1e-4
    optimizer: str = "adam"
    cosine_rampup_frac: float = 0.01
    cosine_hold_frac: float = 0.05
    interval: str = "step"
    monitor: str = "val_loss"


class ExperimentConfig(BaseModel):
    save_every_n_steps: int
    result_dir: str
    experiment_name: str
    restore_from_checkpoint_path: Optional[str]
    wandb_config: Optional[WandbConfig] = None
    save_last_checkpoint: bool = True
    metric_to_monitor_for_checkpoints: str = "reduced_train_loss"
    save_top_k: int = 2
    create_tensorboard_logger: bool = False


# DataConfig -> some config that can make a data module (see ABC definition.)
DataConfigT = TypeVar("DataConfigT", bound=DataConfig)
# ExposedModelConfig -> some config that can make a non-exposed model config (see ABC definition.)
ExModelConfigT = TypeVar("ExModelConfigT", bound=ExposedModelConfig)


class MainConfig(BaseModel, Generic[ExModelConfigT, DataConfigT]):
    """Main configuration class for BioNeMo. All serialized configs that are a valid MainConfig should be Runnable.

    This class is used to define the main configuration for BioNeMo. It defines the minimal pieces of configuration
    to execution a training job with the NeMo2 training api. It accepts two generic type parameters which users
    must define in their own environment for execution.

    Args:
        data_config: Generic config type that contains instructions on instantiating the required DataModule.
        parallel_config: The parallel configuration for the model.
        training_config: The training configuration for the model.
        bionemo_model_config: Generic ExposedModelConfig type. This class hides extra configuration parameters in the
            underlying model configuration as well as providing
        optim_config: The optimizer/scheduler configuration for the model.
        experiment_config: The experiment configuration for the model.
        wandb_config: Optional, the wandb configuration for the model.
    """

    data_config: DataConfigT
    parallel_config: ParallelConfig
    training_config: TrainingConfig
    bionemo_model_config: ExModelConfigT
    optim_config: OptimizerSchedulerConfig
    experiment_config: ExperimentConfig
    wandb_config: Optional[WandbConfig] = None

    @model_validator(mode="after")
    def validate_master_config(self) -> "MainConfig":
        self.bionemo_model_config.seq_length = self.data_config.seq_length
        # What other global validators should we set here?
        return self
    
    @model_validator(mode="after")
    def run_bionemo_model_config_model_validators(self) -> "MainConfig":
        return self.bionemo_model_config.model_validator(self)
    
    @model_validator(mode="after")
    def run_data_config_modeL_validators(self) -> "MainConfig":
        return self.data_config.model_validator(self)
