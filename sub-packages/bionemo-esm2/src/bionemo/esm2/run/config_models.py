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


import importlib
from pathlib import Path
from typing import Optional, Type

import torch
from nemo.utils import logging
from pydantic import field_serializer, field_validator, model_validator

from bionemo.esm2.data.datamodule import ESMDataModule
from bionemo.esm2.data.dataset import RandomMaskStrategy
from bionemo.esm2.data.tokenizer import get_tokenizer
from bionemo.esm2.model.attention import ESM2DotProductAttention, ESM2TEDotProductAttention
from bionemo.esm2.model.model import ESM2Config
from bionemo.llm.config.config_models import (
    DataConfig,
    ExposedModelConfig,
    MainConfig,
)
from bionemo.llm.model.biobert.model import BiobertSpecOption


class ESM2DataConfig(DataConfig[ESMDataModule]):
    # defined in baseclass- listed here for exposure.
    train_cluster_path: Path
    train_database_path: Path
    valid_cluster_path: Path
    valid_database_path: Path

    micro_batch_size: int = 8
    result_dir: str = "./results"
    min_seq_length: int = 128
    max_seq_length: int = 128
    random_mask_strategy: RandomMaskStrategy = RandomMaskStrategy.ALL_TOKENS
    num_dataset_workers: int = 0

    def construct_data_module(self, global_batch_size: int) -> ESMDataModule:
        tokenizer = get_tokenizer()
        data = ESMDataModule(
            train_cluster_path=self.train_cluster_path,
            train_database_path=self.train_database_path,
            valid_cluster_path=self.valid_cluster_path,
            valid_database_path=self.valid_database_path,
            global_batch_size=global_batch_size,
            micro_batch_size=self.micro_batch_size,
            min_seq_length=self.min_seq_length,
            max_seq_length=self.max_seq_length,
            num_workers=self.num_dataset_workers,
            random_mask_strategy=self.random_mask_strategy,
            tokenizer=tokenizer,
        )
        return data


class ExposedESM2PretrainConfig(ExposedModelConfig[ESM2Config]):
    # ESM specific fields
    use_esm_attention: bool = False  # Skip ESM2 custom attention for TE acceleration. Still passes golden value test.
    token_dropout: bool = True
    normalize_attention_scores: bool = False
    variable_seq_lengths: bool = False
    core_attention_override: Type[torch.nn.Module] | None = None

    @field_validator("biobert_spec_option", mode="after")
    @classmethod
    def restrict_biobert_spec_to_esm2(cls, biobert_spec_option: BiobertSpecOption) -> BiobertSpecOption:
        # This has some more complicated validation I see

        if biobert_spec_option in (
            BiobertSpecOption.esm2_bert_layer_with_transformer_engine_spec,
            BiobertSpecOption.esm2_bert_layer_local_spec,
        ):
            return biobert_spec_option
        else:
            raise TypeError(
                f"Unsupported BiobertSpecOption: {biobert_spec_option=}, use one of {BiobertSpecOption.esm2_bert_layer_with_transformer_engine_spec}, {BiobertSpecOption.esm2_bert_layer_local_spec}"
            )

    @field_serializer("core_attention_override")
    def serialize_core_attention_override(self, value: Optional[Type[torch.nn.Module]]) -> Optional[str]:
        if value is None:
            return None
        return f"{value.__module__}.{value.__name__}"

    @field_validator("core_attention_override", mode="before")
    def validate_core_attention_override(cls, value):
        if value is None:
            return None
        if isinstance(value, str):
            module_name, class_name = value.rsplit(".", 1)
            try:
                module = importlib.import_module(module_name)
                cls = getattr(module, class_name)
                if not issubclass(cls, torch.nn.Module):
                    raise ValueError(f"{cls} is not a subclass of torch.nn.Module")
                return cls
            except (ImportError, AttributeError):
                raise ValueError(f"Cannot import {value}")
        return value

    @model_validator(mode="after")
    def validate_and_set_attention_and_scaling(self):
        logging.info(
            "Mutating apply_query_key_layer_scaling and core_attention_override based on biobert_spec_option.."
        )
        if self.biobert_spec_option == BiobertSpecOption.esm2_bert_layer_with_transformer_engine_spec:
            self.apply_query_key_layer_scaling = False
            self.core_attention_override = ESM2TEDotProductAttention
        elif self.biobert_spec_option == BiobertSpecOption.esm2_bert_layer_local_spec:
            logging.warning(
                "BiobertSpecOption.esm2_bert_layer_local_spec is deprecated. "
                "Use BiobertSpecOption.esm2_bert_layer_with_transformer_engine_spec instead."
            )
            self.apply_query_key_layer_scaling = True
            self.core_attention_override = ESM2DotProductAttention
        return self

    def model_validator(self, global_cfg: MainConfig) -> MainConfig:
        global_cfg = super().model_validator(global_cfg)
        # Need to ensure that at the least we have access to min_seq_length and max_seq_length
        if not isinstance(global_cfg.data_config, ESM2DataConfig):
            raise TypeError(f"ESM2PretrainConfig requires ESM2DataConfig, got {global_cfg.data_config=}")

        pipeline_model_parallel_size, tensor_model_parallel_size = (
            global_cfg.parallel_config.pipeline_model_parallel_size,
            global_cfg.parallel_config.tensor_model_parallel_size,
        )
        min_seq_length, max_seq_length = global_cfg.data_config.min_seq_length, global_cfg.data_config.max_seq_length
        assert (
            self.variable_seq_lengths
            == (pipeline_model_parallel_size * tensor_model_parallel_size > 1 and min_seq_length != max_seq_length)
        ), "Must set variable_seq_lengths = (pipeline_model_parallel_size * tensor_model_parallel_size > 1 and min_seq_length != max_seq_length)"
        return global_cfg

    def model_class(self) -> Type[ESM2Config]:
        return ESM2Config


# TODO NOTES on default configuration
# seq_length: int # max_sequence_length
# need_megatron_variable_seq_lengths_reductions = (pipeline_model_parallel_size * tensor_model_parallel_size > 1 and min_seq_length != max_seq_length)
