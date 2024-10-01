# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import math
import tarfile
from typing import Optional, Sequence, Union

import torch
import yaml
from nemo.collections.nlp.parts.nlp_overrides import (
    NLPSaveRestoreConnector,
)
from nemo.utils import logging
from omegaconf import OmegaConf
from pytorch_lightning.trainer.trainer import Trainer


__all__: Sequence[str] = ("BioNeMoSaveRestoreConnector",)


class BioNeMoSaveRestoreConnector(NLPSaveRestoreConnector):
    """
    SaveRestoreConnector that allows changes in the embedding matrix.
    Will truncate the embedding matrix to vocab_size
    if the checkpoint has larger dictionary than the model,
    and upsample the embedding matrix if the the checkpoint
    has smaller dictionary.
    """

    # 128 -- is the number of padded vocabulary in MegatronT5Model
    def __init__(self, vocab_size=None) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def restore_from(
        self,
        calling_cls,
        restore_path: str,
        override_config_path: Optional[Union[OmegaConf, str]] = None,
        map_location: Optional[torch.device] = None,
        strict: bool = True,
        return_config: bool = False,
        trainer: Trainer = None,
    ):
        with tarfile.open(restore_path, "r") as tar:
            _yaml = tar.extractfile("./model_config.yaml")
            obj = yaml.safe_load(_yaml)

        if "target" in obj:
            if not obj["target"] == f"{calling_cls.__module__}.{calling_cls.__name__}":
                logging.warning(
                    f"Restored model is not the same class as the model invoked. restored: {obj['target']}, invoked: {calling_cls}"
                )
        else:
            logging.warning(
                f"No 'target' field in {restore_path=}, cannot guarantee the correct model is restored. Use a newer .nemo file or proceed at your own risk."
            )
        return super().restore_from(
            calling_cls, restore_path, override_config_path, map_location, strict, return_config, trainer
        )

    def modify_state_dict(self, conf, state_dict):
        new_state_dict = {}
        # trunace the word_embeddings and tokens_head
        for key in state_dict.keys():
            if ("word_embeddings" in key or "tokens_head" in key) and "_extra_state" not in key:
                # initialize with pretrained word embeddings
                token_embeddings = state_dict[key]
                if self.vocab_size is None:
                    self.vocab_size = token_embeddings.shape[0]
                logging.info(
                    f"Updating key={key}, token_embeddings.shape={token_embeddings.shape}, vocab_size={self.vocab_size}"
                )
                # tile token_embeddings to be at least self.vocab_size
                dims = (math.ceil(self.vocab_size / token_embeddings.shape[0]),)
                # we want to tile only first dimension
                if len(token_embeddings.shape) == 2:
                    dims += (1,)
                token_embeddings = token_embeddings.tile(dims=dims)
                new_state_dict[key] = token_embeddings[: self.vocab_size]
            elif (
                key.endswith("embedding.position_embeddings.weight")
                and conf.get("encoder", {}).get("arch", None) != "perceiver"
            ):
                position_embeddings = state_dict[key]
                # allow changing the position embeddings for learned_abs
                if ("encoder_embedding" in key) or ("language_model.embedding" in key):
                    if "encoder" in conf:
                        max_position_embeddings = conf.encoder.max_position_embeddings
                    else:
                        max_position_embeddings = conf.max_position_embeddings
                else:
                    if "decoder" in conf:
                        max_position_embeddings = conf.decoder.max_position_embeddings
                    else:
                        max_position_embeddings = conf.max_position_embeddings
                logging.info(
                    f"Updating key={key}, position_embeddings.shape={position_embeddings.shape}, max_position_embeddings={max_position_embeddings}"
                )
                # tile position_embeddings to be at least max_position_embeddings
                position_embeddings = position_embeddings.tile(
                    (math.ceil(max_position_embeddings / position_embeddings.shape[0]), 1)
                )
                new_state_dict[key] = position_embeddings[:max_position_embeddings]
            else:
                new_state_dict[key] = state_dict[key]

        state_dict = new_state_dict
        return state_dict
