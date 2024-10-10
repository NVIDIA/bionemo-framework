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


import math
from typing import Dict, List, Union

import omegaconf
import torch
import torch.nn.functional as F
import webdataset as wds
from einops import rearrange
from einops.layers.torch import Rearrange
from nemo.collections.nlp.modules.common.megatron.megatron_init import initialize_model_parallel_for_nemo
from nemo.core import ModelPT
from torch import nn
from torch.utils.checkpoint import checkpoint_sequential

from bionemo.data.dna.enformer.basenji_dataset import WebBasenji, WebCombinedBasenji
from bionemo.model.dna.enformer.metrics import MeanPearsonCorrCoefPerChannel

from .attention import Attention, AttentionPool
from .blocks import GELU, ConvBlock, Residual, TargetLengthCrop
from .data import seq_indices_to_one_hot, str_to_one_hot
from .helpers import exists, map_values


def exponential_linspace_int(start, end, num, divisible_by=1):
    def _round(x):
        return int(round(x / divisible_by) * divisible_by)

    base = math.exp(math.log(end / start) / (num - 1))
    return [_round(start * base**i) for i in range(num)]


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def poisson_loss(pred, target):
    return (pred - target * log(pred)).mean()


def pearson_corr_coef(x, y, dim=1, reduce_dims=(-1,)):
    x_centered = x - x.mean(dim=dim, keepdim=True)
    y_centered = y - y.mean(dim=dim, keepdim=True)
    return F.cosine_similarity(x_centered, y_centered, dim=dim).mean(dim=reduce_dims)


class Enformer(ModelPT):
    """
    Enformer - Effective gene expression prediction from sequence by integrating long-range interactions
    As described in https://www.nature.com/articles/s41592-021-01252-x

    This is not Megatron-based model.

    """

    def __init__(self, cfg, trainer, *args, **kwarg):
        super().__init__(cfg, trainer, *args, **kwarg)
        self.dim = cfg.dim
        half_dim = cfg.dim // 2
        twice_dim = cfg.dim * 2

        # create stem

        self.stem = nn.Sequential(
            nn.Conv1d(4, half_dim, 15, padding=7), Residual(ConvBlock(half_dim)), AttentionPool(half_dim, pool_size=2)
        )

        # create conv tower

        filter_list = exponential_linspace_int(
            half_dim, cfg.dim, num=(cfg.num_downsamples - 1), divisible_by=cfg.dim_divisible_by
        )
        filter_list = [half_dim, *filter_list]

        conv_layers = []
        for dim_in, dim_out in zip(filter_list[:-1], filter_list[1:]):
            conv_layers.append(
                nn.Sequential(
                    ConvBlock(dim_in, dim_out, kernel_size=5),
                    Residual(ConvBlock(dim_out, dim_out, 1)),
                    AttentionPool(dim_out, pool_size=2),
                )
            )

        self.conv_tower = nn.Sequential(*conv_layers)

        # transformer

        transformer = []
        for _ in range(cfg.depth):
            transformer.append(
                nn.Sequential(
                    Residual(
                        nn.Sequential(
                            nn.LayerNorm(cfg.dim),
                            Attention(
                                cfg.dim,
                                heads=cfg.heads,
                                dim_key=cfg.attn_dim_key,
                                dim_value=cfg.dim // cfg.heads,
                                dropout=cfg.attn_dropout,
                                pos_dropout=cfg.pos_dropout,
                                num_rel_pos_features=cfg.dim // cfg.heads,
                            ),
                            nn.Dropout(cfg.dropout_rate),
                        )
                    ),
                    Residual(
                        nn.Sequential(
                            nn.LayerNorm(cfg.dim),
                            nn.Linear(cfg.dim, cfg.dim * 2),
                            nn.Dropout(cfg.dropout_rate),
                            nn.ReLU(),
                            nn.Linear(cfg.dim * 2, cfg.dim),
                            nn.Dropout(cfg.dropout_rate),
                        )
                    ),
                )
            )

        self.transformer = nn.Sequential(*transformer)

        # target cropping

        self.target_length = cfg.target_length
        self.crop_final = TargetLengthCrop(cfg.target_length)

        # final pointwise

        self.final_pointwise = nn.Sequential(
            Rearrange("b n d -> b d n"),
            ConvBlock(filter_list[-1], twice_dim, 1),
            Rearrange("b d n -> b n d"),
            nn.Dropout(cfg.dropout_rate / 8),
            GELU(),
        )

        # create trunk sequential module

        self._trunk = nn.Sequential(
            Rearrange("b n d -> b d n"),
            self.stem,
            self.conv_tower,
            Rearrange("b d n -> b n d"),
            self.transformer,
            self.crop_final,
            self.final_pointwise,
        )

        # create final heads for human and mouse

        self.add_heads(**cfg.output_heads)

        # use checkpointing on transformer trunk

        self.use_checkpointing = cfg.use_checkpointing
        self.setup_metrics()
        initialize_model_parallel_for_nemo(
            world_size=trainer.world_size,
            global_rank=trainer.global_rank,
            local_rank=trainer.local_rank,
            tensor_model_parallel_size=cfg.tensor_model_parallel_size,
            pipeline_model_parallel_size=cfg.pipeline_model_parallel_size,
        )

    def add_heads(self, **kwargs):
        self.output_heads = kwargs

        self._heads = nn.ModuleDict(
            map_values(lambda features: nn.Sequential(nn.Linear(self.dim * 2, features), nn.Softplus()), kwargs)
        )

    def set_target_length(self, target_length):
        crop_module = self._trunk[-2]
        crop_module.target_length = target_length

    @property
    def trunk(self):
        return self._trunk

    @property
    def heads(self):
        return self._heads

    def trunk_checkpointed(self, x):
        x = rearrange(x, "b n d -> b d n")
        x = self.stem(x)
        x = self.conv_tower(x)
        x = rearrange(x, "b d n -> b n d")
        x = checkpoint_sequential(self.transformer, len(self.transformer), x)
        x = self.crop_final(x)
        x = self.final_pointwise(x)
        return x

    def forward(
        self,
        x: Union[List[str], torch.Tensor],
        target: torch.Tensor = None,
        return_corr_coef: bool = False,
        return_embeddings: bool = False,
        return_only_embeddings: bool = False,
        head: str = None,
    ):
        """
        Args:
            x (Union[List[str], torch.Tensor]): DNA sequence. Might be a List of strings of Nucleic acid sequence
            e.g ['ATCG', 'ACCT'] or torch.Tensor of indices. For mapping please refer to bionemo.data.dna.enformer.genome_interval
            target (torch.Tensor, optional): Gene expression levels from experiment. Defaults to None.
            return_corr_coef (bool, optional): Flag for returnng correlation coefficient between target and predicted
            gene expression. Requires target to be provided. Defaults to False.
            return_embeddings (bool, optional): Flag for returning gene expression predictions and embeddings
            before specialised heads. Defaults to False.
            return_only_embeddings (bool, optional): Flag for returning only embeddings before specialised
            heads, without gene expression predictions. Defaults to False.
            head (str, optional): Specify which head should be used for prediction.
            Defaults to Non which means all heads will be calculated.

        Returns:
            torch.Tensor: prediction of gene expression tracks along with embed
        """
        if isinstance(x, list):
            x = str_to_one_hot(x)

        elif x.dtype == torch.long:
            x = seq_indices_to_one_hot(x)

        no_batch = x.ndim == 2

        if no_batch:
            x = rearrange(x, "... -> () ...")

        trunk_fn = self.trunk_checkpointed if self.use_checkpointing else self._trunk
        x = trunk_fn(x)

        if no_batch:
            x = rearrange(x, "() ... -> ...")

        if return_only_embeddings:
            return x

        out = map_values(lambda fn: fn(x), self._heads)

        if exists(head):
            assert head in self._heads, f"head {head} not found"
            out = out[head]

        if exists(target):
            assert exists(head), "head must be passed in if one were to calculate loss directly with targets"

            if return_corr_coef:
                return pearson_corr_coef(out, target)

            return poisson_loss(out, target)

        if return_embeddings:
            return out, x

        return out

    def setup_training_data(self, train_data_cfg: Union[omegaconf.DictConfig, Dict]):
        organisms = self.cfg.metrics.organisms.keys()
        if ("human" in organisms) and ("mouse" in organisms):
            train_ds = WebCombinedBasenji(
                dataset_path=train_data_cfg.dataset_path,
                batch_size=train_data_cfg.batch_size,
                context_length=self.cfg.context_length,
            )
        else:
            assert "human" in organisms
            train_ds = WebBasenji(
                dataset_path=train_data_cfg.dataset_path,
                subset="train",
                organism="human",
                batch_size=train_data_cfg.batch_size,
                context_length=self.cfg.context_length,
                repeat=True,
            )
        self._train_dl = wds.WebLoader(train_ds, num_workers=1, batch_size=None)

    def training_step(self, batch, batch_idx, dataset_idx=None):
        head = batch["head_name"]
        preds = self(x=batch["sequence"], head=head)
        return poisson_loss(pred=preds, target=batch["target"])

    def _single_infer_dataloader(self, cfg: Union[omegaconf.DictConfig, Dict], split: str):
        def dl_organism(organism: str):
            organism_ds = WebBasenji(
                dataset_path=cfg["dataset_path"],
                subset=split,
                organism=organism,
                batch_size=cfg["batch_size"],
                context_length=self.cfg.context_length,
            )
            return wds.WebLoader(organism_ds, num_workers=1, batch_size=None)

        dls = []
        names = []

        for org in self.cfg.metrics.organisms.keys():
            dls.append(dl_organism(org))
            names.append(org)

        return dls, names

    def setup_validation_data(self, val_data_cfg: Union[omegaconf.DictConfig, Dict]):
        self._validation_dl, self._validation_names = self._single_infer_dataloader(cfg=val_data_cfg, split="valid")

    def setup_test_data(self, test_data_config: Union[omegaconf.DictConfig, Dict]):
        self._test_dl, self._test_names = self._single_infer_dataloader(cfg=test_data_config, split="test")

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if batch is None:
            return
        head = batch["head_name"]
        preds = self(x=batch["sequence"], head=head)
        metric = getattr(self, f"rpearson_{head}")
        metric.update(preds=preds, target=batch["target"])

    def on_validation_epoch_end(self):
        for organism in self.cfg.metrics.organisms.keys():
            metric = getattr(self, f"rpearson_{organism}")
            self.log(
                f"rpearson_{organism}",
                metric.compute().mean(),
                sync_dist=False,
                on_step=False,
                on_epoch=True,
                logger=True,
                prog_bar=True,
            )
            metric.reset()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        self.validation_step(batch, batch_idx, dataloader_idx=dataloader_idx)

    def on_test_epoch_end(self):
        self.on_validation_epoch_end()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return {"name": batch["name"], "pred": self(x=batch["seq"], head="human")}

    def setup_metrics(self):
        for organism, cfg in self.cfg.metrics.organisms.items():
            metric = MeanPearsonCorrCoefPerChannel(n_channels=cfg.num_targets)
            setattr(self, f"rpearson_{organism}", metric)

    def list_available_models(self):
        pass
