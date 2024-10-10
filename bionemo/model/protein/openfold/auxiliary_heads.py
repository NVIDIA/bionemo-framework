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


# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig

from bionemo.model.protein.openfold.layer_norm import LayerNorm
from bionemo.model.protein.openfold.linear import Linear
from bionemo.model.protein.openfold.loss import compute_plddt, compute_predicted_aligned_error, compute_tm
from bionemo.model.protein.openfold.utils.torch_utils import is_autocast_fp16_enabled


class AuxiliaryHeads(nn.Module):
    """Auxiliary Heads module."""

    def __init__(self, config: Union[DictConfig, Dict]) -> None:
        super(AuxiliaryHeads, self).__init__()
        self.plddt = PerResidueLDDTCaPredictor(
            **(config.per_residue_lddt_ca_predictor_config),
        )
        self.distogram = DistogramHead(
            **(config.distogram_head_config),
        )
        self.masked_msa = MaskedMSAHead(
            **(config.masked_msa_head_config),
        )
        self.experimentally_resolved = ExperimentallyResolvedHead(
            **(config.experimentally_resolved_head_config),
        )
        self.tm_score_head_enabled = config.tm_score_head_enabled
        if self.tm_score_head_enabled:
            self.tm = TMScoreHead(
                **(config.tm_score_head_config),
            )

    def forward(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        aux_outputs = {}
        aux_outputs["lddt_logits"] = self.plddt(s=outputs["sm_single"])
        aux_outputs["plddt"] = compute_plddt(logits=aux_outputs["lddt_logits"])
        aux_outputs["distogram_logits"] = self.distogram(outputs["pair"])
        aux_outputs["masked_msa_logits"] = self.masked_msa(outputs["msa"])
        aux_outputs["experimentally_resolved_logits"] = self.experimentally_resolved(outputs["single"])
        if self.tm_score_head_enabled:
            aux_outputs["tm_logits"] = self.tm(outputs["pair"])
            aux_outputs["predicted_tm_score"] = compute_tm(
                logits=aux_outputs["tm_logits"],
                max_bin=self.tm.max_bin,
                num_bins=self.tm.num_bins,
            )
            aux_outputs.update(
                compute_predicted_aligned_error(
                    logits=aux_outputs["tm_logits"],
                    max_bin=self.tm.max_bin,
                    num_bins=self.tm.num_bins,
                )
            )
        return aux_outputs


class PerResidueLDDTCaPredictor(nn.Module):
    """Per-Residue LDDT-Ca Predictor module.

    Supplementary '1.9.6 Model confidence prediction (pLDDT)': Algorithm 29.

    Args:
        c_s: Single representation dimension (channels).
        c_hidden: Hidden dimension (channels).
        num_bins: Output dimension (channels).

    """

    def __init__(
        self,
        c_s: int,
        c_hidden: int,
        num_bins: int,
    ) -> None:
        super(PerResidueLDDTCaPredictor, self).__init__()
        self.layer_norm = LayerNorm(c_s)
        self.linear_1 = Linear(c_s, c_hidden, bias=True, init="relu")
        self.linear_2 = Linear(c_hidden, c_hidden, bias=True, init="relu")
        self.linear_3 = Linear(c_hidden, num_bins, bias=True, init="final")

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        s = self.layer_norm(s)
        s = self.linear_1(s)
        s = torch.relu(s)
        s = self.linear_2(s)
        s = torch.relu(s)
        s = self.linear_3(s)
        return s


class DistogramHead(nn.Module):
    """Distogram Head module.

    Computes a distogram probability distribution.

    Supplementary '1.9.8 Distogram prediction'.

    Args:
        c_z: Pair representation dimension (channels).
        num_bins: Output dimension (channels).

    """

    def __init__(
        self,
        c_z: int,
        num_bins: int,
    ) -> None:
        super(DistogramHead, self).__init__()
        self.linear = Linear(c_z, num_bins, bias=True, init="final")

    def _forward(self, z: torch.Tensor) -> torch.Tensor:
        logits = self.linear(z)
        logits = logits + logits.transpose(-2, -3)
        return logits

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if is_autocast_fp16_enabled():
            with torch.cuda.amp.autocast(enabled=False):
                return self._forward(z.float())
        else:
            return self._forward(z)


class MaskedMSAHead(nn.Module):
    """Masked MSA Head module.

    Supplementary '1.9.9 Masked MSA prediction'.

    Args:
        c_m: MSA representation dimension (channels).
        c_out: Output dimension (channels).

    """

    def __init__(
        self,
        c_m: int,
        c_out: int,
    ) -> None:
        super(MaskedMSAHead, self).__init__()
        self.linear = Linear(c_m, c_out, bias=True, init="final")

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        logits = self.linear(m)
        return logits


class ExperimentallyResolvedHead(nn.Module):
    """Experimentally Resolved Head module.

    Supplementary '1.9.10 Experimentally resolved prediction'.

    Args:
        c_s: Single representation dimension (channels).
        c_out: Output dimension (channels).

    """

    def __init__(
        self,
        c_s: int,
        c_out: int,
    ) -> None:
        super(ExperimentallyResolvedHead, self).__init__()
        self.linear = Linear(c_s, c_out, bias=True, init="final")

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        logits = self.linear(s)
        return logits


class TMScoreHead(nn.Module):
    """TM-Score Head module.

    Supplementary '1.9.7 TM-score prediction'.

    Args:
        c_z: Pair representation dimension (channels).
        num_bins: Output dimension (channels).
        max_bin: Max bin range for discretizing the distribution.

    """

    def __init__(
        self,
        c_z: int,
        num_bins: int,
        max_bin: int,
    ) -> None:
        super(TMScoreHead, self).__init__()
        self.num_bins = num_bins
        self.max_bin = max_bin
        self.linear = Linear(c_z, num_bins, bias=True, init="final")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        logits = self.linear(z)
        return logits
