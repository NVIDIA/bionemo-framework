# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""ProjectorAdamWConfig: trainability applied at the last safe point.

The trainability policy is applied inside ``build`` — after PEFT freezing, before optimizer construction —
and validated by a deterministic manifest. This is the local alternative to
a public AutoModel pre-optimizer hook.

Consumers on older AutoModel revisions may still need a different optimizer
seam. The current fixed-geometry applications use the typed
``AdamWConfig.build`` API through the package's ``ProjectorFinetuneRecipe``;
this class remains the trainability-applying optimizer config for that path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

import torch

from nemotron_stitch.projector.trainability import (
    TrainabilityPolicy,
    configure_trainable_parameters,
    write_manifest_rank_zero,
)

# The base is Any-typed for the checker: the real class comes from the pinned
# AutoModel at runtime, and the except-branch fallback must not union with it.
_AdamWConfig: Any
try:
    from nemo_automodel.components.optim.optimizer import AdamWConfig as _AdamWConfig
except ModuleNotFoundError:

    @dataclass
    class _AdamWConfig:
        """API-compatible fallback used by unit tests outside the training image."""

        lr: float = 1e-4
        weight_decay: float = 0.01
        betas: tuple[float, float] = (0.9, 0.999)
        eps: float = 1e-8
        amsgrad: bool = False
        fused: bool = False

        def build(self, model, *, device_mesh=None, is_peft=False):
            del device_mesh, is_peft
            parts = getattr(model, "parts", [model])
            return [
                torch.optim.AdamW(
                    [parameter for parameter in part.parameters() if parameter.requires_grad],
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                    betas=self.betas,
                    eps=self.eps,
                    amsgrad=self.amsgrad,
                    fused=self.fused,
                )
                for part in parts
            ]


@dataclass
class ProjectorAdamWConfig(_AdamWConfig):
    """Re-enable the trainable families after PEFT freezing, then build AdamW."""

    policy: TrainabilityPolicy | None = None
    stage: int = 1
    train_projector: bool = True
    train_extra: bool = False
    train_decoder: bool = False
    manifest_path: str | None = None

    def build(self, model, *, device_mesh=None, is_peft=False):
        if self.policy is None:
            raise ValueError("ProjectorAdamWConfig requires a TrainabilityPolicy")
        manifest = configure_trainable_parameters(
            model,
            self.stage,
            policy=self.policy,
            train_projector=self.train_projector,
            train_extra=self.train_extra,
            train_decoder=self.train_decoder,
        )
        write_manifest_rank_zero(manifest, self.manifest_path)
        return super().build(model, device_mesh=device_mesh, is_peft=is_peft)

    def _build_optimizer(self, params, *, foreach=None):
        # AutoModel 9af3b45a's AdamWConfig._build_optimizer forwards
        # asdict(self) into torch.optim.AdamW, so this subclass's own fields
        # (policy, stage, ...) would leak in as unknown kwargs. Restrict the
        # mapping to the base class's fields; upstream already guarantees those
        # are valid AdamW kwargs. Delete when the runtime exposes a public pre-optimizer hook.
        base_fields = {field.name for field in fields(_AdamWConfig)}
        kwargs = {key: value for key, value in asdict(self).items() if key in base_fields}
        # r0.6.0 added OptimizerConfig.param_group_overrides: a base field, but
        # a grouping input consumed by build(), not an AdamW constructor kwarg
        # (upstream's own _constructor_kwargs strips it via
        # _NON_CONSTRUCTOR_FIELDS). We never set it; drop it here too.
        kwargs.pop("param_group_overrides", None)
        return torch.optim.AdamW(params, foreach=foreach and not self.fused, **kwargs)
