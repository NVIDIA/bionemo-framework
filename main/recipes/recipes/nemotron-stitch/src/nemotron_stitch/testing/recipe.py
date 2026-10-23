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

"""A framework-free double for the ``BaseRecipe`` state-tracker contract.

``ProjectorSidecarState`` (automodel/recipe.py) relies on a narrow slice of
AutoModel's ``BaseRecipe`` (24b47e856263d313b942f0ed666c63fff83306b4), and both
the package suite and consumer examples demonstrate against it without
importing the framework:

- ``__setattr__`` tracks an assigned object with callable ``state_dict()`` and
  ``load_state_dict()``, skipping ``val``/``eval``/``test``/``loss`` attribute
  names and rejecting duplicate tracked keys;
- ``setup()`` ends in native resume, feeding each tracked ``<key>.pt`` back to
  ``load_state_dict`` (the terminal ``load_checkpoint`` in
  ``recipes/llm/train_ft.py``);
- periodic/final saves ``torch.save`` each tracked object's ``state_dict()``
  as ``<key>.pt`` on the coordinator (sidecar recipes are one-rank, so the
  coordinator is the only rank).

Model/optimizer/dataloader routing is deliberately out of scope: the projector
wrapper is the only stateful attribute the mixin attaches, and the double's
model is a plain attribute the tracker never sees.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

_SKIP_SUBSTRINGS = ("val", "eval", "test", "loss")


class CheckpointTrackerFake:
    """Cooperative base for ``class ConsumerRecipe(ProjectorRecipeMixin, CheckpointTrackerFake)``."""

    def __init__(self, cfg, *, world_size: int = 1, checkpoint_dir: str | Path | None = None):
        self.cfg = cfg
        self.dist_env = SimpleNamespace(world_size=world_size, device=torch.device("cpu"))
        self.model_parts = []
        self.pp_enabled = False
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None

    def __setattr__(self, key, value):
        if (
            callable(getattr(value, "state_dict", None))
            and callable(getattr(value, "load_state_dict", None))
            and not any(skip in key.lower() for skip in _SKIP_SUBSTRINGS)
        ):
            tracked = self.__dict__.setdefault("_tracked_state_keys", set())
            if key in tracked:
                raise RuntimeError(f"State key {key!r} is already tracked")
            tracked.add(key)
        super().__setattr__(key, value)

    @property
    def tracked_state_keys(self) -> set[str]:
        return set(self.__dict__.get("_tracked_state_keys", ()))

    def _get_cp_group_size(self) -> int:
        return 1

    def setup(self):
        # Mirror train_ft.py: native resume runs at the end of base setup, so
        # a wrapper attached before super().setup() is loaded by the tracker.
        self.load_checkpoint()
        return "setup-result"

    def save_checkpoint(self, step: int = 1) -> Path:
        if self.checkpoint_dir is None:
            raise ValueError("checkpoint_dir is required to save a checkpoint")
        path = self.checkpoint_dir / f"epoch_0_step_{step}"
        path.mkdir(parents=True, exist_ok=True)
        for key in sorted(self.tracked_state_keys):
            torch.save(getattr(self, key).state_dict(), path / f"{key}.pt")
        return path

    def load_checkpoint(self, step: int = 1) -> None:
        if self.checkpoint_dir is None:
            return
        path = self.checkpoint_dir / f"epoch_0_step_{step}"
        if not path.is_dir():
            return
        for key in sorted(self.tracked_state_keys):
            # A tracked key without its file is a truncated checkpoint; let
            # torch.load raise rather than resume with partial state.
            getattr(self, key).load_state_dict(torch.load(path / f"{key}.pt", weights_only=True))
