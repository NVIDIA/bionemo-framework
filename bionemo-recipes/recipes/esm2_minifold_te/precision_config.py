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

"""Per-layer MXFP8 precision configuration for the MiniFold TE folding head.

Enables selective MXFP8 quantization on specific sub-layers of the folding head
for precision sensitivity studies. Each sub-layer group can be independently
enabled/disabled for MXFP8.

Usage:
    config = FoldingHeadPrecisionConfig(
        tri_proj=True,       # Triangular update projections
        tri_gate=False,      # Triangular update gates
        ffn=True,            # Transition FFN layers
        struct_attn=False,   # Structure module attention
        struct_ffn=False,    # Structure module FFN
        seq_proj=True,       # Sequence/pair projection layers
    )

    # In training loop:
    with config.get_autocast_context("tri_proj"):
        output = triangular_projection(input)
"""

from contextlib import contextmanager
from dataclasses import dataclass, field

import transformer_engine.pytorch as te


@dataclass
class FoldingHeadPrecisionConfig:
    """Configuration for per-layer MXFP8 precision in the folding head.

    Each field controls whether MXFP8 is enabled for a group of sub-layers.
    When a field is True, the corresponding layers run under te.fp8_autocast().
    When False, they run in the default precision (BF16).

    Attributes:
        enabled: Master switch for MXFP8. If False, all layers run in BF16.
        tri_proj: Triangular update input/output projections (pi, po).
        tri_gate: Triangular update sigmoid gates (gi, go).
        ffn: Transition update FFN layers (fc1, fc2).
        struct_attn: Structure module attention projections (proj, o_proj, g_proj).
        struct_ffn: Structure module transition MLP layers.
        seq_proj: Sequence and pair feature projections (fc_s, fc_z, seq_to_pair).
        dist_head: Distogram output head (fc_out_1, fc_out_2).
        fp8_recipe: TE FP8 recipe class name.
        fp8_recipe_kwargs: Additional kwargs for the FP8 recipe.
    """

    enabled: bool = False
    tri_proj: bool = False
    tri_gate: bool = False
    ffn: bool = False
    struct_attn: bool = False
    struct_ffn: bool = False
    seq_proj: bool = False
    dist_head: bool = False
    fp8_recipe: str = "transformer_engine.common.recipe.DelayedScaling"
    fp8_recipe_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        self._recipe = None

    @property
    def recipe(self):
        """Lazily create the FP8 recipe."""
        if self._recipe is None and self.enabled:
            import importlib

            module_path, class_name = self.fp8_recipe.rsplit(".", 1)
            module = importlib.import_module(module_path)
            recipe_cls = getattr(module, class_name)
            self._recipe = recipe_cls(**self.fp8_recipe_kwargs)
        return self._recipe

    def is_enabled(self, layer_group: str) -> bool:
        """Check if MXFP8 is enabled for a specific layer group.

        Args:
            layer_group: One of "tri_proj", "tri_gate", "ffn", "struct_attn",
                "struct_ffn", "seq_proj", "dist_head".

        Returns:
            True if MXFP8 is enabled for this group.
        """
        if not self.enabled:
            return False
        return getattr(self, layer_group, False)

    @contextmanager
    def get_autocast_context(self, layer_group: str):
        """Get the appropriate autocast context for a layer group.

        Args:
            layer_group: One of "tri_proj", "tri_gate", "ffn", etc.

        Yields:
            te.fp8_autocast context if enabled, else nullcontext.
        """
        if self.is_enabled(layer_group):
            with te.fp8_autocast(enabled=True, fp8_recipe=self.recipe):
                yield
        else:
            yield

    def get_enabled_groups(self) -> list[str]:
        """Return list of layer groups that have MXFP8 enabled."""
        groups = ["tri_proj", "tri_gate", "ffn", "struct_attn", "struct_ffn", "seq_proj", "dist_head"]
        return [g for g in groups if self.is_enabled(g)]

    def summary(self) -> str:
        """Return a human-readable summary of the precision config."""
        if not self.enabled:
            return "MXFP8: disabled (all layers in BF16)"
        enabled = self.get_enabled_groups()
        if not enabled:
            return "MXFP8: enabled but no layer groups selected"
        return f"MXFP8: enabled for {', '.join(enabled)}"
