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

from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from nemotron_stitch.features.cache import CacheBuilder, CacheContract
from nemotron_stitch.nemo_rl.transport import (
    ground_truth_metadata,
    load_cached_features,
    load_encoder_payload,
    project_features,
    task_reward_metadata,
)
from nemotron_stitch.projector import MultimodalProjector
from nemotron_stitch.projector.artifact import save_projector_artifact


def _loader(datum, adapter, *, scale):
    assert adapter == "sequence"
    raw = torch.full((3, 4), float(scale))
    return raw, torch.zeros(3, 6)


def _metadata(datum):
    return {"label": datum["label"]}


class _TokenModel(nn.Module):
    def __init__(self):
        super().__init__()
        config = {"name": "sequence", "kind": "mlp2x_gelu", "mm_hidden_size": 4, "hidden_size": 8}
        self.config = SimpleNamespace(mm_projectors=[config])
        self.mm_projector = MultimodalProjector.from_config([config], output_size=6)


def test_generic_loader_and_task_callbacks_are_explicit():
    datum = {
        "encoder_loader": f"{__name__}._loader",
        "encoder_loader_kwargs": {"scale": 2},
        "task_metadata_callback": f"{__name__}._metadata",
        "label": "yes",
    }
    raw, projected = load_encoder_payload(datum, "sequence")
    assert raw.shape == (3, 4)
    assert projected.shape == (3, 6)
    assert task_reward_metadata(datum) == {"label": "yes"}


def test_mlp2x_gelu_projects_through_the_grpo_sidecar_path(tmp_path):
    model = _TokenModel()
    save_projector_artifact(model, tmp_path, provenance={"source": {"revision": "locked"}})
    projected = project_features(torch.randn(3, 4), str(tmp_path), "sequence")
    assert projected.shape == (3, 6)


def test_cached_feature_callback_returns_raw_and_projected_views(tmp_path):
    contract = CacheContract.for_feature_cache(
        source_manifest_id="0" * 64,
        checkpoint_id="org/encoder",
        revision="revision",
        implementation_revision="implementation",
        layer=-1,
        max_context_units=3,
        dtype="float32",
        source_release="release",
    )
    features = np.arange(12, dtype=np.float32).reshape(3, 4)
    builder = CacheBuilder(tmp_path / "cache", contract)
    builder.add("feature-key", features)
    builder.publish()
    model = _TokenModel()
    save_projector_artifact(model, tmp_path / "projector", provenance={"source": {"revision": "locked"}})

    raw, projected = load_cached_features(
        {"feature_key": "feature-key"},
        "sequence",
        cache_root=str(tmp_path / "cache"),
        artifact_dir=str(tmp_path / "projector"),
    )
    torch.testing.assert_close(raw, torch.from_numpy(features))
    assert projected.shape == (3, 6)
    assert ground_truth_metadata({"ground_truth": 7}) == {"ground_truth": "7"}
