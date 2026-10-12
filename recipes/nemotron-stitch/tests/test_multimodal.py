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

"""Flat multimodal forward inputs, ownership, placement, and validation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nemotron_stitch.automodel.model import MultimodalInputMixin, materialize_mm_projector
from nemotron_stitch.contracts import OWNERSHIP_SIDECAR
from nemotron_stitch.projector import Mlp2xGeluProjector, MultimodalProjector


def test_flat_features_derive_positions_and_preserve_gradient():
    projector = Mlp2xGeluProjector(3, 8, 6)
    registry = MultimodalProjector({"arbitrary_tokens": projector})
    ids = torch.tensor([[32, 1, 32, 2]])
    text = torch.randn(1, 4, 6, requires_grad=True)
    features = torch.randn(1, 2, 3, requires_grad=True)
    result = registry(ids, text, {"arbitrary_tokens": features}, placeholder_token_ids={"arbitrary_tokens": 32})
    assert result.shape == text.shape
    # Placeholders sit at columns 0 and 2; those slots carry the soft tokens.
    soft = projector(features)
    torch.testing.assert_close(result[0, 0], soft[0, 0])
    torch.testing.assert_close(result[0, 2], soft[0, 1])
    torch.testing.assert_close(result[0, [1, 3]], text[0, [1, 3]])
    result.sum().backward()
    assert features.grad is not None and torch.count_nonzero(features.grad) == features.numel()


def test_explicit_token_indices_are_an_optional_override():
    registry = MultimodalProjector({"tokens": Mlp2xGeluProjector(3, 8, 6)})
    ids = torch.tensor([[18, 32, 32]])
    text = torch.zeros(1, 3, 6)
    result = registry(
        ids,
        text,
        {"tokens": torch.randn(1, 1, 3)},
        placeholder_token_ids={"tokens": 32},
        token_indices_by_projector={"tokens": torch.tensor([[1]])},
    )
    assert result[0, 0].sum().item() == 0.0
    assert result[0, 1].sum().item() != 0.0


def test_flat_features_with_flat_indices_scatter_ragged_rows():
    """The canonical contract: flat [N, C] features, flat [N] indices."""
    registry = MultimodalProjector({"tokens": Mlp2xGeluProjector(3, 8, 6)})
    ids = torch.tensor([[32, 32, 1], [2, 32, 3]])
    text = torch.zeros(2, 3, 6)
    features = torch.randn(3, 3)
    result = registry(
        ids,
        text,
        {"tokens": features},
        placeholder_token_ids={"tokens": 32},
        token_indices_by_projector={"tokens": torch.tensor([0, 1, 4])},
    )
    soft = registry.projectors["tokens"](features)
    torch.testing.assert_close(result[0, 0], soft[0])
    torch.testing.assert_close(result[0, 1], soft[1])
    torch.testing.assert_close(result[1, 1], soft[2])
    torch.testing.assert_close(result[0, 2], torch.zeros(6))


def test_flat_features_scatter_into_a_thd_token_row():
    registry = MultimodalProjector({"tokens": Mlp2xGeluProjector(3, 8, 6)})
    ids = torch.tensor([32, 1, 32])
    text = torch.zeros(3, 6)
    features = torch.randn(2, 3)

    result = registry(
        ids,
        text,
        {"tokens": features},
        placeholder_token_ids={"tokens": 32},
        token_indices_by_projector={"tokens": torch.tensor([0, 2])},
    )

    soft = registry.projectors["tokens"](features)
    assert result.shape == text.shape
    torch.testing.assert_close(result[[0, 2]], soft)
    torch.testing.assert_close(result[1], torch.zeros(6))


def test_flat_features_derive_flat_positions_and_preserve_gradient():
    # Ragged per-row slot counts (2 then 1): derivation totals over the batch.
    registry = MultimodalProjector({"tokens": Mlp2xGeluProjector(3, 8, 6)})
    ids = torch.tensor([[32, 32, 1], [2, 32, 3]])
    text = torch.zeros(2, 3, 6)
    features = torch.randn(3, 3, requires_grad=True)
    result = registry(ids, text, {"tokens": features}, placeholder_token_ids={"tokens": 32})
    soft = registry.projectors["tokens"](features)
    torch.testing.assert_close(result[0, 0], soft[0])
    torch.testing.assert_close(result[0, 1], soft[1])
    torch.testing.assert_close(result[1, 1], soft[2])
    result.sum().backward()
    assert features.grad is not None and torch.count_nonzero(features.grad) == features.numel()


def test_flat_explicit_indices_drop_negative_entries():
    registry = MultimodalProjector({"tokens": Mlp2xGeluProjector(3, 8, 6)})
    ids = torch.tensor([[32, 1, 1]])
    text = torch.zeros(1, 3, 6)
    features = torch.randn(2, 3)
    result = registry(
        ids,
        text,
        {"tokens": features},
        placeholder_token_ids={"tokens": 32},
        token_indices_by_projector={"tokens": torch.tensor([0, -1])},
    )
    soft = registry.projectors["tokens"](features)
    torch.testing.assert_close(result[0, 0], soft[0])
    torch.testing.assert_close(result[0, 1:], torch.zeros(2, 6))


def test_mixed_flat_and_dense_projectors_in_one_batch():
    registry = MultimodalProjector(
        {
            "grid": Mlp2xGeluProjector(3, 8, 6),
            "packed": Mlp2xGeluProjector(3, 8, 6),
        }
    )
    ids = torch.tensor([[32, 33, 1], [32, 1, 33]])
    text = torch.zeros(2, 3, 6)
    dense_features = torch.randn(2, 1, 3)
    flat_features = torch.randn(2, 3)
    result = registry(
        ids,
        text,
        {"grid": dense_features, "packed": flat_features},
        placeholder_token_ids={"grid": 33, "packed": 32},
    )
    grid_soft = registry.projectors["grid"](dense_features)
    packed_soft = registry.projectors["packed"](flat_features)
    torch.testing.assert_close(result[0, 0], packed_soft[0])
    torch.testing.assert_close(result[1, 0], packed_soft[1])
    torch.testing.assert_close(result[0, 1], grid_soft[0, 0])
    torch.testing.assert_close(result[1, 2], grid_soft[1, 0])


def test_dense_features_with_flat_indices_fail_closed():
    registry = MultimodalProjector({"tokens": Mlp2xGeluProjector(3, 8, 6)})
    with pytest.raises(ValueError, match="do not match the soft-token geometry"):
        registry(
            torch.tensor([[32, 1]]),
            torch.zeros(1, 2, 6),
            {"tokens": torch.randn(1, 1, 3)},
            placeholder_token_ids={"tokens": 32},
            token_indices_by_projector={"tokens": torch.tensor([0])},
        )


def test_rejects_higher_dimensional_soft_tokens():
    # A 4D feature blob is rejected by the projector itself before the
    # registry's own geometry guard runs.
    registry = MultimodalProjector({"tokens": Mlp2xGeluProjector(3, 8, 6)})
    with pytest.raises(ValueError, match="token projector expected"):
        registry(
            torch.tensor([[32]]),
            torch.zeros(1, 1, 6),
            {"tokens": torch.randn(1, 1, 1, 3)},
            placeholder_token_ids={"tokens": 32},
        )


def test_rejects_overlapping_positions():
    registry = MultimodalProjector(
        {
            "first": Mlp2xGeluProjector(3, 8, 6),
            "second": Mlp2xGeluProjector(3, 8, 6),
        }
    )
    with pytest.raises(ValueError, match="overlapping"):
        registry(
            torch.tensor([[32, 1]]),
            torch.zeros(1, 2, 6),
            {"first": torch.randn(1, 1, 3), "second": torch.randn(1, 1, 3)},
            placeholder_token_ids={"first": 32, "second": 33},
            token_indices_by_projector={
                "first": torch.tensor([[0]]),
                "second": torch.tensor([[0]]),
            },
        )


def test_rejects_partial_explicit_indices():
    # The old router dereferenced None here; the package fails closed.
    registry = MultimodalProjector({"a": Mlp2xGeluProjector(3, 8, 6), "b": Mlp2xGeluProjector(3, 8, 6)})
    with pytest.raises(ValueError, match="every projector or none"):
        registry(
            torch.tensor([[32, 32]]),
            torch.zeros(1, 2, 6),
            {"a": torch.randn(1, 1, 3), "b": torch.randn(1, 1, 3)},
            placeholder_token_ids={"a": 32, "b": 33},
            token_indices_by_projector={"a": torch.tensor([[0]])},
        )


def test_rejects_unassigned_placeholder_slots():
    registry = MultimodalProjector({"a": Mlp2xGeluProjector(3, 8, 6)})
    with pytest.raises(ValueError, match="but 3 placeholder slots"):
        registry(
            torch.tensor([[32, 32, 32, 1]]),
            torch.zeros(1, 4, 6),
            {"a": torch.randn(1, 1, 3)},
            placeholder_token_ids={"a": 32},
        )


def test_rejects_inconsistent_slot_counts():
    registry = MultimodalProjector({"a": Mlp2xGeluProjector(3, 8, 6)})
    with pytest.raises(ValueError, match="vary within a batch"):
        registry(
            torch.tensor([[32, 1, 5], [32, 32, 1]]),
            torch.zeros(2, 3, 6),
            {"a": torch.randn(2, 1, 3)},
            placeholder_token_ids={"a": 32},
        )


def test_rejects_unknown_projector():
    registry = MultimodalProjector({"a": Mlp2xGeluProjector(3, 8, 6)})
    with pytest.raises(KeyError, match="unknown"):
        registry(
            torch.tensor([[32, 1]]),
            torch.zeros(1, 2, 6),
            {"b": torch.randn(1, 1, 3)},
            placeholder_token_ids={"a": 32},
        )


def test_rejects_placeholder_ids_for_unregistered_projectors():
    registry = MultimodalProjector({"a": Mlp2xGeluProjector(3, 8, 6)})
    with pytest.raises(ValueError, match="unregistered"):
        registry(
            torch.tensor([[32, 1]]),
            torch.zeros(1, 2, 6),
            {"a": torch.randn(1, 1, 3)},
            placeholder_token_ids={"a": 32, "b": 33},
        )


def test_rejects_shared_placeholder_id_across_projectors():
    registry = MultimodalProjector({"a": Mlp2xGeluProjector(3, 8, 6), "b": Mlp2xGeluProjector(3, 8, 6)})
    with pytest.raises(ValueError, match="distinct"):
        registry(
            torch.tensor([[32, 32]]),
            torch.zeros(1, 2, 6),
            {"a": torch.randn(1, 1, 3), "b": torch.randn(1, 1, 3)},
            placeholder_token_ids={"a": 32, "b": 32},
        )


def test_rejects_orphan_placeholder_slots():
    # The prompt carries slots for a registered projector that received no
    # features; leaving them as raw placeholder embeddings must fail loudly.
    registry = MultimodalProjector({"a": Mlp2xGeluProjector(3, 8, 6), "b": Mlp2xGeluProjector(3, 8, 6)})
    with pytest.raises(ValueError, match="no features"):
        registry(
            torch.tensor([[32, 33, 1]]),
            torch.zeros(1, 3, 6),
            {"a": torch.randn(1, 1, 3)},
            placeholder_token_ids={"a": 32, "b": 33},
        )


def test_misrouted_explicit_index_fails_the_per_projector_placeholder_check():
    registry = MultimodalProjector({"a": Mlp2xGeluProjector(3, 8, 6), "b": Mlp2xGeluProjector(3, 8, 6)})
    with pytest.raises(ValueError, match="non-placeholder"):
        registry(
            torch.tensor([[32, 33]]),
            torch.zeros(1, 2, 6),
            {"a": torch.randn(1, 1, 3), "b": torch.randn(1, 1, 3)},
            placeholder_token_ids={"a": 32, "b": 33},
            # a's index points at b's slot.
            token_indices_by_projector={"a": torch.tensor([[1]]), "b": torch.tensor([[1]])},
        )


class _Embedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(64, 2)

    def get_input_embeddings(self):
        return self.embed


class _DummyHost(MultimodalInputMixin, nn.Module):
    """Minimal host: config + a text embedder, as the mixin contract requires."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            llm_config=SimpleNamespace(hidden_size=2),
            mm_projectors=[{"name": "tokens", "kind": "mlp2x_gelu", "mm_hidden_size": 3, "hidden_size": 8}],
            mm_placeholder_token_ids={"tokens": 32},
        )
        self.linear = nn.Linear(2, 2, bias=False)
        self.language_model = _Embedder()
        self._init_mm_projector(self.config)


def test_projector_is_project_state_not_a_registered_submodule():
    model = _DummyHost()
    named = {name for name, _ in model.named_parameters()}
    assert "mm_projector" not in named
    assert not any(name.startswith("mm_projector.") for name in model.state_dict())
    assert set(model.state_dict()) == {"linear.weight", "language_model.embed.weight"}
    # Still reachable and parameter-owning.
    projector_parameters = list(model.mm_projector.parameters())
    assert projector_parameters
    assert all(parameter.requires_grad for parameter in projector_parameters)


def test_projector_requires_grad_is_owned_by_project_code():
    model = _DummyHost()
    projector = model.mm_projector
    assert any(parameter.requires_grad for parameter in projector.parameters())
    # Freezing the base must not touch the project-owned projector.
    model.requires_grad_(False)
    assert not model.linear.weight.requires_grad
    assert all(parameter.requires_grad for parameter in projector.parameters())
    # A dedicated optimizer can own the projector independently of the base.
    projector.requires_grad_(True)
    optimizer = torch.optim.AdamW(projector.parameters(), lr=1.0e-3)
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["params"] == list(projector.parameters())


def test_mixin_registers_the_projector_under_module_ownership():
    model = _DummyHost()
    model._init_mm_projector(model.config, ownership="module")
    named = {name for name, _ in model.named_parameters()}
    assert any(name.startswith("mm_projector.") for name in named)
    # The sidecar is replaced, not duplicated: exactly one registry exists.
    assert sum("mm_projector" in name for name in model.state_dict()) == len(list(model.mm_projector.state_dict()))


def test_mixin_rejects_unknown_ownership_modes():
    model = _DummyHost()
    with pytest.raises(ValueError, match="unknown projector ownership"):
        model._init_mm_projector(model.config, ownership="hybrid")


def test_prepare_inputs_embeds_end_to_end():
    model = _DummyHost()
    input_ids = torch.tensor([[32, 32, 7]])
    features = torch.randn(1, 2, 3)
    embeds = model._prepare_inputs_embeds(input_ids, None, {"tokens": features}, {})
    soft = model.mm_projector.projectors["tokens"](features)
    torch.testing.assert_close(embeds[0, :2], soft[0])
    torch.testing.assert_close(embeds[0, 2], model.language_model.embed.weight[7])


def test_init_validates_the_placeholder_id_mapping_at_construction():
    model = _DummyHost()

    model.config.mm_placeholder_token_ids = None
    with pytest.raises(ValueError, match="mm_placeholder_token_ids"):
        model._init_mm_projector(model.config)

    model.config.mm_placeholder_token_ids = {"tokens": 32, "other": 33}
    with pytest.raises(ValueError, match="registered projector names"):
        model._init_mm_projector(model.config)

    model.config.mm_placeholder_token_ids = {"tokens": 32}
    model._init_mm_projector(model.config)  # recovers


def test_init_rejects_placeholder_ids_shared_across_projectors():
    model = _DummyHost()
    model.config.mm_projectors = [
        {"name": "one", "kind": "mlp2x_gelu", "mm_hidden_size": 3, "hidden_size": 8},
        {"name": "two", "kind": "mlp2x_gelu", "mm_hidden_size": 3, "hidden_size": 8},
    ]
    model.config.mm_placeholder_token_ids = {"one": 32, "two": 32}
    with pytest.raises(ValueError, match="distinct"):
        model._init_mm_projector(model.config)


def test_projector_materializes_from_meta_to_the_model_device():
    model = _DummyHost()
    # AutoModel constructs the architecture on meta (L-2); the sidecar
    # projector is project state and is left on meta.
    model.mm_projector.to_empty(device="meta")
    assert next(model.mm_projector.parameters()).is_meta
    materialize_mm_projector(model, dtype=torch.bfloat16)
    parameter = next(model.mm_projector.parameters())
    assert not parameter.is_meta
    assert parameter.device == model.linear.weight.device
    assert parameter.dtype == torch.bfloat16
    assert torch.isfinite(parameter).all()


def test_extract_mm_kwargs_pops_prefixed_fields():
    kwargs = {
        "mm_features__a": torch.zeros(1),
        "mm_token_indices__a": torch.zeros(1),
        "input_ids": torch.zeros(1),
    }
    features, indices = MultimodalInputMixin._extract_mm_kwargs(None, kwargs)
    assert set(features) == {"a"} and set(indices) == {"a"}
    assert set(kwargs) == {"input_ids"}


class TestModuleOwnership:
    """projector_ownership=\"module\": a registered, FSDP2-visible child.

    In this mode the projector is an in-tree child that
    FSDP2 shards and its TP plumbing replicates, so the registry must behave
    like an ordinary submodule and fail closed on placement mismatches.
    """

    def test_default_ownership_is_sidecar(self):
        registry = MultimodalProjector({"tokens": Mlp2xGeluProjector(3, 8, 6)})
        assert registry.projector_ownership == OWNERSHIP_SIDECAR

    def test_unknown_ownership_fails_closed(self):
        with pytest.raises(ValueError, match="unknown projector_ownership"):
            MultimodalProjector({"tokens": Mlp2xGeluProjector(3, 8, 6)}, projector_ownership="hybrid")

    def test_registry_is_a_registered_child_with_one_entry_per_projector(self):
        class _Host(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(6, 6, bias=False)
                self.mm_projector = MultimodalProjector(
                    {"tokens": Mlp2xGeluProjector(3, 8, 6)}, projector_ownership="module"
                )

        host = _Host()
        named = {name for name, _ in host.named_parameters()}
        assert "mm_projector.projectors.tokens.network.1.weight" in named
        state_keys = set(host.state_dict())
        assert any(key.startswith("mm_projector.projectors.tokens.") for key in state_keys)
        # Freezing and optimizers see it through the host, unlike sidecar mode.
        host.requires_grad_(False)
        assert not any(parameter.requires_grad for parameter in host.mm_projector.parameters())

    def test_module_mode_scatters_like_sidecar(self):
        sidecar = MultimodalProjector({"tokens": Mlp2xGeluProjector(3, 8, 6)})
        module = MultimodalProjector({"tokens": Mlp2xGeluProjector(3, 8, 6)}, projector_ownership="module")
        module.load_state_dict(sidecar.state_dict())
        ids = torch.tensor([[32, 1, 32, 2]])
        text = torch.randn(1, 4, 6)
        features = torch.randn(1, 2, 3)
        expected = sidecar(ids, text, {"tokens": features}, placeholder_token_ids={"tokens": 32})
        result = module(ids, text, {"tokens": features}, placeholder_token_ids={"tokens": 32})
        torch.testing.assert_close(result, expected)

    def test_module_mode_fails_closed_on_device_mismatch(self):
        registry = MultimodalProjector({"tokens": Mlp2xGeluProjector(3, 8, 6)}, projector_ownership="module")
        registry.to(device="meta")
        with pytest.raises(ValueError, match="device"):
            registry(
                torch.tensor([[32]]),
                torch.zeros(1, 1, 6),
                {"tokens": torch.randn(1, 1, 3)},
                placeholder_token_ids={"tokens": 32},
            )

    def test_module_mode_meta_lifecycle_matches_automodel(self):
        # AutoModel constructs the tree on meta, materializes with to_empty,
        # and re-initializes via initialize_weights -> reset_parameters.
        with torch.device("meta"):
            registry = MultimodalProjector.from_config(
                [{"name": "tokens", "kind": "mlp2x_gelu", "mm_hidden_size": 3, "hidden_size": 8}],
                output_size=6,
                projector_ownership="module",
            )
        assert registry.projector_ownership == "module"
        assert next(registry.parameters()).is_meta
        registry.to_empty(device="cpu")
        registry.reset_parameters()
        parameter = next(registry.parameters())
        assert not parameter.is_meta
        assert torch.isfinite(parameter).all()

    def test_from_config_rejects_duplicate_names_in_module_mode(self):
        with pytest.raises(ValueError, match="duplicate projector name"):
            MultimodalProjector.from_config(
                [
                    {"name": "tokens", "kind": "mlp2x_gelu", "mm_hidden_size": 3},
                    {"name": "tokens", "kind": "mlp2x_gelu", "mm_hidden_size": 3},
                ],
                output_size=6,
                projector_ownership="module",
            )
