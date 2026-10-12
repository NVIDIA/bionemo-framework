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

"""build_multimodal_host: decorating a dynamically resolved causal-LM class.

The fake host below stands in for a Hub-resolved remote-code class: an opaque
constructor, a forward accepting ``inputs_embeds`` and ``**kwargs``, and a
generation method. The decoration must preserve every base parameter FQN and
method, add only the projector, and delegate exactly one ``super().forward``.
"""

from __future__ import annotations

import inspect
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

from nemotron_stitch.automodel.model import (
    build_cp1_packed_multimodal_host,
    build_multimodal_host,
)

CONFIG = SimpleNamespace(
    hidden_size=8,
    mm_projectors=[{"name": "tokens", "kind": "mlp2x_gelu", "mm_hidden_size": 4, "hidden_size": 8}],
    mm_placeholder_token_ids={"tokens": 32},
)


class _TextBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(64, 8)
        self.blocks = nn.Linear(8, 8)

    def get_input_embeddings(self):
        return self.embed_tokens


class _FakeRemoteHost(nn.Module):
    """A stand-in for an AutoModel-resolved HF *ForCausalLM class."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.language_model = _TextBackbone()
        self.lm_head = nn.Linear(8, 8)
        self.forward_calls = 0

    def forward(self, input_ids=None, *, inputs_embeds=None, **kwargs):
        self.forward_calls += 1
        self.forward_kwargs = kwargs
        embeds = inputs_embeds if inputs_embeds is not None else self.language_model.embed_tokens(input_ids)
        return self.lm_head(self.language_model.blocks(embeds))

    def generate(self, *args, **kwargs):
        return "generated"


def _build(ownership="module"):
    return build_multimodal_host(_FakeRemoteHost, projector_ownership=ownership, architecture="FakeMultimodal")


def test_decoration_preserves_base_state_dict_keys_and_adds_only_the_projector():
    base_keys = set(_FakeRemoteHost(CONFIG).state_dict())
    host = _build("module")(CONFIG)
    host_keys = set(host.state_dict())
    assert base_keys <= host_keys
    added = host_keys - base_keys
    assert added and all(key.startswith("mm_projector.") for key in added)


def test_sidecar_decoration_adds_no_state_dict_keys():
    host = _build("sidecar")(CONFIG)
    assert set(host.state_dict()) == set(_FakeRemoteHost(CONFIG).state_dict())
    assert list(host.mm_projector.parameters())


def test_forward_delegates_exactly_once_with_inputs_embeds():
    host = _build("module")(CONFIG)
    input_ids = torch.tensor([[32, 32, 7]])
    features = torch.randn(1, 2, 4)
    out = host(input_ids, mm_features__tokens=features)
    assert host.forward_calls == 1
    soft = host.mm_projector.projectors["tokens"](features)
    expected = host.lm_head(
        host.language_model.blocks(
            torch.cat([soft[0], host.language_model.embed_tokens.weight[7].unsqueeze(0)]).unsqueeze(0)
        )
    )
    torch.testing.assert_close(out, expected)


def test_forward_without_features_delegates_unchanged():
    host = _build("module")(CONFIG)
    plain = host(torch.tensor([[1, 2, 3]]))
    reference = _FakeRemoteHost(CONFIG)
    reference.load_state_dict({k: v for k, v in host.state_dict().items() if not k.startswith("mm_projector.")})
    torch.testing.assert_close(plain, reference(torch.tensor([[1, 2, 3]])))
    with pytest.raises(ValueError, match="require matching mm_features"):
        host(torch.tensor([[1]]), mm_token_indices__tokens=torch.tensor([[0]]))


def test_forward_preserves_fused_linear_ce_logits_control():
    """Keep AutoModel's fused-CE capability visible through the host bridge."""
    host = _build("module")(CONFIG)
    assert "logits_to_keep" in inspect.signature(host.forward).parameters
    host(torch.tensor([[1, 2, 3]]), logits_to_keep=1)
    assert host.forward_kwargs["logits_to_keep"] == 1


def test_generation_and_base_methods_survive_decoration():
    host = _build()(CONFIG)
    assert host.generate() == "generated"
    assert type(host).__name__ == "FakeMultimodal"
    assert isinstance(host, _FakeRemoteHost)


def test_module_host_initialize_weights_resets_the_projector():
    class _WithInit(_FakeRemoteHost):
        def initialize_weights(self, buffer_device=None):
            nn.init.zeros_(self.lm_head.weight)

    host_cls = build_multimodal_host(_WithInit, projector_ownership="module")
    host = host_cls(CONFIG)
    host.initialize_weights()
    assert not host.lm_head.weight.any()
    # The projector was re-initialized, not zeroed with the base or left stale.
    after = host.mm_projector.state_dict()
    assert all(torch.isfinite(v).all() for v in after.values())


def test_factory_rejects_hosts_without_the_embedding_boundary():
    class NoEmbeds(nn.Module):
        def forward(self, input_ids=None, **kwargs):
            return input_ids

    with pytest.raises(TypeError, match="inputs_embeds"):
        build_multimodal_host(NoEmbeds)

    class NoKwargs(nn.Module):
        def forward(self, input_ids=None, inputs_embeds=None):
            return input_ids

    with pytest.raises(TypeError, match=r"\*\*kwargs"):
        build_multimodal_host(NoKwargs)

    with pytest.raises(ValueError, match="unknown projector ownership"):
        build_multimodal_host(_FakeRemoteHost, projector_ownership="hybrid")


def test_factory_resolves_width_through_the_single_resolver():
    config = SimpleNamespace(
        text_config=SimpleNamespace(hidden_size=8),
        mm_projectors=CONFIG.mm_projectors,
        mm_placeholder_token_ids=CONFIG.mm_placeholder_token_ids,
    )
    host = _build()(config)
    assert host.mm_projector.projectors["tokens"].output_size == 8
    conflicting = SimpleNamespace(
        hidden_size=8,
        text_config=SimpleNamespace(hidden_size=16),
        mm_projectors=CONFIG.mm_projectors,
        mm_placeholder_token_ids=CONFIG.mm_placeholder_token_ids,
    )
    with pytest.raises(ValueError, match="conflicting LM hidden sizes"):
        _build()(conflicting)


def test_ownership_can_come_from_the_model_config():
    host_cls = build_multimodal_host(_FakeRemoteHost)  # ownership deferred to config
    sidecar = host_cls(CONFIG)
    assert sidecar.mm_projector.projector_ownership == "sidecar"
    assert "mm_projector" not in {n for n, _ in sidecar.named_parameters()}
    module_config = SimpleNamespace(**{**vars(CONFIG), "mm_projector_ownership": "module"})
    modular = host_cls(module_config)
    assert any(n.startswith("mm_projector.") for n, _ in modular.named_parameters())


def test_module_initialize_weights_override_is_attached_when_base_defines_it():
    class _WithInit(_FakeRemoteHost):
        def initialize_weights(self, buffer_device=None):
            pass

    host_cls = build_multimodal_host(_WithInit)
    host = host_cls(CONFIG)  # sidecar: reset is a no-op guard, not required
    host.initialize_weights()


def test_factory_consumes_mm_config_keys_from_constructor_kwargs():
    # AutoModel's custom-model path passes non-config kwargs to the constructor.
    bare = SimpleNamespace(hidden_size=8)
    host = _build()(bare, mm_projectors=CONFIG.mm_projectors, mm_placeholder_token_ids={"tokens": 32})
    assert bare.mm_projectors == CONFIG.mm_projectors
    assert list(host.mm_projector.projectors) == ["tokens"]


def test_factory_fails_closed_on_config_constructor_conflict():
    host_cls = build_multimodal_host(_FakeRemoteHost)
    config = SimpleNamespace(**{**vars(CONFIG), "mm_projector_ownership": "module"})
    with pytest.raises(ValueError, match="conflicting mm_projector_ownership"):
        host_cls(config, mm_projector_ownership="sidecar")


def test_module_host_without_base_initialize_weights_still_initializes_projector():
    host = _build("module")(CONFIG)  # _FakeRemoteHost has no initialize_weights
    for p in host.mm_projector.parameters():
        nn.init.constant_(p, 99.0)
    host.initialize_weights()
    assert all(torch.isfinite(p).all() and not (p == 99.0).all() for p in host.mm_projector.parameters())


def test_module_host_excludes_project_state_from_native_base_checkpoint_adapter():
    class _StateAdapter:
        def convert_single_tensor_to_hf(self, fqn, tensor, **_kwargs):
            return [(fqn, tensor)]

        def from_hf(self, state_dict, *_args, **_kwargs):
            return state_dict

    class _NativeStyleHost(_FakeRemoteHost):
        def __init__(self, config):
            super().__init__(config)
            self.state_dict_adapter = _StateAdapter()

    host = build_multimodal_host(
        _NativeStyleHost,
        projector_ownership="module",
        additional_parameter_prefixes=("marker_delta",),
    )(CONFIG)
    adapter = host.state_dict_adapter

    assert adapter.convert_single_tensor_to_hf("mm_projector.projectors.tokens.weight", torch.ones(1)) == []
    assert adapter.convert_single_tensor_to_hf("marker_delta", torch.ones(1)) == []
    assert adapter.convert_single_tensor_to_hf("model.weight", torch.ones(1)) != []
    assert adapter.config is CONFIG


def test_sidecar_host_excludes_frozen_extra_state_from_refit_inventory():
    class _StateAdapter:
        def convert_single_tensor_to_hf(self, fqn, tensor, **_kwargs):
            return [(fqn, tensor)]

        def from_hf(self, state_dict, *_args, **_kwargs):
            return state_dict

    class _NativeStyleHost(_FakeRemoteHost):
        def __init__(self, config):
            super().__init__(config)
            self.state_dict_adapter = _StateAdapter()

    host = build_multimodal_host(
        _NativeStyleHost,
        projector_ownership="sidecar",
        additional_parameter_prefixes=("learned_start", "learned_end"),
    )(CONFIG)
    state = {
        "model.weight": torch.ones(1),
        "learned_start": torch.ones(4),
        "learned_end": torch.ones(4),
    }
    inventory = {
        converted_name
        for name, tensor in state.items()
        for converted_name, _ in host.state_dict_adapter.convert_single_tensor_to_hf(name, tensor)
    }

    assert inventory == {"model.weight"}


def test_factory_rejects_an_empty_additional_parameter_prefix():
    with pytest.raises(ValueError, match="non-empty strings"):
        build_multimodal_host(_FakeRemoteHost, additional_parameter_prefixes=("",))


def test_factory_filters_loading_kwargs_for_a_strict_base_ctor():
    host = _build()(CONFIG, trust_remote_code=True, local_files_only=True, revision="abc")
    assert isinstance(host, _FakeRemoteHost)


def test_factory_forwards_kwargs_when_base_accepts_them():
    class _OpenBase(_FakeRemoteHost):
        def __init__(self, config, **kwargs):
            super().__init__(config)
            self.seen_kwargs = kwargs

    host = build_multimodal_host(_OpenBase)(CONFIG, trust_remote_code=True)
    assert host.seen_kwargs == {"trust_remote_code": True}


def test_mm_projector_trainable_false_freezes_at_construction():
    config = SimpleNamespace(**{**vars(CONFIG), "mm_projector_trainable": False})
    host = _build()(config)
    assert not any(p.requires_grad for p in host.mm_projector.parameters())


def test_transformers_generic_initialize_weights_is_not_delegated():
    class _TransformersStyleBase(_FakeRemoteHost):
        def initialize_weights(self, buffer_device=None):
            raise AssertionError("generic transformers initialize_weights must not run")

    _TransformersStyleBase.initialize_weights.__module__ = "transformers.modeling_utils"
    host_cls = build_multimodal_host(_TransformersStyleBase, projector_ownership="module")
    host = host_cls(CONFIG)
    host.initialize_weights()  # only the projector reset runs


def test_cp1_packed_host_routes_indexed_mask_to_automodel_state(monkeypatch):
    calls = {}
    modules = {
        name: ModuleType(name)
        for name in (
            "nemo_automodel",
            "nemo_automodel.components",
            "nemo_automodel.components.distributed",
            "nemo_automodel.components.distributed.blockdiag_cp",
            "nemo_automodel.components.models",
            "nemo_automodel.components.models.common",
            "nemo_automodel.components.models.common.packing",
        )
    }
    blockdiag_cp = modules["nemo_automodel.components.distributed.blockdiag_cp"]
    blockdiag_cp.configure_cp_varlen = lambda *, attn_backend: calls.update(configured=attn_backend)
    blockdiag_cp.attach_cp1_packed_varlen_hooks = lambda model: calls.update(attached=model)
    blockdiag_cp.disable_cp1_packed_varlen = lambda: calls.update(disabled=True)
    blockdiag_cp.enable_cp1_packed_varlen = lambda ids, backend: calls.update(enabled=(ids, backend))
    blockdiag_cp.cp1_packed_varlen_backend = lambda: "flash"
    modules["nemo_automodel.components.models.common.packing"].is_indexed_packed_mask = (
        lambda value: isinstance(value, torch.Tensor)
        and value.dtype == torch.long
        and value.numel() > 0
        and value.max().item() > 1
    )
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    class _PackedBase(_FakeRemoteHost):
        def forward(
            self,
            input_ids=None,
            *,
            inputs_embeds=None,
            attention_mask=None,
            logits_to_keep=0,
            **kwargs,
        ):
            self.seen_attention_mask = attention_mask
            self.seen_packed_seq_ids = kwargs.get("_packed_seq_ids")
            self.seen_logits_to_keep = logits_to_keep
            return super().forward(input_ids, inputs_embeds=inputs_embeds, **kwargs)

    host_cls = build_cp1_packed_multimodal_host(
        _PackedBase,
        architecture="FakePackedMultimodal",
        projector_ownership="module",
    )
    host = host_cls(CONFIG)
    indexed_mask = torch.tensor([[1, 1, 2, 2, 0]], dtype=torch.long)
    host(
        torch.tensor([[1, 2, 3, 4, 5]]),
        attention_mask=indexed_mask,
        logits_to_keep=torch.tensor([1, 3]),
    )

    assert calls["configured"] == "flash"
    assert calls["attached"] is host
    assert calls["disabled"] is True
    assert calls["enabled"][1] == "flash"
    torch.testing.assert_close(calls["enabled"][0], indexed_mask)
    assert host.seen_attention_mask is None
    torch.testing.assert_close(host.seen_packed_seq_ids, indexed_mask)
    torch.testing.assert_close(host.seen_logits_to_keep, torch.tensor([1, 3]))
