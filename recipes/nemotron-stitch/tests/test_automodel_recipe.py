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

"""Projector lifecycle and optimizer wiring against AutoModel recipe doubles."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nemotron_stitch.automodel.optim import ProjectorAdamWConfig
from nemotron_stitch.automodel.recipe import (
    ProjectorEvaluationRecipe,
    ProjectorFinetuneRecipe,
    ProjectorRecipeMixin,
    ProjectorSidecarState,
    plain_mapping,
    require_one_rank,
)
from nemotron_stitch.projector.artifact import (
    load_projector_artifact,
    read_projector_artifact,
    save_projector_artifact,
)
from nemotron_stitch.projector.multimodal import MultimodalProjector
from nemotron_stitch.projector.trainability import TrainabilityPolicy, collect_extra_state
from nemotron_stitch.testing.recipe import CheckpointTrackerFake

PROJECTOR_CONFIGS = [{"name": "tokens", "kind": "mlp2x_gelu", "mm_hidden_size": 3, "hidden_size": 8}]


class _Cfg(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


class _FakeDistEnv:
    def __init__(self, world_size=1, device=None):
        self.world_size = world_size
        self.device = device if device is not None else torch.device("cpu")


class _FakeBaseRecipe:
    """The slice of the AutoModel recipe surface the mixin reads."""

    def __init__(self, cfg, *, world_size=1, pp_enabled=False, cp_group_size=1):
        self.cfg = cfg
        self.dist_env = _FakeDistEnv(world_size=world_size)
        self.model_parts = []
        self.pp_enabled = pp_enabled
        self._cp_group_size = cp_group_size

    def _get_cp_group_size(self):
        return self._cp_group_size

    def setup(self):
        self.base_setup_ran = True
        return "setup-result"

    def run_train_validation_loop(self):
        self.base_loop_ran = True
        return "loop-result"


class _Recipe(ProjectorRecipeMixin, _FakeBaseRecipe):
    pass


_REGISTRATIONS = 0


def _register_application_models():
    global _REGISTRATIONS
    _REGISTRATIONS += 1


def _model(*, meta_projector=True):
    model = nn.Linear(4, 4)
    model.config = SimpleNamespace(mm_projectors=PROJECTOR_CONFIGS)
    projector = MultimodalProjector.from_config(PROJECTOR_CONFIGS, output_size=4)
    if meta_projector:
        projector = projector.to(torch.device("meta"))
    object.__setattr__(model, "mm_projector", projector)
    return model


def _model_with_extra(*, meta_projector=True):
    """A host also carrying an extra-family parameter (a learned marker)."""
    model = _model(meta_projector=meta_projector)
    model.marker_embed_delta = nn.Parameter(torch.zeros(2, 4))
    return model


def _cfg(**projector):
    return _Cfg(model=_Cfg(mm_training_stage=1), projector=_Cfg(projector))


@pytest.mark.parametrize("train_decoder", [False, True])
def test_shared_finetune_recipe_applies_application_trainability(monkeypatch, train_decoder):
    global _REGISTRATIONS
    _REGISTRATIONS = 0
    monkeypatch.setattr(ProjectorRecipeMixin, "setup", lambda self: "setup-result")
    optimizer = ProjectorAdamWConfig()
    recipe = object.__new__(ProjectorFinetuneRecipe)
    recipe.cfg = _Cfg(
        model=_Cfg(mm_training_stage=2),
        optimizer=optimizer,
        projector=_Cfg(
            model_registry_callback=f"{__name__}._register_application_models",
            train_projector=False,
            trainability_manifest="trainability.json",
            train_decoder=train_decoder,
            trainability_decoder_patterns=["backbone.", "lm_head."] if train_decoder else [],
        ),
    )
    recipe.model_parts = [SimpleNamespace(mm_projector=SimpleNamespace(projector_ownership="module"))]

    assert recipe.setup() == "setup-result"
    assert _REGISTRATIONS == 1
    assert optimizer.stage == 2
    assert optimizer.train_projector is False
    assert optimizer.manifest_path == "trainability.json"
    assert optimizer.train_decoder is train_decoder
    assert optimizer.policy == TrainabilityPolicy(
        projector_patterns=("mm_projector",),
        decoder_patterns=("backbone.", "lm_head.") if train_decoder else (),
    )
    assert optimizer.train_extra is False


def test_shared_finetune_recipe_wires_the_extra_family(monkeypatch):
    monkeypatch.setattr(ProjectorRecipeMixin, "setup", lambda self: "setup-result")
    optimizer = ProjectorAdamWConfig()
    recipe = object.__new__(ProjectorFinetuneRecipe)
    recipe.cfg = _Cfg(
        model=_Cfg(mm_training_stage=1),
        optimizer=optimizer,
        projector=_Cfg(
            model_registry_callback=f"{__name__}._register_application_models",
            train_projector=True,
            trainability_extra_patterns=["marker_embed_delta"],
            train_extra=True,
        ),
    )
    recipe.model_parts = [SimpleNamespace(mm_projector=SimpleNamespace(projector_ownership="module"))]

    assert recipe.setup() == "setup-result"
    assert optimizer.policy == TrainabilityPolicy(
        projector_patterns=("mm_projector",), extra_patterns=("marker_embed_delta",)
    )
    assert optimizer.train_extra is True


class _Closable:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def test_evaluation_recipe_runs_validation_without_training(tmp_path):
    output = tmp_path / "evaluation.json"
    train_logger = _Closable()
    val_logger = _Closable()
    recipe = object.__new__(ProjectorEvaluationRecipe)
    # AutoModel's ConfigNode accepts dotted paths in get(); model that exact
    # lookup without importing the training dependency into the unit suite.
    recipe.cfg = _Cfg({"evaluation.output_path": str(output)})
    recipe.dist_env = SimpleNamespace(is_main=True)
    recipe.val_dataloaders = {"default": [object(), object()]}
    recipe.metric_logger_train = train_logger
    recipe.metric_logger_valid = {"default": val_logger}
    recipe.partial_cuda_graph_manager = None
    recipe._run_validation_epoch = lambda loader: SimpleNamespace(
        metrics={"val_loss": torch.tensor(1.25), "num_label_tokens": 17}
    )
    recipe.log_val_metrics = lambda *args: None
    recipe._finalize_and_close_checkpointer = lambda: None

    result = recipe.run_train_validation_loop()

    assert result == {"default": {"val_loss": 1.25, "num_label_tokens": 17, "num_batches": 2}}
    payload = json.loads(output.read_text())
    assert payload["evaluation_only"] is True
    assert payload["optimizer_steps"] == 0
    assert payload["validation"] == result
    assert train_logger.closed and val_logger.closed


def test_setup_materializes_the_sidecar_projector_after_base_setup():
    recipe = _Recipe(_cfg())
    recipe.model_parts = [_model()]

    assert recipe.setup() == "setup-result"
    assert recipe.base_setup_ran
    parameter = next(recipe.model_parts[0].mm_projector.parameters())
    assert parameter.device == torch.device("cpu")
    assert not parameter.is_meta


def test_setup_attaches_the_tracker_wrapper_in_sidecar_mode_only():
    sidecar = _Recipe(_cfg())
    sidecar.model_parts = [_model()]
    sidecar.setup()
    assert isinstance(sidecar.projector_sidecar_state, ProjectorSidecarState)
    assert not isinstance(sidecar.projector_sidecar_state, nn.Module)

    module = _ModuleRecipe(_cfg())
    module.model_parts = [_module_model()]
    module.setup()
    assert not hasattr(module, "projector_sidecar_state")


def test_setup_fails_closed_above_one_rank():
    recipe = _Recipe(_cfg(), world_size=2)
    recipe.model_parts = [_model()]
    with pytest.raises(NotImplementedError, match="exactly one rank"):
        recipe.setup()


def test_setup_fails_closed_for_pipeline_parallelism():
    recipe = _Recipe(_cfg(), pp_enabled=True)
    recipe.model_parts = [_model()]
    with pytest.raises(NotImplementedError, match="pipeline parallelism"):
        recipe.setup()


def test_setup_fails_closed_for_context_parallelism():
    recipe = _Recipe(_cfg(), cp_group_size=2)
    recipe.model_parts = [_model()]
    with pytest.raises(NotImplementedError, match="context parallelism"):
        recipe.setup()


def test_setup_requires_a_constructed_projector():
    recipe = _Recipe(_cfg())
    recipe.model_parts = [nn.Linear(4, 4)]
    with pytest.raises(TypeError, match="multimodal model was not constructed"):
        recipe.setup()


def test_setup_moves_stranded_buffers_to_the_model_device():
    # A meta device stands in for the training device: copying a real CPU
    # buffer into meta is allowed (copying out of meta is not), and the sweep
    # only needs a device the stranded buffer is not already on.
    recipe = _Recipe(_cfg())
    recipe.dist_env.device = torch.device("meta")
    model = _model()
    model.register_buffer("stranded", torch.zeros(2), persistent=False)
    recipe.model_parts = [model]

    recipe.setup()

    assert model.stranded.device == torch.device("meta")


def test_setup_moves_stranded_buffers_in_module_mode():
    # The PEFT post-shard init load strands custom MoE buffers on CPU with
    # module ownership too (examples/llava Lightning SFT finding); the sweep
    # runs before either ownership branch.
    from nemotron_stitch.contracts import OWNERSHIP_MODULE

    recipe = _Recipe(_cfg())
    recipe.projector_ownership = OWNERSHIP_MODULE
    recipe.dist_env.device = torch.device("meta")
    model = _model(meta_projector=False)
    model.mm_projector = model.mm_projector  # register as a child, not an attribute
    model.register_buffer("stranded", torch.zeros(2), persistent=False)
    recipe.model_parts = [model]

    recipe.setup()

    assert model.stranded.device == torch.device("meta")


def test_setup_loads_the_initial_sidecar(tmp_path):
    reference = _model(meta_projector=False)
    save_projector_artifact(reference, tmp_path / "sidecar", provenance={"origin": "unit-test"}, stage=1)
    recipe = _Recipe(_cfg(initial_artifact=str(tmp_path / "sidecar"), expected_provenance={"origin": "unit-test"}))
    recipe.model_parts = [_model()]

    recipe.setup()

    expected = reference.mm_projector.state_dict()
    loaded = recipe.model_parts[0].mm_projector.state_dict()
    assert loaded.keys() == expected.keys()
    for name, value in expected.items():
        assert torch.equal(loaded[name], value), name


def test_setup_loads_an_initial_sidecar_without_a_provenance_lock(tmp_path):
    save_projector_artifact(_model(meta_projector=False), tmp_path / "sidecar", provenance={"origin": "x"}, stage=1)
    recipe = _Recipe(_cfg(initial_artifact=str(tmp_path / "sidecar")))
    recipe.model_parts = [_model()]
    recipe.setup()


def test_sidecar_stage2_loads_the_stage1_bridge(tmp_path):
    reference = _model(meta_projector=False)
    save_projector_artifact(reference, tmp_path / "stage1", provenance={"origin": "unit-test"}, stage=1)
    recipe = _Recipe(
        _Cfg(
            model=_Cfg(mm_training_stage=2),
            projector=_Cfg(stage1_artifact=str(tmp_path / "stage1"), expected_provenance={"origin": "unit-test"}),
        )
    )
    recipe.model_parts = [_model()]

    recipe.setup()

    for name, value in recipe.model_parts[0].mm_projector.state_dict().items():
        assert torch.equal(value, reference.mm_projector.state_dict()[name]), name


def test_sidecar_export_writes_the_package_manifest(tmp_path):
    artifact_dir = tmp_path / "export"
    recipe = _Recipe(_cfg(artifact_dir=str(artifact_dir), provenance={"origin": "unit-test"}))
    recipe.model_parts = [_model(meta_projector=False)]

    assert recipe.run_train_validation_loop() == "loop-result"

    manifest = json.loads((artifact_dir / "mm-projector-manifest.json").read_text())
    assert manifest["schema"] == "nemotron-add-modality/mm-projector-manifest"
    assert manifest["stage"] == 1
    # Round-trip: the export loads back bit-identically.
    target = _model(meta_projector=False)
    load_projector_artifact(target, artifact_dir, expected={"origin": "unit-test"})
    for name, value in recipe.model_parts[0].mm_projector.state_dict().items():
        assert torch.equal(target.mm_projector.state_dict()[name], value), name


def test_sidecar_export_carries_the_extra_family(tmp_path):
    artifact_dir = tmp_path / "export"
    recipe = _Recipe(
        _cfg(
            artifact_dir=str(artifact_dir),
            provenance={"origin": "unit-test"},
            trainability_extra_patterns=["marker_embed_delta"],
        )
    )
    recipe.model_parts = [_model_with_extra(meta_projector=False)]
    with torch.no_grad():
        recipe.model_parts[0].marker_embed_delta.fill_(2.0)

    assert recipe.run_train_validation_loop() == "loop-result"

    manifest = json.loads((artifact_dir / "mm-projector-manifest.json").read_text())
    assert manifest["extra_state"]["marker_embed_delta"]["shape"] == [2, 4]
    # Round-trip: the export restores both projector and markers.
    target = _model_with_extra(meta_projector=False)
    load_projector_artifact(
        target, artifact_dir, expected={"origin": "unit-test"}, extra_patterns=("marker_embed_delta",)
    )
    torch.testing.assert_close(target.marker_embed_delta, torch.full((2, 4), 2.0))
    for name, value in recipe.model_parts[0].mm_projector.state_dict().items():
        assert torch.equal(target.mm_projector.state_dict()[name], value), name


def test_sidecar_export_fails_closed_when_extra_patterns_match_nothing(tmp_path):
    recipe = _Recipe(
        _cfg(
            artifact_dir=str(tmp_path / "export"),
            provenance={"origin": "unit-test"},
            trainability_extra_patterns=["ghost"],
        )
    )
    recipe.model_parts = [_model(meta_projector=False)]
    with pytest.raises(ValueError, match="matched no parameters"):
        recipe.run_train_validation_loop()


def test_sidecar_warm_start_restores_the_extra_family(tmp_path):
    reference = _model_with_extra(meta_projector=False)
    with torch.no_grad():
        reference.marker_embed_delta.fill_(4.0)
    save_projector_artifact(
        reference,
        tmp_path / "sidecar",
        provenance={"origin": "unit-test"},
        stage=1,
        extra_state=collect_extra_state(reference, TrainabilityPolicy(extra_patterns=("marker_embed_delta",))),
    )
    recipe = _Recipe(
        _cfg(
            initial_artifact=str(tmp_path / "sidecar"),
            expected_provenance={"origin": "unit-test"},
            trainability_extra_patterns=["marker_embed_delta"],
        )
    )
    recipe.model_parts = [_model_with_extra()]

    recipe.setup()

    torch.testing.assert_close(recipe.model_parts[0].marker_embed_delta, torch.full((2, 4), 4.0))


def test_warm_start_fails_closed_when_the_artifact_lacks_the_extra_family(tmp_path):
    save_projector_artifact(_model(meta_projector=False), tmp_path / "sidecar", provenance={"origin": "x"}, stage=1)
    recipe = _Recipe(
        _cfg(
            initial_artifact=str(tmp_path / "sidecar"),
            expected_provenance={"origin": "x"},
            trainability_extra_patterns=["marker_embed_delta"],
        )
    )
    recipe.model_parts = [_model_with_extra()]
    with pytest.raises(ValueError, match="extra family"):
        recipe.setup()


def test_sidecar_export_allows_no_provenance(tmp_path):
    recipe = _Recipe(_cfg(artifact_dir=str(tmp_path / "export")))
    recipe.model_parts = [_model(meta_projector=False)]
    recipe.run_train_validation_loop()
    assert read_projector_artifact(tmp_path / "export").manifest.provenance == {}


def test_sidecar_export_fails_closed_above_one_rank(tmp_path):
    recipe = _Recipe(_cfg(artifact_dir=str(tmp_path / "export"), provenance={"origin": "x"}), world_size=2)
    recipe.model_parts = [_model(meta_projector=False)]
    with pytest.raises(NotImplementedError, match="exactly one rank"):
        recipe.run_train_validation_loop()


def test_require_one_rank():
    require_one_rank(1, "sidecar export")
    with pytest.raises(NotImplementedError, match="exactly one rank"):
        require_one_rank(2, "sidecar export")


# -- ProjectorSidecarState: native checkpoint path ---------------------------


def test_sidecar_state_fails_closed_before_bind():
    state = ProjectorSidecarState()
    with pytest.raises(RuntimeError, match="before setup bound"):
        state.state_dict()
    with pytest.raises(RuntimeError, match="already bound"):
        state.bind(_model(meta_projector=False).mm_projector) or state.bind(_model(meta_projector=False).mm_projector)


def test_sidecar_state_applies_pending_state_at_bind():
    source = _model(meta_projector=False)
    saved = {name: tensor.clone() for name, tensor in source.mm_projector.state_dict().items()}

    state = ProjectorSidecarState()
    state.load_state_dict(saved)  # resume lands before materialization: recorded as pending
    target = _model(meta_projector=False)
    assert state.bind(target.mm_projector) is True
    for name, value in target.mm_projector.state_dict().items():
        assert torch.equal(value, saved[name]), name


def test_sidecar_state_round_trips_cpu_copies_once_bound():
    source = _model(meta_projector=False)
    state = ProjectorSidecarState()
    assert state.bind(source.mm_projector) is False
    exported = state.state_dict()
    assert all(not tensor.requires_grad for tensor in exported.values())

    target = ProjectorSidecarState()
    target.bind(_model(meta_projector=False).mm_projector)
    target.load_state_dict(exported)  # already bound: applies immediately
    for name, value in source.mm_projector.state_dict().items():
        assert torch.equal(value, target.state_dict()[name]), name


class _TrackedRecipe(ProjectorRecipeMixin, CheckpointTrackerFake):
    """Sidecar recipe over the tracker double (mirrors BaseRecipe's contract)."""


def test_sidecar_resume_rides_the_recipe_state_tracker(tmp_path):
    # First run: setup attaches and binds the wrapper; a periodic checkpoint
    # carries it as <attribute>.pt.
    recipe = _TrackedRecipe(_cfg(), checkpoint_dir=tmp_path)
    recipe.model_parts = [_model()]
    recipe.setup()
    assert "projector_sidecar_state" in recipe.tracked_state_keys
    with torch.no_grad():
        for parameter in recipe.model_parts[0].mm_projector.parameters():
            parameter.add_(1.0)  # stand-in for training
    expected = {name: tensor.clone() for name, tensor in recipe.model_parts[0].mm_projector.state_dict().items()}
    checkpoint = recipe.save_checkpoint()
    assert (checkpoint / "projector_sidecar_state.pt").is_file()

    # Resume: the base setup's load_checkpoint restores the wrapper before the
    # projector exists; bind() applies it after materialization.
    restored = _TrackedRecipe(_cfg(), checkpoint_dir=tmp_path)
    restored.model_parts = [_model()]
    restored.setup()
    loaded = restored.model_parts[0].mm_projector.state_dict()
    assert loaded.keys() == expected.keys()
    for name, value in expected.items():
        assert torch.equal(loaded[name], value), name


def test_sidecar_resume_beats_the_initial_artifact(tmp_path):
    reference = _model(meta_projector=False)
    save_projector_artifact(reference, tmp_path / "initial", provenance={"origin": "unit-test"}, stage=1)
    stale = {name: tensor.clone() for name, tensor in reference.mm_projector.state_dict().items()}

    first = _TrackedRecipe(_cfg(), checkpoint_dir=tmp_path)
    first.model_parts = [_model()]
    first.setup()
    with torch.no_grad():
        for parameter in first.model_parts[0].mm_projector.parameters():
            parameter.add_(1.0)  # diverge from the initial artifact
    first.save_checkpoint()

    resumed = _TrackedRecipe(
        _cfg(initial_artifact=str(tmp_path / "initial"), expected_provenance={"origin": "unit-test"}),
        checkpoint_dir=tmp_path,
    )
    resumed.model_parts = [_model()]
    resumed.setup()
    loaded = resumed.model_parts[0].mm_projector.state_dict()
    for name, value in first.model_parts[0].mm_projector.state_dict().items():
        assert torch.equal(loaded[name], value), name
        assert not torch.equal(loaded[name], stale[name]), name


def test_plain_mapping_accepts_plain_and_to_dict_mappings():
    assert plain_mapping({"a": 1}) == {"a": 1}
    assert plain_mapping(SimpleNamespace(to_dict=lambda: {"b": 2})) == {"b": 2}


def test_plain_mapping_resolves_omegaconf_configs():
    omegaconf = pytest.importorskip("omegaconf")
    assert plain_mapping(omegaconf.OmegaConf.create({"a": "${b}", "b": 3})) == {"a": 3, "b": 3}


# -- module ownership  -------------------------

from nemotron_stitch.automodel.recipe import (  # noqa: E402
    resolve_initialization_artifact,
    resolve_resume_artifact,
)
from nemotron_stitch.contracts import OWNERSHIP_MODULE  # noqa: E402


class _ModuleRecipe(ProjectorRecipeMixin, _FakeBaseRecipe):
    projector_ownership = OWNERSHIP_MODULE


def test_module_ownership_allows_multiple_ranks_without_configured_artifacts():
    recipe = _ModuleRecipe(_cfg(), world_size=8)
    recipe.model_parts = [nn.Linear(4, 4)]  # no projector touchpoints when no artifact is configured
    assert recipe.setup() == "setup-result"
    assert recipe.base_setup_ran
    # No export at end of loop without projector.artifact_dir.
    assert recipe.run_train_validation_loop() == "loop-result"
    assert recipe.base_loop_ran


def test_module_ownership_keeps_the_pp_cp_guards():
    with pytest.raises(NotImplementedError, match="pipeline parallelism"):
        _ModuleRecipe(_cfg(), pp_enabled=True).setup()
    with pytest.raises(NotImplementedError, match="context parallelism"):
        _ModuleRecipe(_cfg(), cp_group_size=2).setup()


def test_unknown_ownership_mode_fails_closed():
    class _Bogus(ProjectorRecipeMixin, _FakeBaseRecipe):
        projector_ownership = "sideways"

    with pytest.raises(ValueError, match="unknown projector ownership mode"):
        _Bogus(_cfg()).setup()


def _module_model():
    """A host whose projector is a registered (module-ownership) child."""
    model = nn.Linear(4, 4)
    model.config = SimpleNamespace(mm_projectors=PROJECTOR_CONFIGS)
    model.mm_projector = MultimodalProjector.from_config(
        PROJECTOR_CONFIGS, output_size=4, projector_ownership=OWNERSHIP_MODULE
    )
    return model


def test_module_mode_setup_warm_starts_from_the_initial_artifact(tmp_path):
    reference = _module_model()
    save_projector_artifact(reference, tmp_path / "artifact", provenance={"origin": "unit-test"}, stage=1)
    recipe = _ModuleRecipe(
        _cfg(initial_artifact=str(tmp_path / "artifact"), expected_provenance={"origin": "unit-test"}),
        world_size=2,
    )
    recipe.model_parts = [_module_model()]

    assert recipe.setup() == "setup-result"

    expected = reference.mm_projector.state_dict()
    loaded = recipe.model_parts[0].mm_projector.state_dict()
    assert loaded.keys() == expected.keys()
    for name, value in expected.items():
        assert torch.equal(loaded[name], value), name


def test_module_mode_native_resume_does_not_overwrite_restored_projector(tmp_path):
    source = _module_model()
    save_projector_artifact(source, tmp_path / "initial", provenance={"origin": "unit-test"}, stage=1)
    target = _module_model()
    with torch.no_grad():
        for parameter in target.mm_projector.parameters():
            parameter.fill_(7)
    restored = {name: tensor.clone() for name, tensor in target.mm_projector.state_dict().items()}
    cfg = _cfg(initial_artifact=str(tmp_path / "initial"), expected_provenance={"origin": "unit-test"})
    cfg["checkpoint"] = _Cfg(restore_from="LATEST")
    recipe = _ModuleRecipe(cfg, world_size=2)
    recipe.model_parts = [target]

    recipe.setup()

    for name, value in target.mm_projector.state_dict().items():
        assert torch.equal(value, restored[name]), name


def test_module_mode_warm_start_restores_the_extra_family(tmp_path):
    reference = _module_model()
    reference.marker_embed_delta = nn.Parameter(torch.full((2, 4), 5.0))
    save_projector_artifact(
        reference,
        tmp_path / "artifact",
        provenance={"origin": "unit-test"},
        stage=1,
        extra_state=collect_extra_state(reference, TrainabilityPolicy(extra_patterns=("marker_embed_delta",))),
    )
    target = _module_model()
    target.marker_embed_delta = nn.Parameter(torch.zeros(2, 4))
    recipe = _ModuleRecipe(
        _cfg(
            initial_artifact=str(tmp_path / "artifact"),
            expected_provenance={"origin": "unit-test"},
            trainability_extra_patterns=["marker_embed_delta"],
        ),
        world_size=2,
    )
    recipe.model_parts = [target]

    assert recipe.setup() == "setup-result"
    torch.testing.assert_close(target.marker_embed_delta, torch.full((2, 4), 5.0))


def test_module_mode_export_carries_the_extra_family(tmp_path):
    artifact_dir = tmp_path / "export"
    model = _module_model()
    model.marker_embed_delta = nn.Parameter(torch.full((2, 4), 7.0))
    recipe = _ModuleRecipe(
        _cfg(
            artifact_dir=str(artifact_dir),
            provenance={"origin": "unit-test"},
            trainability_extra_patterns=["marker_embed_delta"],
        ),
        world_size=2,
    )
    recipe.model_parts = [model]

    assert recipe.run_train_validation_loop() == "loop-result"

    manifest = json.loads((artifact_dir / "mm-projector-manifest.json").read_text())
    assert manifest["extra_state"]["marker_embed_delta"]["shape"] == [2, 4]


def test_module_mode_warm_start_allows_no_provenance_lock(tmp_path):
    save_projector_artifact(_module_model(), tmp_path / "artifact", provenance={"origin": "x"}, stage=1)
    recipe = _ModuleRecipe(_cfg(initial_artifact=str(tmp_path / "artifact")))
    recipe.model_parts = [_module_model()]
    recipe.setup()


def test_module_mode_stage2_requires_the_stage1_bridge():
    recipe = _ModuleRecipe(_Cfg(model=_Cfg(mm_training_stage=2), projector=_Cfg({})))
    recipe.model_parts = [_module_model()]
    with pytest.raises(FileNotFoundError, match="Stage 2 requires"):
        recipe.setup()


def test_module_mode_export_writes_the_package_manifest(tmp_path):
    artifact_dir = tmp_path / "export"
    recipe = _ModuleRecipe(_cfg(artifact_dir=str(artifact_dir), provenance={"origin": "unit-test"}))
    recipe.model_parts = [_module_model()]

    assert recipe.run_train_validation_loop() == "loop-result"

    manifest = json.loads((artifact_dir / "mm-projector-manifest.json").read_text())
    assert manifest["schema"] == "nemotron-add-modality/mm-projector-manifest"
    target = _module_model()
    load_projector_artifact(target, artifact_dir, expected={"origin": "unit-test"})
    for name, value in recipe.model_parts[0].mm_projector.state_dict().items():
        assert torch.equal(target.mm_projector.state_dict()[name], value), name


def test_module_mode_export_allows_no_provenance(tmp_path):
    recipe = _ModuleRecipe(_cfg(artifact_dir=str(tmp_path / "export")))
    recipe.model_parts = [_module_model()]
    recipe.run_train_validation_loop()
    assert read_projector_artifact(tmp_path / "export").manifest.provenance == {}


def test_resolve_resume_artifact_follows_native_checkpoint(tmp_path):
    artifact = tmp_path / "checkpoints" / "epoch_0_step_10" / "artifact"
    artifact.mkdir(parents=True)
    resolved = resolve_resume_artifact(
        checkpoint_dir=tmp_path / "checkpoints",
        restore_from="epoch_0_step_10",
    )
    assert resolved == artifact
    # LATEST and absent checkpoints both resolve through the LATEST symlink dir.
    assert resolve_resume_artifact(checkpoint_dir=tmp_path / "checkpoints", restore_from=None) is None


def test_resolve_initialization_artifact_precedence(tmp_path):
    initial = tmp_path / "baseline-artifact"
    initial.mkdir()
    stage1 = tmp_path / "stage1-artifact"
    stage1.mkdir()

    # Fresh stage-1 warm start.
    assert (
        resolve_initialization_artifact(
            checkpoint_dir=tmp_path / "new-checkpoints",
            restore_from=None,
            initial_artifact=initial,
            stage1_artifact=None,
            stage=1,
        )
        == initial.resolve()
    )
    # Stage 2 bridges from the stage-1 artifact.
    assert (
        resolve_initialization_artifact(
            checkpoint_dir=tmp_path / "new-checkpoints",
            restore_from=None,
            initial_artifact=None,
            stage1_artifact=stage1,
            stage=2,
        )
        == stage1.resolve()
    )
    # Native resume wins over everything.
    resumed = tmp_path / "new-checkpoints" / "LATEST" / "artifact"
    resumed.mkdir(parents=True)
    assert (
        resolve_initialization_artifact(
            checkpoint_dir=tmp_path / "new-checkpoints",
            restore_from=None,
            initial_artifact=initial,
            stage1_artifact=stage1,
            stage=2,
        )
        == resumed.resolve()
    )
    # Stage 2 without a bridge artifact fails closed, naming the config key.
    # Fresh checkpoint dirs: the LATEST artifact above would win precedence.
    with pytest.raises(FileNotFoundError, match="Stage 2 requires projector\\.stage1_artifact"):
        resolve_initialization_artifact(
            checkpoint_dir=tmp_path / "stage2-checkpoints",
            restore_from=None,
            initial_artifact=None,
            stage1_artifact=None,
            stage=2,
            stage1_artifact_key="projector.stage1_artifact",  # gitleaks:allow
        )
    with pytest.raises(FileNotFoundError, match="initialization artifact is absent"):
        resolve_initialization_artifact(
            checkpoint_dir=tmp_path / "missing-checkpoints",
            restore_from=None,
            initial_artifact=tmp_path / "missing",
            stage1_artifact=None,
            stage=1,
        )
