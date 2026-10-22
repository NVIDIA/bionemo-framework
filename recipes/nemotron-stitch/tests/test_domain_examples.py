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

"""Numerical and index contracts retained by the small domain examples."""

from __future__ import annotations

import importlib
import re
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.fixture
def examples(monkeypatch):
    root = Path(__file__).resolve().parents[1] / "examples"
    for name in ("genetics", "chemistry"):
        monkeypatch.syspath_prepend(str(root / name))
    return importlib.import_module("genetics_example"), importlib.import_module("chemistry_example")


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 1
    bos_token = None

    def convert_tokens_to_ids(self, token):
        return int(token[9:-1])

    def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
        text = "".join(message["role"] + ":" + message["content"] for message in messages)
        return text + ("assistant:" if add_generation_prompt else "")

    def __call__(self, text, **kwargs):
        parts = re.findall(r"<SPECIAL_\d+>|.", text)
        ids = [int(part[9:-1]) if part.startswith("<SPECIAL_") else ord(part) + 100 for part in parts]
        return SimpleNamespace(input_ids=ids)


def _graph_row(atoms, bonds):
    matrix = np.arange((1 + atoms + bonds) * 3200, dtype=np.float32).reshape(-1, 3200)
    return dict(
        features=matrix,
        molecule_count=1,
        atom_count=atoms,
        directed_bond_count=bonds,
        bond_representation="undirected_concat",
        prompt="<SPECIAL_30><SPECIAL_33>"
        + " atom<SPECIAL_32>" * atoms
        + " bond<SPECIAL_34>" * (bonds // 2)
        + "<SPECIAL_31>",
        target="answer",
    )


def test_kermt_native_widths_and_reverse_bond_order(examples):
    _, chemistry = examples
    row = _graph_row(3, 4)
    molecule, atoms, bonds = chemistry.split_features(row)
    np.testing.assert_array_equal(molecule, row["features"][:1])
    np.testing.assert_array_equal(atoms, row["features"][1:4, :1600])
    expected = np.stack(
        [
            np.concatenate([row["features"][4, :1600], row["features"][5, :1600]]),
            np.concatenate([row["features"][6, :1600], row["features"][7, :1600]]),
        ]
    )
    np.testing.assert_array_equal(bonds, expected)


def test_kermt_ragged_batch_flat_indices_and_empty_bonds(examples, monkeypatch):
    _, chemistry = examples

    collator = chemistry.ChemistryCollator(_Tokenizer())
    rows = [_graph_row(3, 4), _graph_row(1, 0)]
    batch = collator(rows)
    flat_ids = batch["input_ids"].reshape(-1)
    for branch, token_id, width in chemistry.BRANCHES:
        indices = batch[f"mm_token_indices__{branch}"]
        assert torch.equal(indices, flat_ids.eq(token_id).nonzero().flatten())
        features = batch[f"mm_features__{branch}"]
        assert features.shape == (len(indices), width)
    assert batch["mm_token_indices__chem_atom"][-1] >= batch["input_ids"].shape[1]
    assert batch["mm_features__chem_bond"].shape == (2, 3200)
    assert torch.all(batch["labels"][batch["attention_mask"] == 0] == -100)
    rows[1]["prompt"] += "<SPECIAL_34>"
    with pytest.raises(ValueError, match="placeholder slots"):
        collator(rows)


@pytest.mark.parametrize(
    "change",
    [
        {"directed_bond_count": 3},
        {"atom_count": 0},
        {"bond_representation": "directed_concat"},
        {"features": np.zeros((2, 3200))},
    ],
)
def test_kermt_rejects_ambiguous_geometry(examples, change):
    _, chemistry = examples
    with pytest.raises(ValueError):
        chemistry.split_features({**_graph_row(3, 4), **change})


def test_grpo_graph_order_matches_policy_and_items_have_compact_storage(examples, monkeypatch):
    formatter = pytest.importorskip("nemo_rl.data.llm_message_utils")
    from nemotron_stitch.nemo_rl import transport

    _, chemistry = examples
    row = _graph_row(2, 2)
    # Deliberately interleave branches: do not assume all atoms precede bonds.
    question = "|".join(f"<SPECIAL_30><SPECIAL_{token}><SPECIAL_31>" for token in (33, 32, 34, 32))

    def formatted(messages, tokenizer, *args, **kwargs):
        return [{**message, "token_ids": torch.tensor(tokenizer(message["content"]).input_ids)} for message in messages]

    monkeypatch.setattr(formatter, "get_formatted_message_log", formatted)
    monkeypatch.setattr(chemistry, "_feature_cache", lambda root: SimpleNamespace(get=lambda key: row["features"]))
    monkeypatch.setattr(transport, "project_features", lambda features, artifact, name: features[:, :2].clone())
    datum = dict(
        row,
        prompt=question,
        question=question,
        sample_id="graph",
        feature_key="graph",
        ground_truth="A",
        task_metadata_callback="nemotron_stitch.nemo_rl.transport.ground_truth_metadata",
        task_name="chemistry",
        encoder_loader_kwargs={"cache_root": "unused", "artifact_dir": "unused"},
        chat_template_kwargs={},
    )
    result = chemistry.grpo_processor(datum, None, _Tokenizer(), 4096, 0)
    arrays = chemistry.split_features(row)
    items = result["vllm_multi_modal_data"]["chemistry"]
    expected = [arrays[0][0, :2], arrays[1][0, :2], arrays[2][0, :2], arrays[1][1, :2]]
    for item, value in zip(items, expected, strict=True):
        np.testing.assert_array_equal(item.numpy(), value[None])
        assert item.untyped_storage().nbytes() == item.numel() * item.element_size()
    assert result["vllm_content"].count("<chemistry>") == 4
    assert result["extra_env_info"] == {"ground_truth": "A"}
    datum["prompt"] = question.replace("<SPECIAL_32>", "", 1)
    with pytest.raises(ValueError, match="policy slots disagree"):
        chemistry.grpo_processor(datum, None, _Tokenizer(), 4096, 0)


@pytest.mark.parametrize("domain", ["genetics", "chemistry"])
def test_grpo_upstream_config_composes_with_sft_contract(domain, monkeypatch):
    nemo_rl = pytest.importorskip("nemo_rl")
    from nemo_rl.utils.config import load_config, register_omegaconf_resolvers
    from omegaconf import OmegaConf

    monkeypatch.setenv("STITCH_SFT_MODEL", "/tmp/example-sft")
    monkeypatch.setenv("NEMO_RL_ROOT", str(Path(nemo_rl.__file__).resolve().parents[1]))
    register_omegaconf_resolvers()
    root = Path(__file__).resolve().parents[1] / "examples" / domain / "configs"
    cfg = OmegaConf.to_container(load_config(root / "grpo.yaml"), resolve=True)
    sft = OmegaConf.to_container(OmegaConf.load(root / "sft.yaml"), resolve=True)
    policy = cfg["policy"]
    from nemotron_stitch.nemo_rl.policy import require_replicated_projector_policy_config

    require_replicated_projector_policy_config(policy)
    for key in ("architectures", "mm_projectors", "mm_placeholder_token_ids"):
        assert policy["hf_config_overrides"][key] == sft["model"][key]
    assert policy["model_name"] == "/tmp/example-sft"
    assert policy["hf_config_overrides"]["mm_projector_trainable"] is False
    assert cfg["data"]["train"]["encoder_loader_kwargs"]["artifact_dir"] == sft["projector"]["artifact_dir"]
    assert list(cfg["env"]) == ["vlm"]
    assert (
        policy["train_global_batch_size"]
        == cfg["grpo"]["num_prompts_per_step"] * cfg["grpo"]["num_generations_per_prompt"]
    )


@pytest.mark.parametrize("domain", ["genetics", "chemistry"])
@pytest.mark.parametrize("stage", ["alignment", "sft"])
def test_supervised_fused_loss_constructs(domain, stage):
    pytest.importorskip("nemo_automodel")
    import yaml

    config = Path(__file__).resolve().parents[1] / "examples" / domain / "configs" / f"{stage}.yaml"
    loss_config = yaml.safe_load(config.read_text())["loss_fn"]
    module, _, name = loss_config.pop("_target_").rpartition(".")
    criterion = getattr(importlib.import_module(module), name)(**loss_config)
    from nemo_automodel.components.loss.linear_ce import FusedLinearCrossEntropy

    assert isinstance(criterion, FusedLinearCrossEntropy)
    if not torch.cuda.is_available():
        return  # The native fused kernel requires CUDA; construction is CPU-safe.
    hidden = torch.randn(2, 5, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    labels = torch.tensor([[7, 8, 9, -100, -100], [2, 3, 4, 5, 6]], device="cuda")
    loss = criterion(hidden, labels, weight)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(hidden.grad).all()
    assert torch.isfinite(weight.grad).all()


@pytest.mark.parametrize(
    "gold,response,expected",
    [
        ("A", r"\boxed{A}", 1.0),
        ("A", r"\boxed{B}", 0.0),
        ("A", r"Earlier: \boxed{A}. Final: \boxed{B}", 0.0),
        ("A", r"<think>Reasoning</think>\boxed{B}", 0.0),
        ("A", "No final answer", 0.0),
        ("64", r"\boxed{64}", 1.0),
    ],
)
def test_native_grpo_reward_scores_only_final_boxed_answer(gold, response, expected):
    pytest.importorskip("nemo_rl")
    from nemo_rl.environments.rewards import geo3k_reward

    score, correct = geo3k_reward(gold, response, format_score=0.0)
    assert score == expected
    assert correct is bool(expected)


@pytest.mark.parametrize("domain", ["genetics", "chemistry"])
def test_sft_schedule_accepts_production_batch_and_checkpoint_cadence(domain):
    pytest.importorskip("nemo_automodel")
    import yaml
    from nemo_automodel.components.training.step_scheduler import StepScheduler

    config = Path(__file__).resolve().parents[1] / "examples" / domain / "configs" / "sft.yaml"
    settings = yaml.safe_load(config.read_text())["step_scheduler"]
    scheduler = StepScheduler(**settings, dp_size=8, dataloader=list(range(32)), preemption_signal=None)
    scheduler.step = settings["max_steps"] - 1
    assert scheduler.is_ckpt_step


@pytest.mark.parametrize("domain", ["genetics", "chemistry"])
def test_full_parameter_grpo_keeps_projectors_frozen(domain, monkeypatch):
    pytest.importorskip("nemo_rl")
    from nemo_rl.utils.config import load_config

    monkeypatch.setenv("NEMO_RL_ROOT", "/opt/nemo-rl")
    root = Path(__file__).resolve().parents[1] / "examples" / domain / "configs"
    config = load_config(root / "grpo-full.yaml")
    assert config.policy.dtensor_cfg.lora_cfg.enabled is False
    assert config.policy.hf_config_overrides.mm_projector_trainable is False
    assert config.policy.hf_config_overrides.mm_projector_ownership == "sidecar"


@pytest.mark.parametrize("domain", ["genetics", "chemistry"])
def test_grpo_variants_isolate_checkpoint_paths(domain, monkeypatch):
    pytest.importorskip("nemo_rl")
    from nemo_rl.utils.config import load_config

    monkeypatch.setenv("NEMO_RL_ROOT", "/opt/nemo-rl")
    root = Path(__file__).resolve().parents[1] / "examples" / domain / "configs"
    variants = ["grpo", "grpo-full"]
    if domain == "genetics":
        variants += ["grpo-super", "grpo-super-full"]
    paths = []
    for variant in variants:
        config = load_config(root / f"{variant}.yaml")
        assert config.checkpointing.checkpoint_dir == f"outputs/{variant}/checkpoints"
        assert config.logger.log_dir == f"outputs/{variant}/logs"
        paths.append(config.checkpointing.checkpoint_dir)
    assert len(set(paths)) == len(variants)


def test_genetics_sft_bounds_checkpoint_retention():
    pytest.importorskip("nemo_automodel")
    import yaml
    from nemo_automodel.components.checkpoint.config import CheckpointingConfig

    config = Path(__file__).resolve().parents[1] / "examples/genetics/configs/sft.yaml"
    checkpoint = CheckpointingConfig(**yaml.safe_load(config.read_text())["checkpoint"])
    assert checkpoint.max_recent_checkpoints == 1


def test_super_configs_select_native_hosts_and_frozen_grpo(monkeypatch):
    pytest.importorskip("nemo_rl")
    import yaml
    from nemo_rl.utils.config import load_config

    monkeypatch.setenv("NEMO_RL_ROOT", "/opt/nemo-rl")
    root = Path(__file__).resolve().parents[1] / "examples/genetics/configs"
    for stage in ("alignment", "sft"):
        config = yaml.safe_load((root / f"{stage}-super.yaml").read_text())
        assert config["model"]["architectures"] == ["GeneticsExampleSuperForCausalLM", "NemotronH_Omni_Reasoning_V3"]
        assert config["model"]["llm_config"]["num_nextn_predict_layers"] == 0
        assert config["packed_sequence"]["prepacked"] is True
        assert config["projector"]["model_registry_callback"] == "genetics_example.register_models"
    config = load_config(root / "grpo-super-full.yaml")
    assert config.policy.generation.mm_architecture == "GeneticsExampleSuperProjected"
    assert config.policy.dtensor_cfg.lora_cfg.enabled is False
    assert config.policy.hf_config_overrides.mm_projector_trainable is False


@pytest.mark.parametrize("domain", ["genetics", "chemistry"])
def test_full_decoder_update_learns_boundaries_with_frozen_projector(examples, domain):
    app = examples[0 if domain == "genetics" else 1]
    spec = replace(app.MODELS[0], automodel_cls="test_domain_preparation._TinyLM")
    model = app.build_host_cls(spec)(
        SimpleNamespace(
            hidden_size=8,
            mm_projectors=[dict(name="dna", kind="mlp2x_gelu_norm", mm_hidden_size=3, hidden_size=8)],
            mm_placeholder_token_ids={"dna": 32},
            mm_projector_ownership="module",
            mm_projector_trainable=False,
        )
    )
    frozen = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters() if not parameter.requires_grad
    }
    trainable = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert any(name.startswith("mm_projector.") for name in frozen)
    embeddings = model.get_input_embeddings().weight
    before = embeddings.detach().clone()
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=0.01)
    logits = model(
        input_ids=torch.tensor([[30, 32, 31, 100]]),
        mm_features__dna=torch.ones(1, 3),
        mm_token_indices__dna=torch.tensor([1]),
    )
    (-logits[:, -1].log_softmax(-1)[:, 7].mean()).backward()
    for token in (30, 31):
        assert torch.count_nonzero(embeddings.grad[token]) > 0
    assert torch.count_nonzero(embeddings.grad[32]) == 0
    optimizer.step()
    for token in (30, 31):
        assert not torch.equal(embeddings[token], before[token])
    for name, parameter in model.named_parameters():
        if name in frozen:
            assert parameter.grad is None
            assert torch.equal(parameter, frozen[name])
    assert any(
        not torch.equal(parameter, trainable[name]) for name, parameter in model.named_parameters() if name in trainable
    )
