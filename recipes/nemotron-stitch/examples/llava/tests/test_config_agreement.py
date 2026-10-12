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

"""Keep the self-contained model stage configs aligned with application constants."""

from pathlib import Path
from types import SimpleNamespace

import llava_qwen
import pytest
import yaml

CONFIGS = Path(__file__).parents[1] / "configs"


def _load(name: str) -> dict:
    return yaml.safe_load((CONFIGS / name).read_text())


def test_qwen_automodel_configs_agree():
    for name in ("alignment-qwen.yaml", "sft-qwen.yaml"):
        config = _load(name)
        model = config["model"]
        projector = model["mm_projectors"][0]

        assert model["architectures"] == [llava_qwen.TRAINING_ARCHITECTURE]
        assert model["backend"] == {
            "_target_": "nemo_automodel.components.models.common.BackendConfig",
            "attn": "sdpa",
        }
        assert model["attn_implementation"] == "flash_attention_2"
        assert projector["name"] == llava_qwen.PROJECTOR_NAME
        assert model["mm_placeholder_token_ids"] == {llava_qwen.PROJECTOR_NAME: 248055}
        assert config["step_scheduler"]["local_batch_size"] == 1
        assert config["packed_sequence"] == {
            "packed_sequence_size": 2048,
            "packing_strategy": "nemotron_stitch.automodel.data.StreamingFeaturePackingConfig",
            "packing_format": "neat",
        }
        assert config["dataset"]["_target_"] == ("nemotron_stitch.automodel.data.ManifestFeatureIterableDatasetConfig")
        assert config["dataset"]["projector_name"] == llava_qwen.PROJECTOR_NAME
        assert config["dataset"]["placeholder_token_id"] == 248055
        assert config["dataloader"]["batch_size"] is None
        assert config["dataloader"]["shuffle_buffer_size"] == 1000


def test_lightning_alignment_and_sft_stream_feature_preserving_thd_packs():
    for name in ("alignment-lightning.yaml", "sft-lightning.yaml"):
        config = _load(name)
        packing = config["packed_sequence"]
        dataset = config["dataset"]

        assert config["model"]["backend"] == {
            "_target_": "nemo_automodel.components.models.common.BackendConfig",
            "attn": "te",
        }
        # EP=8 is the default 8xH100 topology for the Lightning targets: it
        # shards the 128 routed experts 16/rank and takes the routed-expert
        # weights out of the per-microbatch FSDP2 all-gather (qualified
        # 2026-09-03). Single-GPU runs override it with
        # --distributed.ep_size=1.
        assert config["distributed"]["ep_size"] == 8
        assert config["step_scheduler"]["local_batch_size"] == 1
        assert packing == {
            "packed_sequence_size": 2048,
            "packing_strategy": "nemotron_stitch.automodel.data.StreamingFeaturePackingConfig",
            "packing_format": "thd",
        }
        assert dataset["_target_"] == "nemotron_stitch.automodel.data.ManifestFeatureIterableDatasetConfig"
        assert config["dataloader"]["batch_size"] is None
        assert config["dataloader"]["shuffle_buffer_size"] == 1000


def test_qwen_grpo_configs_agree():
    model = _load("models/qwen3.5-4b.yaml")["policy"]
    run = _load("grpo-qwen.yaml")

    assert model["hf_config_overrides"]["architectures"] == [llava_qwen.TRAINING_ARCHITECTURE]
    assert model["hf_config_overrides"]["mm_encoder_start"] == "<|vision_start|>"
    assert model["hf_config_overrides"]["mm_encoder_placeholder"] == "<|vision_pad|>"
    assert model["hf_config_overrides"]["mm_encoder_end"] == "<|vision_end|>"
    assert run["policy"]["generation"]["mm_plugin_callback"] == "llava_qwen.register_vllm"
    assert run["policy"]["generation"]["mm_architecture"] == llava_qwen.ROLLOUT_ARCHITECTURE


@pytest.mark.parametrize(
    "name",
    (
        "grpo.yaml",
        "grpo-qwen.yaml",
        "grpo-lightning.yaml",
        "grpo-qwen3.6-35b.yaml",
    ),
)
def test_colocated_grpo_keeps_both_workers_resident(name):
    run = _load(name)

    assert run["loss_fn"]["force_on_policy_ratio"] is True
    rollout_batch = run["grpo"]["num_prompts_per_step"] * run["grpo"]["num_generations_per_prompt"]
    assert run["policy"]["train_global_batch_size"] == rollout_batch
    assert run["policy"]["generation_batch_size"] == rollout_batch
    assert run["policy"]["keep_policy_on_gpu"] is True
    generation = run["policy"]["generation"]
    assert generation["keep_vllm_on_gpu"] is True
    assert generation["colocated"]["enabled"] is True
    assert generation["vllm_kwargs"]["kv_cache_memory_bytes"] == 4 * 1024**3
    assert generation["vllm_cfg"]["enforce_eager"] is False
    assert generation["vllm_kwargs"]["compilation_config"] == {
        "mode": "none",
        "cudagraph_mode": "full_decode_only",
    }
    assert generation["vllm_kwargs"]["max_num_seqs"] == "${policy.generation_batch_size}"
    # U-70: never inherit vLLM 0.25.1's async_scheduling default on colocated rollouts.
    assert generation["vllm_kwargs"]["async_scheduling"] is False


def test_two_gpu_grpo_uses_upstream_non_colocated_lifecycle():
    run = _load("grpo-2gpu.yaml")

    assert run["policy"]["keep_policy_on_gpu"] is False
    assert run["policy"]["generation"]["keep_vllm_on_gpu"] is False
    assert run["policy"]["generation"]["colocated"]["enabled"] is False


@pytest.mark.parametrize("name", ("grpo.yaml", "grpo-qwen.yaml", "grpo-lightning.yaml", "grpo-qwen3.6-35b.yaml"))
def test_rollout_uses_the_policy_projector_artifact(name):
    run = _load(name)

    assert run["data"]["train"]["encoder_loader_kwargs"]["artifact_dir"] == "${policy.projector_artifact_path}"


def test_eight_gpu_grpo_uses_upstream_non_colocated_lifecycle():
    run = _load("grpo-8gpu.yaml")

    assert run["policy"]["keep_policy_on_gpu"] is False
    assert run["policy"]["generation"]["keep_vllm_on_gpu"] is False
    assert run["policy"]["generation"]["colocated"]["enabled"] is False


def test_qwen36_moe_configs_agree():
    for name in ("alignment-qwen3.6-35b.yaml", "sft-qwen3.6-35b.yaml"):
        config = _load(name)
        model = config["model"]

        assert model["architectures"] == [llava_qwen.MOE_TRAINING_ARCHITECTURE]
        assert model["backend"]["experts"] == "torch_mm"
        assert model["backend"]["dispatcher"] == "torch"
        assert model["mm_placeholder_token_ids"] == {llava_qwen.PROJECTOR_NAME: 248055}
        assert config["step_scheduler"]["max_steps"] == 1
        assert config["packed_sequence"]["packing_strategy"] == (
            "nemotron_stitch.automodel.data.StreamingFeaturePackingConfig"
        )
        assert config["packed_sequence"]["packing_format"] == "neat"
        assert config["dataset"]["_target_"] == ("nemotron_stitch.automodel.data.ManifestFeatureIterableDatasetConfig")
        assert config["dataloader"]["batch_size"] is None
        assert config["dataloader"]["shuffle_buffer_size"] == 1000
        assert config["projector"]["model_registry_callback"] == "llava_qwen.register_moe_models"

    assert _load("sft-qwen3.6-35b.yaml")["peft"]["target_modules"] == [
        "model.language_model.layers.*.self_attn.q_proj",
        "model.language_model.layers.*.self_attn.k_proj",
        "model.language_model.layers.*.self_attn.v_proj",
        "model.language_model.layers.*.self_attn.o_proj",
    ]

    model = _load("models/qwen3.6-35b-a3b.yaml")["policy"]
    run = _load("grpo-qwen3.6-35b.yaml")
    assert model["hf_config_overrides"]["architectures"] == [llava_qwen.MOE_TRAINING_ARCHITECTURE]
    assert run["policy"]["generation"]["mm_plugin_callback"] == "llava_qwen.register_moe_vllm"
    assert run["policy"]["generation"]["mm_architecture"] == llava_qwen.MOE_ROLLOUT_ARCHITECTURE
    assert run["grpo"]["max_num_steps"] == 1


def test_qwen_vllm_accepts_external_embeddings_without_native_media():
    torch = pytest.importorskip("torch")
    pytest.importorskip("vllm")
    from vllm import ModelRegistry

    llava_qwen.register_vllm()
    model_cls = ModelRegistry._try_load_model_cls(llava_qwen.ROLLOUT_ARCHITECTURE)
    model = object.__new__(model_cls)
    payload = torch.zeros(49, 2560)

    embeddings = model.embed_multimodal(clip_image_embeds=payload)

    assert len(embeddings) == 1
    assert embeddings[0] is payload


def test_qwen_moe_vllm_accepts_external_embeddings_without_native_media():
    torch = pytest.importorskip("torch")
    pytest.importorskip("vllm")
    from vllm import ModelRegistry

    llava_qwen.register_moe_vllm()
    model_cls = ModelRegistry._try_load_model_cls(llava_qwen.MOE_ROLLOUT_ARCHITECTURE)
    model = object.__new__(model_cls)
    payload = torch.zeros(49, 2048)

    embeddings = model.embed_multimodal(clip_image_embeds=payload)

    assert len(embeddings) == 1
    assert embeddings[0] is payload


def test_qwen_vllm_excludes_external_embeddings_from_native_mrope(monkeypatch):
    pytest.importorskip("vllm")
    from vllm import ModelRegistry
    from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration

    observed = []

    def get_positions(self, input_tokens, mm_features):
        del self, input_tokens
        observed.extend(feature.modality for feature in mm_features)
        return "positions"

    monkeypatch.setattr(Qwen3_5ForConditionalGeneration, "get_mrope_input_positions", get_positions)
    llava_qwen.register_vllm()
    model_cls = ModelRegistry._try_load_model_cls(llava_qwen.ROLLOUT_ARCHITECTURE)
    model = object.__new__(model_cls)

    result = model.get_mrope_input_positions(
        [1, 2, 3],
        [SimpleNamespace(modality="clip_image"), SimpleNamespace(modality="image")],
    )

    assert result == "positions"
    assert observed == ["image"]


def test_trainable_projector_variant_overlays_the_frozen_stage_contract():
    """The variant changes exactly the joint-training recipe, nothing else.

    A config drift between grpo.yaml and its trainable-projector overlay would
    be an accidental recipe change, so the test enumerates the overlay's keys
    and asserts the rest of the two files agree.
    """
    frozen = _load("grpo.yaml")
    variant = _load("grpo-trainable-projector.yaml")

    assert variant["defaults"] == "grpo.yaml"
    assert variant["loss_fn"] == {
        "reference_policy_kl_penalty": 0.01,
        "force_on_policy_ratio": False,
    }
    assert variant["policy"] == {
        "hf_config_overrides": {
            "mm_projector_ownership": "module",
            "mm_projector_trainable": True,
        },
        "projector_lr": 1.0e-5,
    }
    assert variant["checkpointing"]["checkpoint_dir"] == "outputs/grpo-trainable-projector/checkpoints"
    assert variant["logger"]["log_dir"] == "outputs/grpo-trainable-projector/logs"

    # The overlay must not restate any part of the base recipe it does not
    # change: an override that silently diverges from the frozen stage is a
    # copy of grpo.yaml drifting, not an overlay.
    def overlay_keys(node, prefix=""):
        keys = set()
        for key, value in node.items():
            path = f"{prefix}{key}"
            if isinstance(value, dict):
                keys |= overlay_keys(value, path + ".")
            else:
                keys.add(path)
        return keys

    expected = {
        "defaults",
        "loss_fn.reference_policy_kl_penalty",
        "loss_fn.force_on_policy_ratio",
        "policy.hf_config_overrides.mm_projector_ownership",
        "policy.hf_config_overrides.mm_projector_trainable",
        "policy.projector_lr",
        "checkpointing.checkpoint_dir",
        "logger.log_dir",
    }
    assert overlay_keys(variant) == expected

    # The rollout contract is inherited unchanged: the data plane still embeds
    # rollouts through the SFT-artifact sidecar (one step stale by design, the
    # unforced ratio absorbs it), so the variant must not touch the data
    # section, the sampling recipe, or the model contract.
    assert "data" not in variant
    assert "grpo" not in variant

    assert frozen["loss_fn"]["force_on_policy_ratio"] is True
    # The frozen sidecar contract lives in the composed model contract, not in
    # grpo.yaml itself.
    model = _load("models/nano-4b.yaml")["policy"]["hf_config_overrides"]
    assert model["mm_projector_ownership"] == "sidecar"
    assert model["mm_projector_trainable"] is False


def test_ep8_trainable_overlay_changes_only_the_mesh_and_the_step_count():
    """The EP=8 Lightning overlay is the trainable recipe plus the mesh.

    It composes over the frozen grpo-lightning.yaml recipe and supplies module
    ownership, the unforced ratio, and projector_lr alongside the EP=8 mesh,
    bounded step count, and output paths.
    """
    overlay = _load("grpo-lightning-trainable-ep8.yaml")

    assert overlay["defaults"] == "grpo-lightning.yaml"
    assert overlay["cluster"] == {"gpus_per_node": 8, "num_nodes": 1}
    assert overlay["grpo"] == {"max_num_steps": 3}
    assert overlay["loss_fn"] == {"force_on_policy_ratio": False}
    assert overlay["policy"]["hf_config_overrides"] == {
        "mm_projector_ownership": "module",
        "mm_projector_trainable": True,
    }
    assert overlay["policy"]["projector_lr"] == 1.0e-5
    assert overlay["policy"]["dtensor_cfg"] == {"expert_parallel_size": 8}
    assert overlay["policy"]["generation"]["vllm_cfg"] == {"expert_parallel_size": 8}
    assert overlay["logger"]["log_dir"] == "outputs/grpo-lightning-trainable-ep8/logs"

    # The EP=8 vLLM expert-parallel rollout is external-DP: vLLM_DP_SIZE
    # derives from expert_parallel_size / tensor_parallel_size, and the
    # generation worker count comes from the cluster. A cluster that does
    # not match the mesh deadlocks the first engine's DP rendezvous
    # (reproduced 2026-10-01), so the two must move together.
    assert overlay["policy"]["generation"]["vllm_cfg"]["expert_parallel_size"] == overlay["cluster"]["gpus_per_node"]


def test_hsdp_overlay_geometry():
    """Replication, worker count, and rollout batch must describe one mesh."""
    overlay = _load("grpo-trainable-projector-hsdp.yaml")
    joint = _load(overlay["defaults"])
    frozen = _load(joint["defaults"])
    dtensor = frozen["policy"]["dtensor_cfg"] | overlay["policy"]["dtensor_cfg"]
    world = overlay["cluster"]["gpus_per_node"] * overlay["cluster"]["num_nodes"]
    assert dtensor["tensor_parallel_size"] == dtensor["context_parallel_size"] == dtensor["expert_parallel_size"] == 1
    assert world % dtensor["dp_replicate_size"] == 0
    assert world // dtensor["dp_replicate_size"] == 2
    responses = overlay["grpo"]["num_prompts_per_step"] * overlay["grpo"]["num_generations_per_prompt"]
    assert responses == overlay["policy"]["train_global_batch_size"] == overlay["policy"]["generation_batch_size"]
    assert joint["policy"]["hf_config_overrides"]["mm_projector_trainable"] is True
    assert joint["policy"]["hf_config_overrides"]["mm_projector_ownership"] == "module"
    assert joint["loss_fn"]["force_on_policy_ratio"] is False
    assert joint["loss_fn"]["reference_policy_kl_penalty"] > 0
    assert overlay["checkpointing"]["save_period"] == overlay["grpo"]["max_num_steps"]


def test_dpo_stage_handoffs():
    """Preference training uses the SFT donor and the matching projector."""
    config = _load("dpo.yaml")
    model = _load(config["defaults"][1])["policy"]
    assert model["dtensor_cfg"]["lora_cfg"]["restore_from"]
    assert model["projector_artifact_path"]
    assert (
        config["policy"]["initial_adapter_expected_config"]["r"] == config["policy"]["dtensor_cfg"]["lora_cfg"]["dim"]
    )
    assert (
        config["policy"]["initial_adapter_expected_config"]["lora_alpha"]
        == config["policy"]["dtensor_cfg"]["lora_cfg"]["alpha"]
    )
    assert config["data"]["train"]["dataset_name"] == "PreferenceDataset"
    assert config["data"]["max_input_seq_length"] == "${policy.max_total_sequence_length}"
    assert config["policy"]["train_global_batch_size"] % config["policy"]["train_micro_batch_size"] == 0
    assert config["dpo"]["reference_policy_kl_penalty"] > 0
