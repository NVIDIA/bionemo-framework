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

import importlib.util
import json
from types import SimpleNamespace

import pytest
import torch

from nemotron_stitch.nemo_rl.data import EncoderRLDataset


def test_rl_dataset_preserves_rollout_template_kwargs(tmp_path):
    manifest = tmp_path / "rl.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "case-0",
                "volume_name": "volume.nii.gz",
                "volume_path": "dataset/volume.nii.gz",
                "question": "Answer yes or no.",
                "reward_type": "binary",
                "expected_polarity": "yes",
            }
        )
        + "\n"
    )

    dataset = EncoderRLDataset(
        str(manifest),
        adapter="ct",
        encoder_loader="example.loader",
        encoder_loader_kwargs={"cache_root": "cache"},
        task_metadata_callback="example.metadata",
        placeholder_token="<slot>",
        start_token="<start>",
        end_token="<end>",
        chat_template_kwargs={"enable_thinking": False},
    )

    assert dataset.dataset[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert dataset.dataset[0]["encoder_loader"] == "example.loader"


def test_rl_dataset_can_disable_bos_for_checkpoint_prompt_parity(tmp_path):
    """The dataset binds its configured BOS policy into the public processor."""
    manifest = tmp_path / "rl.jsonl"
    manifest.write_text(json.dumps({"sample_id": "case-0", "question": "Answer."}) + "\n")
    dataset = EncoderRLDataset(
        str(manifest),
        adapter="sample",
        encoder_loader="example.loader",
        encoder_loader_kwargs={},
        placeholder_token="<slot>",
        start_token="<start>",
        end_token="<end>",
        add_bos_token=False,
    )

    dataset.set_processor()
    assert dataset.add_bos_token is False
    assert dataset.processor.keywords == {"add_bos": False}


def test_chat_template_kwargs_proxy_applies_datum_kwargs_uniformly():
    from nemotron_stitch.nemo_rl.data import ChatTemplateKwargsProxy

    class _Tok:
        def apply_chat_template(self, messages, **kwargs):
            return kwargs

        def convert_tokens_to_ids(self, token):
            return 32

    tokenizer = _Tok()
    proxy = ChatTemplateKwargsProxy(tokenizer, {"enable_thinking": False})
    assert proxy.apply_chat_template([], tokenize=False) == {"enable_thinking": False, "tokenize": False}
    # Explicit call-site kwargs win over the datum's.
    assert proxy.apply_chat_template([], enable_thinking=True) == {"enable_thinking": True}
    assert proxy.convert_tokens_to_ids("<SPECIAL_32>") == 32  # delegates everything else
    assert proxy.tokenizer is tokenizer


def test_prepend_bos_if_needed_matches_policy_bos_prefix():
    from nemotron_stitch.nemo_rl.data import prepend_bos_if_needed

    class _Tok:
        bos_token = "<s>"

    assert prepend_bos_if_needed("<|im_start|>user", _Tok(), add_bos=True) == "<s><|im_start|>user"
    # Already prefixed, or disabled, or no BOS token: unchanged.
    assert prepend_bos_if_needed("<s><|im_start|>user", _Tok(), add_bos=True) == "<s><|im_start|>user"
    assert prepend_bos_if_needed("<|im_start|>user", _Tok(), add_bos=False) == "<|im_start|>user"
    assert (
        prepend_bos_if_needed("<|im_start|>user", SimpleNamespace(bos_token=None), add_bos=True) == "<|im_start|>user"
    )


def test_encoder_rl_prompt_accepts_authored_marker_position(monkeypatch):
    """One authored marker governs policy and rollout prompts alike."""
    pytest.importorskip("nemo_rl", reason="NeMo RL prompt transport tests need the framework installed")
    import nemo_rl.data.llm_message_utils as message_utils

    from nemotron_stitch.nemo_rl import data as encoder_data

    monkeypatch.setattr(
        encoder_data,
        "load_encoder_payload",
        lambda _datum, _adapter: (torch.arange(6).reshape(2, 3), torch.arange(8).reshape(2, 4)),
    )

    def formatted_message_log(messages, tokenizer, *_args, **_kwargs):
        count = messages[-1]["content"].count("<slot>")
        return [
            {
                "role": "user",
                "content": messages[-1]["content"],
                "token_ids": torch.tensor([1, *([tokenizer.placeholder_token_id] * count), 2]),
            }
        ]

    monkeypatch.setattr(message_utils, "get_formatted_message_log", formatted_message_log)

    class _Tokenizer:
        bos_token = "<bos>"
        placeholder_token_id = 32

        def convert_tokens_to_ids(self, _token):
            return self.placeholder_token_id

        def apply_chat_template(self, messages, **_kwargs):
            return messages[-1]["content"]

    common = {
        "adapter": "sample",
        "encoder_placeholder_token": "<slot>",
        "encoder_start_token": "<start>",
        "encoder_end_token": "<end>",
        "task_name": "test",
    }

    embedded = encoder_data.encoder_rl_processor(
        {"sample_id": "row-embedded", "question": "Before?\n{mm:sample}\nAfter", **common},
        None,
        _Tokenizer(),
        max_seq_length=64,
        idx=0,
        add_bos=False,
    )
    # The policy prompt keeps the authored soft-token position, and the rollout
    # prompt carries the generation sentinel there with no literal marker left.
    assert [message["content"] for message in embedded["message_log"] if message["role"] == "user"] == [
        "Before?\n<start><slot><slot><end>\nAfter"
    ]
    assert embedded["vllm_content"] == "Before?\n<sample>\nAfter"

    plain = encoder_data.encoder_rl_processor(
        {"sample_id": "row-plain", "question": "question", **common},
        None,
        _Tokenizer(),
        max_seq_length=64,
        idx=1,
        add_bos=False,
    )
    # Marker-free questions retain the conventional modality-prefix layout.
    assert plain["vllm_content"] == "<sample>\nquestion"

    # The rollout payload is one modality item containing all soft tokens,
    # not T separate items.
    embedded_payload = embedded["vllm_multi_modal_data"]["sample"]
    assert len(embedded_payload) == 1
    assert torch.equal(embedded_payload[0], torch.arange(8).reshape(2, 4))
    assert plain["vllm_multi_modal_data"]["sample"][0].shape == (2, 4)


def test_variable_length_payloads_remain_logical_rows_and_materialize_flat(monkeypatch):
    pytest.importorskip("nemo_rl", reason="NeMo RL packing tests need the framework installed")
    from nemo_rl.data.llm_message_utils import batched_message_log_to_flat_message
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

    from nemotron_stitch.nemo_rl import data as encoder_data

    features = [torch.arange(6).reshape(2, 3), torch.arange(15).reshape(5, 3) + 100]
    projected = [torch.arange(8).reshape(2, 4), torch.arange(20).reshape(5, 4) + 100]

    monkeypatch.setattr(
        encoder_data,
        "load_encoder_payload",
        lambda datum, _adapter: (features[datum["row"]], projected[datum["row"]]),
    )

    def formatted_message_log(messages, tokenizer, *_args, **_kwargs):
        count = messages[-1]["content"].count("<slot>")
        return [
            {
                "role": "user",
                "content": messages[-1]["content"],
                "token_ids": torch.tensor([1, *([tokenizer.placeholder_token_id] * count), 2]),
            }
        ]

    import nemo_rl.data.llm_message_utils as message_utils

    monkeypatch.setattr(message_utils, "get_formatted_message_log", formatted_message_log)

    class _Tokenizer:
        bos_token = "<bos>"
        placeholder_token_id = 32

        def convert_tokens_to_ids(self, _token):
            return self.placeholder_token_id

        def apply_chat_template(self, messages, **_kwargs):
            return messages[-1]["content"]

    tokenizer = _Tokenizer()
    rows = []
    for row in range(2):
        rows.append(
            encoder_data.encoder_rl_processor(
                {
                    "row": row,
                    "sample_id": f"row-{row}",
                    "adapter": "sample",
                    "question": "question",
                    "encoder_placeholder_token": "<slot>",
                    "encoder_start_token": "<start>",
                    "encoder_end_token": "<end>",
                    "task_name": "test",
                },
                None,
                tokenizer,
                max_seq_length=32,
                idx=row,
                add_bos=False,
            )
        )

    repeated = BatchedDataDict(message_log=[row["message_log"] for row in rows]).repeat_interleave(
        2, share_immutable_media=True
    )
    flat, _ = batched_message_log_to_flat_message(
        repeated["message_log"],
        pad_value_dict={"token_ids": 0},
    )

    policy = flat["mm_features__sample"]
    assert len(policy) == 4
    assert len(policy.tensors) == 2
    assert [tuple(policy.slice([row]).as_tensor().shape) for row in range(4)] == [
        (2, 3),
        (2, 3),
        (5, 3),
        (5, 3),
    ]
    materialized = flat.get_multimodal_dict(as_tensors=True)
    assert materialized["mm_features__sample"].shape == (14, 3)
    for row, value in enumerate(projected):
        assert rows[row]["vllm_multi_modal_data"]["sample"][0] is value

    training = BatchedDataDict(
        input_ids=flat["token_ids"],
        **{"mm_features__sample": policy},
    )
    training.micro_batch_indices = [[(0, 3), (3, 4)]]
    training.micro_batch_lengths = [[flat["token_ids"].shape[1]] * 2]
    microbatches = list(training.make_microbatch_iterator_with_dynamic_shapes())
    assert [
        tuple(batch.get_multimodal_dict(as_tensors=True)["mm_features__sample"].shape) for batch in microbatches
    ] == [(9, 3), (5, 3)]


def test_upstream_collators_and_formatter_preserve_projected_payload():
    """The adopted NeMo RL path carries one opaque custom modality end to end."""
    pytest.importorskip("nemo_rl", reason="NeMo RL multimodal tests need the framework installed")
    from nemo_rl.data.collate_fn import eval_collate_fn, rl_collate_fn
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    from nemo_rl.models.generation.vllm.utils import format_prompt_for_vllm_generation

    projected = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
    datum = {
        "message_log": [{"role": "user", "content": "question", "token_ids": torch.tensor([1, 2])}],
        "length": 2,
        "loss_multiplier": 1.0,
        "vllm_content": "<sample>\nquestion",
        "vllm_multi_modal_data": {"sample": [projected]},
        "extra_env_info": {"sample_id": "row-0"},
        "idx": 0,
        "task_name": "sample",
        "stop_strings": None,
    }

    for collate in (rl_collate_fn, eval_collate_fn):
        batch = collate([datum])
        assert batch["vllm_multi_modal_data"][0]["sample"][0] is projected

    prompt = format_prompt_for_vllm_generation(
        BatchedDataDict(
            input_ids=torch.tensor([[1, 2]]),
            input_lengths=torch.tensor([2]),
            vllm_content=[datum["vllm_content"]],
            vllm_multi_modal_data=[datum["vllm_multi_modal_data"]],
        )
    )[0]
    assert prompt["prompt"] == datum["vllm_content"]
    assert prompt["multi_modal_data"]["sample"] == [projected]
    if importlib.util.find_spec("vllm"):
        # One modality item holds the whole soft-token run; a bare
        # rank-two tensor would count as T items and trip the item limit.
        from vllm.multimodal.parse import EmbeddingItems

        assert len(EmbeddingItems(prompt["multi_modal_data"]["sample"], "sample", 4)) == 1
