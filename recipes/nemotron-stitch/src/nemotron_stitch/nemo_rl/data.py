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

"""NeMo RL dataset using modality-neutral packed encoder tensors."""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Any

import torch

from nemotron_stitch.nemo_rl.transport import load_encoder_payload, resolve_callback, task_reward_metadata
from nemotron_stitch.prompt import PlaceholderSpec, render_soft_token_prompt

POLICY_PREFIX = "mm_features__"


class ChatTemplateKwargsProxy:
    """Apply a datum's ``chat_template_kwargs`` to every ``apply_chat_template`` call.

    NeMo RL e98357616c0bb8fb13140841e225583b5cc36b0e builds policy message-log
    token ids through ``get_formatted_message_log`` with fixed template kwargs
    — there is no channel for flags like ``enable_thinking``. Policy and
    rollout prompts must tokenize identically, so the processor applies the
    datum's template kwargs uniformly through this proxy. Explicit call-site
    kwargs (``add_generation_prompt`` etc.) win. U-21: delete when upstream
    threads chat_template_kwargs through TaskDataSpec / llm_message_utils.
    """

    def __init__(self, tokenizer: Any, chat_template_kwargs: dict[str, Any]) -> None:
        self._tokenizer = tokenizer
        self._chat_template_kwargs = dict(chat_template_kwargs)

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> Any:
        merged = {**self._chat_template_kwargs, **kwargs}
        return self._tokenizer.apply_chat_template(messages, **merged)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._tokenizer(*args, **kwargs)

    @property
    def tokenizer(self) -> Any:
        # NeMo RL 19244a0949fb363326f71675223da75e09d14866 sends
        # non-PreTrainedTokenizerBase wrappers through
        # get_multimodal_keys_from_processor and expects processor.tokenizer.
        # Delete with this proxy when upstream carries per-datum template kwargs.
        return self._tokenizer

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tokenizer, name)


def _encoder_prompt_template(question: str, adapter: str) -> str:
    """Preserve an authored soft-token marker; otherwise supply the prefix.

    RL questions may embed one ``{mm:<adapter>}`` marker at their chosen
    position (the frozen balanced pool does). Marker-free questions keep the
    conventional modality-prefix layout. Wrong-projector names, duplicates, and
    missing markers still fail closed in the public renderer, which validates
    marker identity and multiplicity.
    """
    return question if "{mm:" in question else f"{{mm:{adapter}}}\n" + question


def _render_policy_content(
    question: str,
    token_count: int,
    *,
    adapter: str,
    placeholder_token: str,
    start_token: str,
    end_token: str,
) -> str:
    # The placeholder spec and template marker are keyed by the adapter name so
    # the consumer names the modality (the LLaVA example passes ``image``) — the
    # package holds no modality names.
    return render_soft_token_prompt(
        _encoder_prompt_template(question, adapter),
        [PlaceholderSpec(adapter, start_token, placeholder_token, end_token)],
        {adapter: token_count},
    )


def prepend_bos_if_needed(text: str, tokenizer: Any, add_bos: bool) -> str:
    """Match get_formatted_message_log's BOS prefix on the rollout prompt.

    NeMo RL's ``get_formatted_message_log`` prepends the tokenizer's BOS token
    to the first message chunk when ``add_bos_token`` is set. The rollout
    prompt must carry the same prefix or every token position — the soft-token
    slots and the whole response context — shifts by one token.
    """
    if add_bos and tokenizer.bos_token and not str(text).startswith(str(tokenizer.bos_token)):
        return str(tokenizer.bos_token) + str(text)
    return str(text)


def encoder_rl_processor(
    datum_dict: dict[str, Any],
    task_data_spec,
    tokenizer,
    max_seq_length: int,
    idx: int,
    add_bos: bool = True,
    add_eos: bool = False,
    add_generation_prompt: bool = True,
):
    """Create one prompt with raw policy features and projected rollout tokens."""
    from nemo_rl.data.llm_message_utils import get_formatted_message_log
    from nemo_rl.data.multimodal_utils import PackedTensor

    adapter = str(datum_dict["adapter"])
    features, projected = load_encoder_payload(datum_dict, adapter)
    token_count = int(projected.shape[0])
    template_kwargs = dict(datum_dict.get("chat_template_kwargs") or {})
    if template_kwargs:
        tokenizer = ChatTemplateKwargsProxy(tokenizer, template_kwargs)
    # L-3: validate the projected rollout tensor and its sentinel at data
    # construction instead of on every vLLM generation call.
    if projected.ndim != 2 or token_count <= 0:
        raise ValueError(f"projected encoder payload {adapter!r} must be [T,H], got {tuple(projected.shape)}")
    placeholder_token = str(datum_dict["encoder_placeholder_token"])
    placeholder_token_id = int(tokenizer.convert_tokens_to_ids(placeholder_token))
    marker = f"<{adapter}>"
    policy_content = _render_policy_content(
        str(datum_dict["question"]),
        token_count,
        adapter=adapter,
        placeholder_token=placeholder_token,
        start_token=str(datum_dict["encoder_start_token"]),
        end_token=str(datum_dict["encoder_end_token"]),
    )
    messages = []
    if datum_dict.get("system_prompt"):
        messages.append({"role": "system", "content": str(datum_dict["system_prompt"])})
    messages.append({"role": "user", "content": policy_content})
    message_log = get_formatted_message_log(
        messages,
        tokenizer,
        task_data_spec,
        add_bos_token=add_bos,
        add_eos_token=add_eos,
        add_generation_prompt=add_generation_prompt,
    )
    user_messages = [message for message in message_log if message["role"] == "user"]
    if len(user_messages) != 1:
        raise RuntimeError("encoder prompts require exactly one user message")

    token_ids = user_messages[0]["token_ids"]
    placeholder_count = int(token_ids.eq(placeholder_token_id).sum())
    if placeholder_count != token_count:
        raise RuntimeError(f"policy prompt contains {placeholder_count} encoder slots, expected {token_count}")
    # Each wrapper contains one logical row. Canonical flat token matrices pack
    # directly along N; structured projector inputs retain their leading batch
    # dimension. Both forms keep row boundaries for GRPO slicing.
    policy_features = features if features.ndim == 2 else features.unsqueeze(0)
    user_messages[0][POLICY_PREFIX + adapter] = PackedTensor(policy_features, dim_to_pack=0)

    rollout_messages = []
    if datum_dict.get("system_prompt"):
        rollout_messages.append({"role": "system", "content": str(datum_dict["system_prompt"])})
    rollout_messages.append(
        {
            "role": "user",
            # The rollout sentinel takes the marker's authored position; a
            # literal ``{mm:...}`` must never reach the generation prompt.
            "content": _encoder_prompt_template(str(datum_dict["question"]), adapter).replace(
                f"{{mm:{adapter}}}", marker
            ),
        }
    )
    rollout_content = tokenizer.apply_chat_template(
        rollout_messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    # The policy message log prepends the tokenizer's BOS token to the first
    # message chunk (get_formatted_message_log's add_bos_token above). The
    # rollout prompt must carry the same prefix or every token position — the
    # soft-token slots and the whole response context — shifts by one, which
    # surfaces as a value-independent policy/rollout log-probability mismatch.
    rollout_content = prepend_bos_if_needed(rollout_content, tokenizer, add_bos)
    # The vLLM formatter requires exactly one sentinel per projector; enforce that
    # invariant once here rather than per generation call.
    if str(rollout_content).count(marker) != 1:
        raise ValueError(
            f"rollout prompt must contain exactly one {marker} sentinel, got "
            f"{str(rollout_content).count(marker)}: {str(rollout_content)!r}"
        )
    length = sum(len(message["token_ids"]) for message in message_log)
    reward_metadata = task_reward_metadata(datum_dict)
    return {
        "message_log": message_log,
        "length": length,
        "extra_env_info": {
            "sample_id": datum_dict["sample_id"],
            "question": str(datum_dict["question"]),
            **reward_metadata,
        },
        "loss_multiplier": float(length <= int(max_seq_length)),
        "idx": idx,
        "task_name": datum_dict["task_name"],
        "stop_strings": None,
        "vllm_content": rollout_content,
        # One item containing T soft tokens, not T separate modality items:
        # vLLM's EmbeddingItems counts a rank-two tensor's rows as
        # separate items, so the projected payload must arrive as a one-item
        # list. Upstream formatting preserves this vLLM-ready item list.
        "vllm_multi_modal_data": {adapter: [projected]},
    }


def encoder_preference_processor(
    datum_dict: dict[str, Any],
    task_data_spec: Any,
    tokenizer: Any,
    max_seq_length: int,
    idx: int,
) -> dict[str, Any]:
    """Attach the same encoder features to both DPO branches."""
    from nemo_rl.data.multimodal_utils import PackedTensor
    from nemo_rl.data.processors import preference_preprocessor

    adapter = str(datum_dict["adapter"])
    features = resolve_callback(str(datum_dict["encoder_loader"]))(
        datum_dict, adapter, **dict(datum_dict.get("encoder_loader_kwargs") or {})
    )
    if not isinstance(features, torch.Tensor) or features.ndim != 2 or features.shape[0] <= 0:
        raise ValueError("preference encoder loader must return nonempty [tokens, hidden] features")
    placeholder_id = int(tokenizer.convert_tokens_to_ids(datum_dict["encoder_placeholder_token"]))
    template_kwargs = dict(datum_dict.get("chat_template_kwargs") or {})
    rendered_tokenizer = ChatTemplateKwargsProxy(tokenizer, template_kwargs) if template_kwargs else tokenizer
    result = preference_preprocessor(datum_dict, task_data_spec, rendered_tokenizer, max_seq_length, idx)
    if result["loss_multiplier"] != 1.0:
        raise ValueError("preference branch exceeds max_seq_length; feature positions would be truncated")
    for branch in ("message_log_chosen", "message_log_rejected"):
        messages = result[branch]
        positions = sum(int(message["token_ids"].eq(placeholder_id).sum()) for message in messages)
        if positions != features.shape[0]:
            raise ValueError(f"{branch} has {positions} soft-token slots, expected {features.shape[0]}")
        user_messages = [message for message in messages if message["role"] == "user"]
        if len(user_messages) != 1:
            raise ValueError(f"{branch} requires exactly one user message")
        user_messages[0][POLICY_PREFIX + adapter] = PackedTensor(features, dim_to_pack=0)
    return result


class EncoderRLDataset:
    """Small JSONL response dataset resolved by dotted path in NeMo RL."""

    def __init__(
        self,
        data_path: str,
        adapter: str,
        encoder_loader: str,
        encoder_loader_kwargs: dict[str, Any],
        placeholder_token: str,
        start_token: str,
        end_token: str,
        task_metadata_callback: str | None = None,
        task_name: str = "encoder_task",
        system_prompt: str | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        add_bos_token: bool = True,
        split_validation_size: float = 0,
        seed: int | None = None,
        **_: Any,
    ) -> None:
        if split_validation_size:
            raise ValueError("encoder RL uses explicit manifests; implicit validation splits are disabled")
        rows = [json.loads(line) for line in Path(data_path).read_text().splitlines() if line.strip()]
        if not rows:
            raise ValueError(f"empty encoder RL manifest: {data_path}")
        self.task_name = task_name
        self.seed = seed
        self.add_bos_token = bool(add_bos_token)
        self.dataset = [
            {
                **row,
                "adapter": adapter,
                "encoder_loader": encoder_loader,
                "encoder_loader_kwargs": dict(encoder_loader_kwargs),
                "task_metadata_callback": task_metadata_callback,
                "encoder_placeholder_token": placeholder_token,
                "encoder_start_token": start_token,
                "encoder_end_token": end_token,
                "system_prompt": system_prompt,
                # The rollout prompt bypasses NeMo RL's normal message renderer,
                # so explicitly preserve the policy template mode.
                "chat_template_kwargs": dict(chat_template_kwargs or {}),
                "task_name": self.task_name,
            }
            for row in rows
        ]
        self.val_dataset = None
        self.preprocessor = None

    def set_task_spec(self, data_config: dict[str, Any]) -> None:
        from nemo_rl.data.interfaces import TaskDataSpec

        if data_config.get("prompt_file"):
            raise ValueError("encoder RL does not support prompt_file; prompts are rendered from manifest rows")
        self.data_config = data_config
        self.task_spec = TaskDataSpec(
            task_name=self.task_name,
            prompt_file=None,
            system_prompt_file=data_config.get("system_prompt_file"),
        )

    def set_processor(self) -> None:
        self.processor = partial(encoder_rl_processor, add_bos=self.add_bos_token)
