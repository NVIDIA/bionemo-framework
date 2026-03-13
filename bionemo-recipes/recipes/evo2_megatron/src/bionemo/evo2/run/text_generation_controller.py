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

"""Evo2TextGenerationController — adds prompt segmentation threshold (PST) support.

Stock megatron-core (0.16.0rc0) ``TextGenerationController`` hard-codes the initial
``context_end_position`` for prefill to ``min_prompt_length_in_batch``.  This means
the entire prompt is processed in one forward pass, which can OOM on long prompts.

This subclass overrides ``generate_all_output_tokens_static_batch`` to read an
optional ``prompt_segmentation_threshold`` attribute from the
``InferenceWrapperConfig``.  When set, the first prefill covers only ``pst`` tokens
and the remaining prompt tokens are processed one-at-a-time (decode speed) before
normal generation begins.

The override is a verbatim copy of the megatron-core 0.16.0rc0 method (commit
``bbbedbb9f53``, installed via Megatron-Bridge ``549e3cb``) with only the
``context_end_position`` initialisation changed (marked ``# --- PST ---``).

The method body is wrapped in ``# fmt: off`` and lint suppressions to keep it as
close to the upstream source as possible, simplifying future diffs when upgrading
megatron-core.
"""

# ruff: noqa: N812, C417, RUF015, F841, D417

import concurrent.futures
import copy
import functools
from collections import defaultdict
from typing import Any, Dict, List, Optional, OrderedDict

import torch
import torch.nn.functional as F
from megatron.core.inference.async_stream import AsyncStream
from megatron.core.inference.communication_utils import (
    broadcast_from_last_pipeline_stage,
    is_pipeline_last_stage,
)
from megatron.core.inference.contexts.dynamic_context import MaxSequenceLengthOverflowError
from megatron.core.inference.inference_request import InferenceRequest, Status
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.inference.utils import set_decode_expert_padding
from megatron.core.transformer.enums import CudaGraphScope
from megatron.core.transformer.moe.moe_layer import BaseMoELayer
from megatron.core.transformer.utils import set_model_to_sequence_parallel
from megatron.core.utils import get_model_config, unwrap_model


class Evo2TextGenerationController(TextGenerationController):
    """TextGenerationController with prompt segmentation threshold (PST) support.

    When ``prompt_segmentation_threshold`` is set on the ``InferenceWrapperConfig``
    (via ``config.add_attributes(...)``), prompts longer than the threshold are
    segmented during prefill to reduce peak activation memory.
    """

    # ------------------------------------------------------------------
    # Verbatim copy of megatron-core 0.16.0rc0
    # ``TextGenerationController.generate_all_output_tokens_static_batch``
    # with a single change at the ``# --- PST ---`` marker.
    # ------------------------------------------------------------------
    # fmt: off
    @torch.inference_mode()
    def generate_all_output_tokens_static_batch(
        self,
        active_requests: OrderedDict[int, InferenceRequest],
        active_streams: Optional[OrderedDict[str, AsyncStream]] = None,
    ) -> OrderedDict[int, InferenceRequest]:
        """Utility to generate all the output tokens and probabilities for the prompts.

        This utility generates the output tokens for a static batch. It runs the forward steps till
        all prompts complete generation, updates the status of these requests to completed, adds
        the generated result and returns these requests

        Args:
            active_requests (OrderedDict[int, InferenceRequest]): The input active requests.

        Returns:
            OrderedDict[int, InferenceRequest]: The result for each of the incoming requests
        """
        assert all(request.prompt_tokens is not None for request in active_requests.values())

        # Perform a deep copy so that the request prompt tokens do not get modified.
        batch_prompt_tokens_list: List[List[int]] = list(
            map(
                lambda request: copy.deepcopy(request.prompt_tokens),  # type: ignore[arg-type]
                active_requests.values(),
            )
        )
        prompt_lengths_in_batch = torch.tensor(
            [len(prompt_tokens) for prompt_tokens in batch_prompt_tokens_list],
            device=torch.cuda.current_device(),
        )
        max_prompt_length_in_batch = max(prompt_lengths_in_batch)
        min_prompt_length_in_batch = min(prompt_lengths_in_batch)

        # For batch inference the sampling params are the same for all request
        sampling_params: SamplingParams = list(active_requests.values())[0].sampling_params

        # Remove Float16Module wrapper if it exists
        unwrapped_model = unwrap_model(self.inference_wrapped_model.model)
        model_config = get_model_config(unwrapped_model)

        # We only need an attention mask if we are exclusively doing prefill over
        # prompts of variable length
        use_attention_mask = (
            sampling_params.num_tokens_to_generate == 0
            and min_prompt_length_in_batch != max_prompt_length_in_batch
        )

        # Check whether CUDA graphs are enabled
        enable_cuda_graph = (
            model_config.cuda_graph_impl == "local"
            and CudaGraphScope.full_iteration not in model_config.cuda_graph_scope
        )

        # Pad batch tokens if necessary
        batch_size = len(active_requests)
        max_sequence_length = max_prompt_length_in_batch + sampling_params.num_tokens_to_generate
        inference_wrapper_config = self.inference_wrapped_model.inference_wrapper_config
        inference_max_batch_size = inference_wrapper_config.inference_max_requests
        inference_max_sequence_length = inference_wrapper_config.inference_max_seq_length
        padded_batch_size = inference_max_batch_size if enable_cuda_graph else batch_size
        if padded_batch_size > inference_max_batch_size:
            raise ValueError(
                f"Padded batch size {padded_batch_size} > max batch size {inference_max_batch_size}"
            )
        padded_batch_prompt_tokens = self.pad_input_prompt_tokens(
            batch_prompt_tokens_list,
            padded_batch_size=padded_batch_size,
            padded_sequence_length=max_sequence_length,
        )

        # Verify that output sequence length is within configured limit
        if max_sequence_length > inference_max_sequence_length:
            raise MaxSequenceLengthOverflowError(
                f"Maximum allowed sequence length was set to {inference_max_sequence_length} "
                f"tokens but requested generation of {max_sequence_length} tokens"
            )

        top_n_logprobs_dict = defaultdict(list)

        # Pre allocate log probs tensor
        output_log_probs = None
        if sampling_params.return_log_probs:
            output_log_probs = torch.empty(
                (batch_size, max_sequence_length - 1),
                dtype=torch.float32,
                device=torch.cuda.current_device(),
            )

        # An array to check which of the prompts have reached end of generation condition
        is_generation_done_tensor = torch.zeros(
            batch_size, dtype=torch.bool, device=torch.cuda.current_device()
        )

        # An array to act as a counter to keep track of generated sequence lengths
        generated_sequence_lengths = torch.zeros(
            batch_size, device=torch.cuda.current_device()
        ).cuda()

        # Use padded vocab size because tokenizer vocab size might not include padding
        # to nearest power of 2
        vocab_size = inference_wrapper_config.padded_vocab_size

        # Check whether early termination is enabled
        no_early_termination = getattr(sampling_params, "no_early_termination", False)
        termination_id = -1 if no_early_termination else self.tokenizer.eod

        streaming_enabled = active_streams is not None and len(active_streams) > 0
        if streaming_enabled:
            # Start a separate thread for streaming tokens to avoid blocking the
            # main computation
            streaming_idx: List[int] = [
                i
                for (i, request_id) in enumerate(active_requests.keys())
                if request_id in active_streams
            ]
            streaming_request_ids: List[int] = list(active_streams.keys())
            streams: List[AsyncStream] = list(active_streams.values())
            streaming_requests: List[InferenceRequest] = [
                active_requests[request_id] for request_id in streaming_request_ids
            ]
            streaming_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            stream_tokens = functools.partial(self.stream_tokens, sampling_params)

        for request in active_requests.values():
            # Initialize to a list to store a latency measurement for each generated token.
            request.tpot = []
        timing_events = []

        with torch.inference_mode():
            self.inference_wrapped_model.prep_model_for_inference()

            inference_input: Dict[str, Any] = self.prep_inference_input(
                prompts_tokens=padded_batch_prompt_tokens,
                active_requests=active_requests,
                use_attention_mask=use_attention_mask,
            )

            assert (
                not self.inference_wrapped_model.inference_context.is_decode_only()
            ), "Generation must start in prefill mode"

            # Sequence parallelism is required for MoE layers when using expert parallelism (EP)
            # becausethe expert routing mechanism relies on sequence parallelism's communication
            # infrastructure to distribute tokens across expert ranks. However, sequence parallelism
            # is not currently supported for non-MoE layers during inference, so we selectively
            # disable it for all other layer types. This is safe because MoE layers perform an
            # all-gather operation on sequences before passing data to subsequent layers, ensuring
            # that each rank has the complete sequence data needed for the next non-MoE layer.
            tp_size = model_config.tensor_model_parallel_size
            ep_size = model_config.expert_model_parallel_size
            model_is_tp_ep = tp_size > 1 and ep_size > 1
            if model_is_tp_ep:
                set_model_to_sequence_parallel(
                    unwrapped_model, False, exclude_modules=[BaseMoELayer]
                )
            elif model_config.sequence_parallel and (ep_size == 1 or tp_size == 1):
                raise NotImplementedError(
                    "Sequence parallellism is only supported for static batching with MoE models"
                )

            # If using symmetric kernels and we are using using nccl
            # for prefill turn off symmetric kernels
            symmetric_ar_type = model_config.symmetric_ar_type
            nccl_all_reduce_for_prefill = inference_wrapper_config.nccl_all_reduce_for_prefill
            if symmetric_ar_type is not None and nccl_all_reduce_for_prefill:
                unwrapped_model.set_symmetric_ar(None)

            # Turning off MoE padding for prefill
            moe_pad_experts_for_cuda_graph_inference = (
                inference_wrapper_config.moe_pad_experts_for_cuda_graph_inference
            )
            if moe_pad_experts_for_cuda_graph_inference:
                set_decode_expert_padding(unwrapped_model, False)

            context_start_position = 0

            # If we are exclusively doing prefill then we can process all prompt tokens
            # together even if the prompt lengths are different
            if sampling_params.num_tokens_to_generate == 0:
                context_end_position = max_prompt_length_in_batch
            else:
                # --- PST ---
                # Stock megatron-core uses: context_end_position = min_prompt_length_in_batch
                # With PST, we cap the initial prefill to reduce peak activation memory.
                pst = getattr(inference_wrapper_config, 'prompt_segmentation_threshold', None)
                pst = min(pst or min_prompt_length_in_batch, min_prompt_length_in_batch)
                context_end_position = pst
                # --- end PST ---

            # The initial iteration of this loop runs the prefill phase up to the shortest
            # prompt length in the batch. Then every subsequent iterations runs a decode step.
            # At least one new token will be generated in each iteration. The generated token
            # will be ignored for requests which have prompt length > the current generated
            # sequence length. Similarly, the generated token is ignored for requests which
            # have maximum total sequence length < the current generated sequence length.
            while True:
                # Add a timing event at the start of each iteration. The token generation
                # time will be the elapsed time between consective timing events.
                timing_events.append(torch.cuda.Event(enable_timing=True))
                timing_events[-1].record()

                # Pick the context window that we need to pass through the network.
                inference_input_for_context_window: Dict[str, Any] = (
                    self.inference_wrapped_model.get_batch_for_context_window(
                        inference_input, context_start_position, context_end_position
                    )
                )

                # Disable attention mask when using CUDA graphs for decode
                if (
                    enable_cuda_graph
                    and self.inference_wrapped_model.inference_context.is_decode_only()
                    and "attention_mask" in inference_input_for_context_window
                ):
                    inference_input_for_context_window["attention_mask"] = None
                elif use_attention_mask:
                    assert (
                        attention_mask := inference_input_for_context_window.get(
                            "attention_mask", None
                        )
                        is not None
                    )

                # Only materialize prompt log probs if the user requests log probs
                materialize_only_last_token_logits = (
                    self.inference_wrapped_model.inference_context.is_decode_only()
                    or not (sampling_params.return_log_probs or sampling_params.top_n_logprobs > 0)
                )
                inference_context = self.inference_wrapped_model.inference_context
                inference_context.materialize_only_last_token_logits = (
                    materialize_only_last_token_logits
                )

                # Returns the final logits of shape [batch_size, context_length, vocab_size]
                # Note: This is returned in all TP ranks or last PP stage in PP models
                logits = self.inference_wrapped_model.run_one_forward_step(
                    inference_input_for_context_window
                )

                # Undo padding if necessary
                batch_prompt_tokens = self.unpad_input_prompt_tokens(
                    padded_batch_prompt_tokens, batch_size
                )
                assert batch_prompt_tokens.shape[0] == batch_size, batch_prompt_tokens.shape[0]
                if is_pipeline_last_stage(self.pp_group):
                    logits = logits[:batch_size]

                if self.model_is_pipeline_parallel:
                    context_length = context_end_position - context_start_position
                    logits_seq_len = 1 if materialize_only_last_token_logits else context_length
                    logits_shape = [batch_size, logits_seq_len, vocab_size]
                    if is_pipeline_last_stage(self.pp_group):
                        assert logits is not None and torch.Size(logits_shape) == logits.shape
                    # TODO(ksanthanam): Evaluate whether it makes more sense to sample on 1 rank
                    # and then broadcast the sampled tokens rather than broadcasting the raw logits.
                    logits = broadcast_from_last_pipeline_stage(
                        [batch_size, logits_seq_len, vocab_size],
                        dtype=inference_wrapper_config.params_dtype,
                        tensor=logits,
                        pp_group=self.pp_group,
                    )

                # Turn on symmetric all reduce kernels for decode stage
                # if we turned it off for prefill
                if (
                    context_end_position == min_prompt_length_in_batch
                    and symmetric_ar_type is not None
                    and nccl_all_reduce_for_prefill
                ):
                    if symmetric_ar_type is not None and nccl_all_reduce_for_prefill:
                        unwrapped_model.set_symmetric_ar(symmetric_ar_type)

                # Indicates which of the input prompts have started generating tokens.
                # A 1D boolean tensor with [batch_size] elements (i.e) The shortest
                # prompts will start generating first and so on
                generation_started = prompt_lengths_in_batch <= context_end_position
                last_token_logits = logits[:, -1, :]

                logits_for_top_n_prompt_logprobs = (
                    logits
                    if context_start_position == 0 and not sampling_params.skip_prompt_log_probs
                    else None
                )
                sampled_logits = self.sample_from_logits(
                    last_token_logits,
                    sampling_params,
                    vocab_size,
                    generation_started=generation_started,
                    top_n_logprobs_dict=top_n_logprobs_dict,
                    logits=logits_for_top_n_prompt_logprobs,
                )

                if sampling_params.num_tokens_to_generate > 0:
                    # Substitute the sampled logits only for the prompts that
                    # have started generating tokens
                    batch_prompt_tokens[generation_started, context_end_position] = sampled_logits[
                        generation_started
                    ]

                # Compute log probs
                if sampling_params.return_log_probs:
                    log_probs = F.log_softmax(logits, dim=2).to(torch.float32)

                    indices = torch.unsqueeze(
                        batch_prompt_tokens[
                            :, (context_start_position + 1) : (context_end_position + 1)
                        ],
                        2,
                    )
                    # Get the log probabilities for only the prompt tokens
                    assert output_log_probs is not None
                    output_log_probs[:, context_start_position:context_end_position] = torch.gather(
                        log_probs, 2, indices
                    ).squeeze(2)

                context_start_position = context_end_position

                if sampling_params.num_tokens_to_generate > 0:
                    # Check end of generation status for each tensor
                    # and update generated sequence lengths
                    (is_generation_done_tensor, generated_sequence_lengths) = (
                        self.update_generation_status(
                            updated_prompts_tokens=batch_prompt_tokens,
                            generation_started=generation_started,
                            current_context_end_position=context_end_position,
                            is_generation_done_tensor=is_generation_done_tensor,
                            generated_sequence_lengths=generated_sequence_lengths,
                            termination_id=termination_id,
                        )
                    )

                    # Stream intermediate outputs
                    if streaming_enabled:
                        streaming_executor.submit(
                            stream_tokens,
                            streaming_request_ids,
                            streaming_requests,
                            streams,
                            generation_started[streaming_idx].cpu(),
                            is_generation_done_tensor[streaming_idx].cpu(),
                            batch_prompt_tokens[streaming_idx].cpu(),
                            prompt_lengths_in_batch[streaming_idx].cpu(),
                            generated_sequence_lengths[streaming_idx].cpu(),
                            (
                                output_log_probs[streaming_idx].cpu()
                                if output_log_probs is not None
                                else [None] * len(streaming_idx)
                            ),
                        )

                # Boolean flag indicating if all prompts are finished
                all_prompts_done = torch.all(is_generation_done_tensor)
                if all_prompts_done:
                    break

                # Change to decode mode if all prefill is complete
                if torch.all(generation_started):
                    self.inference_wrapped_model.inference_context.enable_decode_mode()
                    # Turn on padding for decode if flag set
                    if moe_pad_experts_for_cuda_graph_inference:
                        capacity_factor = (
                            model_config.num_moe_experts / model_config.moe_router_topk
                        )
                        set_decode_expert_padding(
                            unwrapped_model, True, capacity_factor=capacity_factor
                        )

                context_end_position = context_start_position + 1
                if context_end_position >= max_sequence_length:
                    break

        # Add a final timing event to compute the latency of every loop iteration
        timing_events.append(torch.cuda.Event(enable_timing=True))
        timing_events[-1].record()

        # Close all streams
        if streaming_enabled:
            streaming_executor.shutdown()
            for stream in streams:
                stream.finish()

        # Include all the generated tokens
        batch_prompt_tokens_with_generations = padded_batch_prompt_tokens[
            :batch_size, : (context_end_position + 1)
        ]
        if sampling_params.return_log_probs:
            assert output_log_probs is not None
            output_log_probs = output_log_probs[:, :context_end_position]

        generated_sequence_lengths[
            generated_sequence_lengths > sampling_params.num_tokens_to_generate
        ] = sampling_params.num_tokens_to_generate

        timing_events[-1].synchronize()
        tpot = torch.tensor(
            [
                timing_events[i].elapsed_time(timing_events[i + 1]) / 1e3
                for i in range(len(timing_events) - 1)
            ],
            dtype=torch.float32,
        )

        for idx, request in enumerate(active_requests.values()):
            input_prompt_length = int(prompt_lengths_in_batch[idx])
            # Shorter prompts might have generated more than required tokens. So we trim them down
            required_sequence_length = int(
                min(generated_sequence_lengths[idx], sampling_params.num_tokens_to_generate)
            )
            # Extract only the generated tokens
            required_result_tokens = batch_prompt_tokens_with_generations[
                idx, input_prompt_length : (input_prompt_length + required_sequence_length)
            ]
            request.generated_sequence_lengths = generated_sequence_lengths[idx].to(dtype=torch.int32)
            request.generated_length = required_sequence_length
            request.generated_tokens = required_result_tokens

            # Record the decode latencies for only the generated tokens
            request_tpot = tpot.clone()
            # Sum up the latencies of the first prompt tokens if the
            # request prompt length > minimum prompt length
            spill_length = input_prompt_length - min_prompt_length_in_batch
            if spill_length > 0:
                spill_latency = request_tpot[:spill_length].sum()
                request_tpot = torch.cat((spill_latency.unsqueeze(0), request_tpot[spill_length:]))

            # Remove the extraneous latencies if the
            # request sequence length < maximum sequence length
            request_tpot = request_tpot[:required_sequence_length]
            request.tpot = request_tpot.tolist()

            if output_log_probs is not None:
                request.prompt_log_probs = output_log_probs[idx, : input_prompt_length - 1].tolist()
                request.generated_log_probs = output_log_probs[
                    idx,
                    input_prompt_length - 1 : (input_prompt_length + required_sequence_length - 1),
                ].tolist()
            if sampling_params.top_n_logprobs > 0:
                if not sampling_params.skip_prompt_log_probs:
                    assert (
                        len(top_n_logprobs_dict[idx])
                        >= input_prompt_length + required_sequence_length - 1
                    ), (
                        "Did not collect required number of top-N logprobs: "
                        f"{len(top_n_logprobs_dict[idx])}"
                    )
                    request.prompt_top_n_logprobs = top_n_logprobs_dict[idx][
                        : input_prompt_length - 1
                    ]
                    request.generated_top_n_logprobs = top_n_logprobs_dict[idx][
                        input_prompt_length
                        - 1 : (input_prompt_length + required_sequence_length - 1)
                    ]
                else:
                    assert len(top_n_logprobs_dict[idx]) >= required_sequence_length, (
                        "Did not collect required number of top-N logprobs: "
                        f"{len(top_n_logprobs_dict[idx])}"
                    )
                    request.generated_top_n_logprobs = top_n_logprobs_dict[idx][
                        :required_sequence_length
                    ]

            request.status = Status.COMPLETED

            text, segments = self.detokenize_generations(
                batch_prompt_tokens_with_generations[
                    idx, : (input_prompt_length + required_sequence_length)
                ],
                input_prompt_length + generated_sequence_lengths[idx],
                sampling_params.return_segments,
            )
            request.text = text  # Inference server returns prompts & generations together
            if sampling_params.return_segments:
                request.segments = segments[0]
            request.generated_text = text[len(request.prompt) :]
        return active_requests
    # fmt: on
