# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Arc Institute. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Michael Poli. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Stanford University. All rights reserved
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

import nemo.lightning as nl
import torch
from megatron.core.inference.common_inference_params import CommonInferenceParams
from nemo.collections.llm import generate

from bionemo.core.data.load import load
from bionemo.testing.megatron_parallel_state_utils import clean_parallel_state_context


RANDOM_SEED = 42


def test_infer_model_generates_expected_single_token_output():
    # Create PTL trainer.
    TENSOR_PARALLEL_SIZE = 1
    PIPELINE_MODEL_PARALLEL_SIZE = 1
    CONTEXT_PARALLEL_SIZE = 1
    NUM_GPUS = 1
    NUM_NODES = 1

    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_model_parallel_size=PIPELINE_MODEL_PARALLEL_SIZE,
        context_parallel_size=CONTEXT_PARALLEL_SIZE,
        pipeline_dtype=torch.bfloat16,
        ckpt_load_optimizer=False,  # Needs to be false for a normal model checkpoint.
        ckpt_save_optimizer=False,
        ckpt_async_save=False,
        save_ckpt_format="torch_dist",
        ckpt_load_strictness="log_all",
    )
    trainer = nl.Trainer(
        accelerator="gpu",
        num_nodes=NUM_NODES,
        devices=NUM_GPUS,
        strategy=strategy,
        log_every_n_steps=1,
        limit_val_batches=10,
        num_sanity_val_steps=0,
        plugins=nl.MegatronMixedPrecision(
            precision="bf16-mixed",
            params_dtype=torch.bfloat16,
        ),
    )

    prompt = (
        "|d__Bacteria;"
        + "p__Pseudomonadota;"
        + "c__Gammaproteobacteria;"
        + "o__Enterobacterales;"
        + "f__Enterobacteriaceae;"
        + "g__Escherichia;"
        + "s__Escherichia|"
    )
    temperature = 1.0
    top_k = 0
    top_p = 0.0
    max_new_tokens = 1
    # TODO (dorotat) remove PBSS source once the model is available on NGC
    checkpoint_path = load("evo2/7b-8k:1.0", source="pbss")

    with clean_parallel_state_context():
        results = generate(
            path=checkpoint_path,
            prompts=[prompt],
            trainer=trainer,
            inference_params=CommonInferenceParams(
                temperature,
                top_k,
                top_p,
                return_log_probs=False,
                num_tokens_to_generate=max_new_tokens,
            ),
            random_seed=RANDOM_SEED,
            text_only=True,
        )

        assert isinstance(results, list)
        assert results == ["T"]


# def test_infer_model_generates_expected_single_token_output_from_input_seq():
#     # Create PTL trainer.
#     # TODO: Uncomment when the GPU Memory allocation issue is resolved.
#     _teardown_apex_megatron_cuda()
#     torch.cuda.empty_cache()
#     TENSOR_PARALLEL_SIZE = 1
#     PIPELINE_MODEL_PARALLEL_SIZE = 1
#     CONTEXT_PARALLEL_SIZE = 1
#     NUM_GPUS = 1
#     NUM_NODES = 1

#     strategy = nl.MegatronStrategy(
#         tensor_model_parallel_size=TENSOR_PARALLEL_SIZE,
#         pipeline_model_parallel_size=PIPELINE_MODEL_PARALLEL_SIZE,
#         context_parallel_size=CONTEXT_PARALLEL_SIZE,
#         pipeline_dtype=torch.bfloat16,
#         ckpt_load_optimizer=False,  # Needs to be false for a normal model checkpoint.
#         ckpt_save_optimizer=False,
#         ckpt_async_save=False,
#         save_ckpt_format="zarr",
#     )
#     trainer = nl.Trainer(
#         accelerator="gpu",
#         num_nodes=NUM_NODES,
#         devices=NUM_GPUS,
#         strategy=strategy,
#         log_every_n_steps=1,
#         limit_val_batches=10,
#         num_sanity_val_steps=0,
#         plugins=nl.MegatronMixedPrecision(
#             precision="bf16-mixed",
#             params_dtype=torch.bfloat16,
#         ),
#     )
#     # Last char from gold std removed.
#     input_seq = "GAAATTAGCGCGTCCGGAATGATACGAGGGGAAACGAAATTTTGAATTAATGGAGAAAAAAGACGAGAAACCTTAAGCAAAAAAATTTTAGCTTCGAATATTTATTAATTTCTGAGATGTTGTTAAACGATTTTCGATTCCAAGTTGTGCGCACGAACGTTATTGCAAATAAATGCTGCTTATTCGGATGTTTCCACGATCTTTGTTGCAATGGTAGTCGAGTACCCGATAACCCAATTTCGTTACATCGGCCTATCTGTAGAATATCCAATCTATGGTTCATAAAAAATCTGATCGTTTGTTTTTAAGAAATTAAACGCGTTAAATTGAACGAATTTCGAATACCGGTCTTAGCGAAGGACCTCCCCTCTTGCTTGCGTATTGCCCCGCGAAATTTCTTTTCGGCGATGAACGATACAAAAAATTCTATCGAATGTTACTTCTATTCTCTGCCTCGTCTATGACTTGGAGATTGGTCTATGTCGTTCGTTTTCTCGCGAGTTTCCAATATGTCCGTAGTATGTGAACGCTGGTATTCGTGAAGATAAATTATTGTTTTTACAATTTCTTTCAAAAATATATAATTTTAATTTATATAA"
#     deleted_char = "T"
#     temperature = 1.0
#     top_k = 0
#     top_p = 0.0
#     max_new_tokens = 1
#     checkpoint_path = load("evo2/7b-8k-zarr:1.1", source="pbss")
#     gold_standard_no_fp8 = load("evo2/7b-8k-nofp8-te-goldvalue-testdata:1.0")
#     gold_standard_no_fp8_tensor = torch.load(gold_standard_no_fp8)
#     gold_standard_no_fp8_tensor = gold_standard_no_fp8_tensor[0, -1]
#     results = generate(
#         path=checkpoint_path,
#         prompts=[input_seq],
#         trainer=trainer,
#         inference_params=CommonInferenceParams(
#             temperature,
#             top_k,
#             top_p,
#             return_log_probs=False,
#             num_tokens_to_generate=max_new_tokens,
#         ),
#         random_seed=RANDOM_SEED,
#         text_only=False,
#     )

#     # Text equal to "T" (deleted char)
#     assert results[0].generated_text == deleted_char
#     assert isinstance(results, list)

# TODO: Later...
# Do comparison to test golden values for the logit vector.
# gold_standard_logits_vector = gold_standard_no_fp8_tensor

# Do cosine similarity between the two vectors, for the topk=4 indices.
# Make sure topk=4 = ACTG
# Use indices to go from 512 -> 4.
# Do cosine similarity between the two vectors.
