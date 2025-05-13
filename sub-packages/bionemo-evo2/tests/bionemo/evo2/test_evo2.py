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

import logging
from pathlib import Path
from typing import Literal, Set

import numpy as np
import pytest
import torch
from megatron.core.transformer.module import Float16Module
from nemo.collections import llm
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.lightning.io.pl import MegatronCheckpointIO

from bionemo.core.data.load import load
from bionemo.llm.utils.weight_utils import (
    MegatronModelType,
    _key_in_filter,
    _munge_key_megatron_to_nemo2,
    _munge_sharded_tensor_key_megatron_to_nemo2,
)
from bionemo.testing.megatron_parallel_state_utils import distributed_model_parallel_state
from bionemo.testing.torch import check_fp8_support


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Capture all levels in the logger itself


def load_weights_sharded_inplace_nemo2_to_mcore(
    model: MegatronModelType,
    distributed_checkpoint_dir: str | Path,
    skip_keys_with_these_prefixes: Set[str],
    ckpt_format: Literal["zarr", "torch_dist"] = "torch_dist",
):
    logger.info("Start setting up state dict")
    sharded_state_dict = {
        _munge_key_megatron_to_nemo2(k): _munge_sharded_tensor_key_megatron_to_nemo2(v)
        for k, v in model.sharded_state_dict().items()
        if not _key_in_filter(
            k, skip_keys_with_these_prefixes
        )  # and "_extra_state" not in k  # extra state is needed for fp8 sharded states
    }
    # Load the checkpoint with strict=false to allow for missing keys (backward compatibility)
    # Error: megatron.core.dist_checkpointing.core.CheckpointingException:
    # Object shard ... module.decoder.final_norm._extra_state/shard_0_1.pt not found
    MegatronCheckpointIO(save_ckpt_format=ckpt_format).load_checkpoint(
        distributed_checkpoint_dir, sharded_state_dict=sharded_state_dict, strict=False
    )


@pytest.mark.parametrize("seq_len", [8_192, 16_384])
def test_golden_values_top_k_logits_and_cosine_similarity(seq_len: int):
    try:
        evo2_1b_checkpoint_weights: Path = load("evo2/1b-8k:1.0") / "weights"
        gold_standard_no_fp8 = load("evo2/1b-8k-nofp8-te-goldvalue-testdata-A6000:1.0")
    except ValueError as e:
        if e.args[0].endswith("does not have an NGC URL."):
            raise ValueError(
                "Please re-run test with `BIONEMO_DATA_SOURCE=pbss py.test ...`, "
                "one or more files are missing from ngc."
            )
        else:
            raise e
    with distributed_model_parallel_state(), torch.no_grad():
        hyena_config = llm.Hyena1bConfig(use_te=True, seq_length=seq_len)
        tokenizer = get_nmt_tokenizer(
            "byte-level",
        )
        raw_megatron_model = hyena_config.configure_model(tokenizer).eval().cuda()
        device = raw_megatron_model.parameters().__next__().device
        load_weights_sharded_inplace_nemo2_to_mcore(raw_megatron_model, evo2_1b_checkpoint_weights, {}, "torch_dist")
        model = Float16Module(hyena_config, raw_megatron_model)
        input_seq = "GAAATTAGCGCGTCCGGAATGATACGAGGGGAAACGAAATTTTGAATTAATGGAGAAAAAAGACGAGAAACCTTAAGCAAAAAAATTTTAGCTTCGAATATTTATTAATTTCTGAGATGTTGTTAAACGATTTTCGATTCCAAGTTGTGCGCACGAACGTTATTGCAAATAAATGCTGCTTATTCGGATGTTTCCACGATCTTTGTTGCAATGGTAGTCGAGTACCCGATAACCCAATTTCGTTACATCGGCCTATCTGTAGAATATCCAATCTATGGTTCATAAAAAATCTGATCGTTTGTTTTTAAGAAATTAAACGCGTTAAATTGAACGAATTTCGAATACCGGTCTTAGCGAAGGACCTCCCCTCTTGCTTGCGTATTGCCCCGCGAAATTTCTTTTCGGCGATGAACGATACAAAAAATTCTATCGAATGTTACTTCTATTCTCTGCCTCGTCTATGACTTGGAGATTGGTCTATGTCGTTCGTTTTCTCGCGAGTTTCCAATATGTCCGTAGTATGTGAACGCTGGTATTCGTGAAGATAAATTATTGTTTTTACAATTTCTTTCAAAAATATATAATTTTAATTTATATAAT"
        input_ids = torch.tensor(tokenizer.text_to_ids(input_seq)).int().unsqueeze(0).to(device)
        position_ids = torch.arange(len(input_seq)).unsqueeze(0).to(device)
        attention_mask = None
        outputs = model(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask)
        gold_standard_no_fp8_tensor = torch.load(gold_standard_no_fp8).to(device=outputs.device, dtype=outputs.dtype)
        top_2_logits_golden = gold_standard_no_fp8_tensor.topk(dim=-1, sorted=True, largest=True, k=2)
        ambiguous_positions = (
            top_2_logits_golden.values[..., 0] - top_2_logits_golden.values[..., 1]
        ).abs() < 9.9e-3  # hand tunes for observed diffs from A100 and H100
        n_ambiguous = ambiguous_positions.sum()

        assert n_ambiguous <= 19

        our_char_indices = outputs.softmax(dim=-1).argmax(dim=-1).flatten().detach().cpu().numpy()
        not_amb_positions = ~ambiguous_positions.flatten().cpu().numpy()
        # Generate our string, removing the ambiguous positions.
        our_generation_str = "".join([chr(idx) for idx in our_char_indices[not_amb_positions].tolist()])
        # Do the same to the golden values
        gold_std_char_indices = (
            gold_standard_no_fp8_tensor.softmax(dim=-1).argmax(dim=-1).flatten().detach().cpu().numpy()
        )
        # Make the string
        gold_std_str = "".join([chr(idx) for idx in gold_std_char_indices[not_amb_positions].tolist()])
        array_eq = np.array(list(our_generation_str)) == np.array(list(gold_std_str))
        # Ensure the two strings are approximately equal.
        if array_eq.mean() < 0.95:
            array_eq = np.array(list(our_generation_str)) == np.array(list(gold_std_str))
            mismatch_positions = np.arange(outputs.shape[1])[not_amb_positions][~array_eq]
            err_str = f"Fraction of expected mismatch positions exceeds 5%: {(~array_eq).mean()}"
            err_str += f"Mismatch positions: {mismatch_positions}"
            err_str += f"Fraction of unexpected mismatch positions: {(~array_eq).mean()}"
            top_two_logits_at_mismatch = top_2_logits_golden.values[0, mismatch_positions]
            top_2_logits_pred = outputs.topk(dim=-1, sorted=True, largest=True, k=2)
            top_two_pred_logits_at_mismatch = top_2_logits_pred.values[0, mismatch_positions]
            err_str += f"Top two logits at mismatch positions: {top_two_logits_at_mismatch}"
            err_str += f"Top two pred logits at mismatch positions: {top_two_pred_logits_at_mismatch}"
            raise AssertionError(err_str)

        # Verify that the top-4 from the logit vectors are the same.
        # A: 65
        # C: 67
        # G: 71
        # T: 84
        # Find the corresponding ATGC and compare the two vectors with those four values.
        # Ensures that the top 4 ascii characters of the output are ACGT.
        top_4_inds = outputs.topk(dim=-1, sorted=False, largest=True, k=4)
        assert set(top_4_inds.indices.flatten().cpu().numpy().tolist()).issubset((65, 67, 71, 84))
        output_vector = outputs[0, -1, top_4_inds.indices]

        # Then its the top 4 indices of the gold standard tensor
        top_4_inds_golden = gold_standard_no_fp8_tensor.topk(dim=-1, sorted=False, largest=True, k=4)
        assert set(top_4_inds_golden.indices.flatten().cpu().numpy().tolist()).issubset((65, 67, 71, 84))
        gold_standard_no_fp8_vector = gold_standard_no_fp8_tensor[0, -1, top_4_inds_golden.indices]

        # Run cosine similarity between the two vectors.
        logit_similarity = torch.nn.functional.cosine_similarity(output_vector, gold_standard_no_fp8_vector, dim=-1)
        assert torch.mean(torch.abs(logit_similarity - torch.ones_like(logit_similarity))) < 0.03


@pytest.mark.slow
def test_golden_values_top_k_logits_and_cosine_similarity_7b(seq_len: int = 8_192):
    try:
        evo2_7b_checkpoint_weights: Path = load("evo2/7b-8k:1.0") / "weights"
        gold_standard_no_fp8 = load("evo2/7b-8k-nofp8-te-goldvalue-testdata:1.0")
    except ValueError as e:
        if e.args[0].endswith("does not have an NGC URL."):
            raise ValueError(
                "Please re-run test with `BIONEMO_DATA_SOURCE=pbss py.test ...`, "
                "one or more files are missing from ngc."
            )
        else:
            raise e
    with distributed_model_parallel_state(), torch.no_grad():
        hyena_config = llm.Hyena7bConfig(use_te=True, seq_length=seq_len)
        tokenizer = get_nmt_tokenizer(
            "byte-level",
        )
        raw_megatron_model = hyena_config.configure_model(tokenizer).eval().cuda()
        device = raw_megatron_model.parameters().__next__().device
        load_weights_sharded_inplace_nemo2_to_mcore(raw_megatron_model, evo2_7b_checkpoint_weights, {}, "torch_dist")
        model = Float16Module(hyena_config, raw_megatron_model)
        input_seq = "GAAATTAGCGCGTCCGGAATGATACGAGGGGAAACGAAATTTTGAATTAATGGAGAAAAAAGACGAGAAACCTTAAGCAAAAAAATTTTAGCTTCGAATATTTATTAATTTCTGAGATGTTGTTAAACGATTTTCGATTCCAAGTTGTGCGCACGAACGTTATTGCAAATAAATGCTGCTTATTCGGATGTTTCCACGATCTTTGTTGCAATGGTAGTCGAGTACCCGATAACCCAATTTCGTTACATCGGCCTATCTGTAGAATATCCAATCTATGGTTCATAAAAAATCTGATCGTTTGTTTTTAAGAAATTAAACGCGTTAAATTGAACGAATTTCGAATACCGGTCTTAGCGAAGGACCTCCCCTCTTGCTTGCGTATTGCCCCGCGAAATTTCTTTTCGGCGATGAACGATACAAAAAATTCTATCGAATGTTACTTCTATTCTCTGCCTCGTCTATGACTTGGAGATTGGTCTATGTCGTTCGTTTTCTCGCGAGTTTCCAATATGTCCGTAGTATGTGAACGCTGGTATTCGTGAAGATAAATTATTGTTTTTACAATTTCTTTCAAAAATATATAATTTTAATTTATATAAT"
        input_ids = torch.tensor(tokenizer.text_to_ids(input_seq)).int().unsqueeze(0).to(device)
        position_ids = torch.arange(len(input_seq)).unsqueeze(0).to(device)
        attention_mask = None
        outputs = model(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask)
        gold_standard_no_fp8_tensor = torch.load(gold_standard_no_fp8).to(device=outputs.device, dtype=outputs.dtype)
        is_fp8_supported, compute_capability, device_info = check_fp8_support(device.index)
        if is_fp8_supported and compute_capability == "9.0":
            # Most rigurous assertion for output equivalence currently works on devices that are new enough to
            #  support FP8.
            logger.info(
                f"Device {device_info} ({compute_capability}) supports FP8 with 9.0 compute capability, the "
                "same configuration as the gold standard was generated with. Running most rigurous assertion."
            )
            torch.testing.assert_close(outputs, gold_standard_no_fp8_tensor)
        else:
            logger.info(
                f"Device {device_info} ({compute_capability}) does not support FP8. Running less rigurous assertions."
            )
        top_2_logits_golden = gold_standard_no_fp8_tensor.topk(dim=-1, sorted=True, largest=True, k=2)
        ambiguous_positions = (
            top_2_logits_golden.values[..., 0] - top_2_logits_golden.values[..., 1]
        ).abs() < 9.9e-3  # hand tunes for observed diffs from A100 and H100 with 7b model
        n_ambiguous = ambiguous_positions.sum()

        assert n_ambiguous <= 19

        our_char_indices = outputs.softmax(dim=-1).argmax(dim=-1).flatten().detach().cpu().numpy()
        not_amb_positions = ~ambiguous_positions.flatten().cpu().numpy()
        # Generate our string, removing the ambiguous positions.
        our_generation_str = "".join([chr(idx) for idx in our_char_indices[not_amb_positions].tolist()])
        # Do the same to the golden values
        gold_std_char_indices = (
            gold_standard_no_fp8_tensor.softmax(dim=-1).argmax(dim=-1).flatten().detach().cpu().numpy()
        )
        # Make the string
        gold_std_str = "".join([chr(idx) for idx in gold_std_char_indices[not_amb_positions].tolist()])

        # Ensure the two strings are equal.
        assert all(np.array(list(our_generation_str)) == np.array(list(gold_std_str)))

        # Verify that the top-4 from the logit vectors are the same.
        # A: 65
        # C: 67
        # G: 71
        # T: 84
        # Find the corresponding ATGC and compare the two vectors with those four values.
        # Ensures that the top 4 ascii characters of the output are ACGT.
        top_4_inds = outputs.topk(dim=-1, sorted=False, largest=True, k=4)
        assert set(top_4_inds.indices.flatten().cpu().numpy().tolist()).issubset((65, 67, 71, 84))
        output_vector = outputs[0, -1, top_4_inds.indices]

        # Then its the top 4 indices of the gold standard tensor
        top_4_inds_golden = gold_standard_no_fp8_tensor.topk(dim=-1, sorted=False, largest=True, k=4)
        assert set(top_4_inds_golden.indices.flatten().cpu().numpy().tolist()).issubset((65, 67, 71, 84))
        gold_standard_no_fp8_vector = gold_standard_no_fp8_tensor[0, -1, top_4_inds_golden.indices]

        # Run cosine similarity between the two vectors.
        logit_similarity = torch.nn.functional.cosine_similarity(output_vector, gold_standard_no_fp8_vector, dim=-1)
        assert torch.mean(torch.abs(logit_similarity - torch.ones_like(logit_similarity))) < 9.9e-3
