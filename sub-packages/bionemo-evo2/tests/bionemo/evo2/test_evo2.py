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


def prune_caches():
    import gc
    gc.collect()
    torch.cuda.empty_cache()


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


@pytest.fixture
def sequences():
    with (Path(__file__).parent / "data" / "prompts.csv").open(newline="") as f:
        from csv import DictReader

        reader = DictReader(f)
        return [row["Sequence"] for row in reader]


def get_trainer(pipeline_parallel=1):
    import nemo.lightning as nl

    fp8 = True
    full_fp8 = False
    return nl.Trainer(
        accelerator="gpu",
        devices=pipeline_parallel,
        strategy=nl.MegatronStrategy(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=pipeline_parallel,
            context_parallel_size=1,
            pipeline_dtype=torch.bfloat16,
            ckpt_load_optimizer=False,
            ckpt_save_optimizer=False,
            ckpt_async_save=False,
            save_ckpt_format="torch_dist",
            ckpt_load_strictness="log_all",
        ),
        log_every_n_steps=1,
        limit_val_batches=10,
        num_sanity_val_steps=0,
        plugins=nl.MegatronMixedPrecision(
            precision="bf16-mixed",
            params_dtype=torch.bfloat16,
            # Only use FP8 in this plugin when using full FP8 precision and FP8.
            #   Otherwise use vortex_style_fp8 in the model config.
            fp8="hybrid" if fp8 and full_fp8 else None,
            fp8_amax_history_len=16 if fp8 and full_fp8 else 1,
            fp8_amax_compute_algo="max" if fp8 and full_fp8 else "most_recent",
        ),
    )

def detect_pst(ckpt_name):
    mem_gb = torch.cuda.get_device_properties(0).total_memory // 1024 // 1024 // 1024
    gpus = torch.cuda.device_count()
    ret = None
    if "40b" in ckpt_name:
        if gpus >= 2 and mem_gb > 120: # e.g. h200-x2
            ret = 8192
        elif mem_gb > 120: # e.g. h200-x1
            ret = 4096
        elif gpus >= 3 and mem_gb > 60: # e.g. h100-x3
            ret = 512
        elif gpus >= 2 and mem_gb > 60: # e.g. h100-x2
            ret = 256
        elif gpus >= 3: # e.g. l40-x3
            ret = 128
        else: # e.g. l40-x2
            ret = 64
    elif "7b" in ckpt_name:
        if gpus < 2 and mem_gb < 60:
            ret = 4096
    logger.info(f"Will use prompt_segmentation_threshold={ret}, {gpus=} {mem_gb=}")
    return ret

def get_inference_params(ckpt_name):
    # Below are parameters that need to be tuned based on GPU memory size.
    return {
        "fp32_residual_connection": False,

        # This mostly determines size of KV cache.
        "inference_max_seq_length": 8192,

        # Affects KV cache size, can cause OOM easily, set to 1 to save memory (i.e. running 40b model)
        "inference_max_requests": 1,

        # This is used to split batch into mini-batches.
        # If use batch size = 1, set same as inference_max_seq_length to avoid
        # more complex code path.
        "inference_batch_times_seqlen_threshold": 8192,

        # Affects max memory usage during parallel hyena filters pass.
        "prompt_segmentation_threshold": detect_pst(ckpt_name),
    }

def get_model_and_tokenizer(ckpt_name, vortex_style_fp8=False):
    prune_caches()
    trainer = get_trainer()
    from bionemo.core.data.load import load

    try:
        ckpt_dir: Path = load(ckpt_name)
    except ValueError as e:
        if "40b" in ckpt_name: # NeMo 40b is not yet published
            raise ValueError(f"Place 40b checkpoint to ~/.cache/bionemo/overrides/{ckpt_name}") from e
        raise

    from nemo.collections.llm import inference

    inference_wrapped_model, mcore_tokenizer = inference.setup_model_and_tokenizer(
        path=ckpt_dir,
        trainer=trainer,
        params_dtype=torch.bfloat16,
        vortex_style_fp8=vortex_style_fp8,
        # use_te_rng_tracker=True,
        # te_rng_tracker=True,
        # inference_rng_tracker=True,
        # enable_cuda_graph=True,
        # cudagraph_rng_tracker=True,
        # flash_decode=True,
        recompute_granularity=None,
        recompute_num_layers=None,
        recompute_method=None,
        **get_inference_params(ckpt_name),
    )
    return inference_wrapped_model, mcore_tokenizer


def calc_matchrate(*, tokenizer, in_seq, logits):
    softmax_logprobs = torch.log_softmax(logits, dim=-1)
    softmax_logprobs = softmax_logprobs[:, :-1]
    o = softmax_logprobs.argmax(dim=-1)[0]
    if hasattr(tokenizer, "tokenize"):
        i = torch.tensor(tokenizer.tokenize(in_seq[1:]), device=o.device)
    else:
        i = torch.tensor(tokenizer.text_to_ids(in_seq[1:]), device=o.device)
    return (i == o).sum().item() / (i.size()[0] - 1)


def check_matchrate(*, ckpt_name, matchrate, assert_matchrate=True):
    logger.info(f"{ckpt_name} {matchrate = }")
    if "1b-" in ckpt_name:
        if assert_matchrate:
            assert matchrate > 0.70, (ckpt_name, matchrate)
        else:
            print(f"{ckpt_name} {matchrate = }")
    elif "7b-" in ckpt_name:
        if assert_matchrate:
            assert matchrate > 0.79, (ckpt_name, matchrate)
        else:
            print(f"{ckpt_name} {matchrate = }")
    else:
        raise NotImplementedError


@pytest.mark.parametrize(
    "ckpt_name,expected_matchpercents",
    [
        ("evo2/1b-8k-bf16:1.0", [96.27, 67.93, 77.50, 80.30]),
        ("evo2/1b-8k:1.0", [96.27, 67.93, 77.50, 80.30]),
        # ("evo2/7b-8k:1.0", [97.60, 89.63, 80.03, 84.57]),
        # ("evo2/7b-1m:1.0", [97.60, 89.63, 80.03, 84.57]),
    ],
)
def test_forward(sequences: list[str], ckpt_name: str, expected_matchpercents: list[float]):
    assert len(sequences) > 0
    is_fp8_supported, compute_capability, device_info = check_fp8_support(torch.cuda.current_device())
    skip = "evo2/1b-8k:" in ckpt_name and not is_fp8_supported
    if skip:
        # This checkpoint is sensitive to FP8, so we skip it if it is not supported on the current device.
        pytest.skip(f"Skipping {ckpt_name} because it is not supported on {device_info} ({compute_capability})")
    vortex_style_fp8 = is_fp8_supported and "bf16" not in ckpt_name

    matchrates = []
    for seq in sequences:
        inference_wrapped_model, mcore_tokenizer = get_model_and_tokenizer(
            ckpt_name, vortex_style_fp8=vortex_style_fp8
        )

        seq = seq[:6000]  # TODO: artificial limit, megatron uses more memory. Vortex can process full sequences
        with torch.no_grad():
            device = torch.cuda.current_device()
            tokens = torch.tensor([mcore_tokenizer.tokenize(seq)], device=device)
            forward_args = {
                "tokens": tokens,
                "position_ids": None,
                "attention_mask": None,
            }

            inference_wrapped_model.prep_model_for_inference(prompts_tokens=None)
            logits = inference_wrapped_model.run_one_forward_step(forward_args)

            from megatron.core.inference.communication_utils import broadcast_from_last_pipeline_stage

            batch_size, context_length, vocab_size = 1, len(seq), 512
            logits = broadcast_from_last_pipeline_stage(
                [batch_size, context_length, vocab_size],
                dtype=inference_wrapped_model.inference_wrapper_config.params_dtype,
                tensor=logits,
            )

            matchrate = calc_matchrate(tokenizer=mcore_tokenizer, in_seq=seq, logits=logits)
            matchrates.append(matchrate)
            check_matchrate(ckpt_name=ckpt_name, matchrate=matchrate, assert_matchrate=False)
    assert len(matchrates) == len(expected_matchpercents)
    matchperc_print = [f"{m * 100.0:.1f}%" for m in matchrates]
    matchperc_print_expected = [f"{ep:.1f}%" for ep in expected_matchpercents]
    assert all(m * 100.0 >= 0.95 * ep for m, ep in zip(matchrates, expected_matchpercents)), (
        f"Expected at least 95% of {matchperc_print_expected=}, got {matchperc_print=}"
    )


@pytest.mark.parametrize(
    "ckpt_name,expected_matchpercents",
    [
        ("evo2/1b-8k-bf16:1.0", [96.27, 67.93, 77.50, 80.30]),
        ("evo2/1b-8k:1.0", [96.27, 67.93, 77.50, 80.30]),
        # ("evo2/7b-8k:1.0", [97.60, 89.63, 80.03, 84.57]),
        # ("evo2/7b-1m:1.0", [97.60, 89.63, 80.03, 84.57]),
    ],
)
def test_forward_manual(sequences: list[str], ckpt_name: str, expected_matchpercents: list[float]):
    assert len(sequences) > 0
    is_fp8_supported, compute_capability, device_info = check_fp8_support(torch.cuda.current_device())
    skip = "evo2/1b-8k:" in ckpt_name and not is_fp8_supported
    vortex_style_fp8 = is_fp8_supported and "bf16" not in ckpt_name
    if skip:
        # This checkpoint is sensitive to FP8, so we skip it if it is not supported on the current device.
        pytest.skip(f"Skipping {ckpt_name} because it is not supported on {device_info} ({compute_capability})")
    with distributed_model_parallel_state(), torch.no_grad():
        tokenizer = get_nmt_tokenizer(
            "byte-level",
        )
        if "1b-8k" in ckpt_name:
            model_config = llm.Hyena1bConfig(
                use_te=True,
                seq_length=8192,
                flash_decode=False,
                fp32_residual_connection=False,
                vortex_style_fp8=vortex_style_fp8,
            )
        elif "7b-8k" in ckpt_name:
            model_config = llm.Hyena7bConfig(
                use_te=True,
                seq_length=8192,
                flash_decode=False,
                fp32_residual_connection=False,
                vortex_style_fp8=vortex_style_fp8,
            )
        elif "7b-1m" in ckpt_name:
            model_config = llm.Hyena7bARCLongContextConfig(
                use_te=True,
                seq_length=8192,
                flash_decode=False,
                fp32_residual_connection=False,
                vortex_style_fp8=vortex_style_fp8,
            )
        else:
            raise NotImplementedError
        ckpt_weights: Path = load(ckpt_name) / "weights"
        raw_megatron_model = model_config.configure_model(tokenizer).eval().cuda()
        device = raw_megatron_model.parameters().__next__().device
        load_weights_sharded_inplace_nemo2_to_mcore(raw_megatron_model, ckpt_weights, {}, "torch_dist")
        model = Float16Module(model_config, raw_megatron_model)
        matchrates = []
        for seq in sequences:
            seq = seq[:6000]  # TODO: artificial limit, megatron uses more memory. Vortex can process full sequences
            with torch.no_grad():
                device = torch.cuda.current_device()
                # tokens = torch.tensor([tokenizer.tokenize(seq)], device=device)
                input_ids = torch.tensor(tokenizer.text_to_ids(seq)).int().unsqueeze(0).to(device)
                attention_mask = None
                # when labels is None, the model returns logits
                logits = model(input_ids=input_ids, position_ids=None, attention_mask=attention_mask, labels=None)

                matchrate = calc_matchrate(tokenizer=tokenizer, in_seq=seq, logits=logits)
                matchrates.append(matchrate)
                check_matchrate(ckpt_name=ckpt_name, matchrate=matchrate, assert_matchrate=False)
        assert len(matchrates) == len(expected_matchpercents)
        matchperc_print = [f"{m * 100.0:.1f}%" for m in matchrates]
        matchperc_print_expected = [f"{ep:.1f}%" for ep in expected_matchpercents]
        assert all(m * 100.0 >= 0.95 * ep for m, ep in zip(matchrates, expected_matchpercents)), (
            f"Expected at least 95% of {matchperc_print_expected=}, got {matchperc_print=}"
        )


def mid_point_split(*, seq, num_tokens):
    mid_point = 2 * (len(seq) // 4)
    prompt = seq[:mid_point]
    target = seq[mid_point : mid_point + num_tokens]  # Only compare to the section of sequence directly
    return prompt, target


def calculate_sequence_identity(seq1: str, seq2: str) -> float | None:
    """Calculate sequence identity between two sequences through direct comparison."""
    if not seq1 or not seq2:
        return None

    # Direct comparison of sequences
    min_length = min(len(seq1), len(seq2))
    matches = sum(a == b for a, b in zip(seq1[:min_length], seq2[:min_length]))

    return (matches / min_length) * 100


@pytest.mark.parametrize(
    "ckpt_name,expected_matchpercents",
    [
        ("evo2/1b-8k-bf16:1.0", [96.8, 29.7, 76.6, 71.6]),
        ("evo2/1b-8k:1.0", [96.8, 29.7, 76.6, 71.6]),
        # ("evo2/40b-1m:1.0", [0, 0, 0, 0]),
        # ("evo2/7b-8k:1.0", [97.60, 89.63, 80.03, 84.57]),
        # ("evo2/7b-1m:1.0", [97.60, 89.63, 80.03, 84.57]),
    ],
)
def test_batch_generate(sequences: list[str], ckpt_name: str, expected_matchpercents: list[float]):
    assert len(sequences) > 0
    is_fp8_supported, compute_capability, device_info = check_fp8_support(torch.cuda.current_device())
    skip = "evo2/1b-8k:" in ckpt_name and not is_fp8_supported
    if skip:
        # This checkpoint is sensitive to FP8, so we skip it if it is not supported on the current device.
        pytest.skip(f"Skipping {ckpt_name} because it is not supported on {device_info} ({compute_capability})")
    # only use vortex_style_fp8 for non-bf16 checkpoints with fp8 support
    vortex_style_fp8 = is_fp8_supported and "bf16" not in ckpt_name
    inference_wrapped_model, mcore_tokenizer = get_model_and_tokenizer(ckpt_name, vortex_style_fp8=vortex_style_fp8)

    match_percents = []
    num_tokens = 500
    seq_prompts = [mid_point_split(seq=seq, num_tokens=num_tokens) for seq in sequences]
    from megatron.core.inference.common_inference_params import CommonInferenceParams
    from nemo.collections.llm.inference import generate

    results = generate(
        model=inference_wrapped_model,
        max_batch_size=1,  # vortex only supports batch size 1
        tokenizer=mcore_tokenizer,
        prompts=[sq[0] for sq in seq_prompts],
        random_seed=42,
        inference_params=CommonInferenceParams(
            temperature=1.0,
            top_k=1,
            top_p=0.0,
            return_log_probs=False,
            num_tokens_to_generate=num_tokens,
        ),
    )

    for i, (result, (prompt, target)) in enumerate(zip(results, seq_prompts)):
        gen_seq = result.generated_text
        logging.info(f"{ckpt_name} {torch.distributed.get_rank()=} {gen_seq=}")
        logging.info(f"{ckpt_name} {torch.distributed.get_rank()=} {target=}")
        match_percent = calculate_sequence_identity(target, gen_seq)
        logging.info(
            f"{ckpt_name} {torch.distributed.get_rank()=} {match_percent=} expected: {expected_matchpercents[i]}"
        )
        match_percents.append(match_percent)

    assert len(match_percents) == len(expected_matchpercents)
    matchperc_print = [f"{mp:.1f}%" for mp in match_percents]
    matchperc_print_expected = [f"{ep:.1f}%" for ep in expected_matchpercents]
    assert all(mp >= 0.90 * ep for mp, ep in zip(match_percents, expected_matchpercents)), (
        f"Expected at least 90% of {matchperc_print_expected=}, got {matchperc_print=}"
    )
