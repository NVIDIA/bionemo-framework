# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import gc
import os
from typing import Iterator
from unittest.mock import MagicMock

import hydra
import pytest
import torch

from tests.esm2_hf_to_nemo_conversion.compare import compare_outputs
from tests.esm2_hf_to_nemo_conversion.convert_hf_esm2_to_nemo import (
    convert_esm_hf_model_to_nemo,
    load_sample_protein_sequence_data,
    run_impulse_sanity_check,
    run_parameter_gradient_sanity_check,
)


@pytest.fixture
def esm2_650m_hf_model_setup() -> Iterator[tuple[str, str, str]]:
    bionemo_home = os.getenv("BIONEMO_HOME")
    hf_model_dir = os.path.join(bionemo_home, "models/protein/esm2nv")
    hf_model_size = "650M"
    hf_model_name = "esm2-650M-huggingface"
    yield hf_model_name, hf_model_size, hf_model_dir
    torch.cuda.empty_cache()
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    gc.collect()


@pytest.fixture
def esm2_3b_hf_model_setup() -> Iterator[tuple[str, str, str]]:
    bionemo_home = os.getenv("BIONEMO_HOME")
    hf_model_dir = os.path.join(bionemo_home, "models/protein/esm2nv")
    hf_model_size = "3B"
    hf_model_name = "esm2-3B-huggingface"
    yield hf_model_name, hf_model_size, hf_model_dir
    torch.cuda.empty_cache()
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    gc.collect()


def get_hf_to_nemo_conversion_args(hf_model_checkpoint, hf_model_dir, hf_model_size, precision=32, use_nemo_apex=True):
    args = MagicMock()
    args.input_hf_model_file = os.path.join(hf_model_dir, hf_model_checkpoint)
    converted_nemo_model_name = f"converted-esm2-{hf_model_size}.nemo"
    args.output_nemo_model_file = os.path.join(hf_model_dir, converted_nemo_model_name)
    args.precision = precision
    args.hf_precision = precision
    args.use_nemo_apex = use_nemo_apex
    args.model_size = hf_model_size
    args.run_impulse_sanity_check = False
    args.model_class = "esm2"
    return args


@pytest.mark.internal
@pytest.mark.needs_gpu
def test_esm2_650m_model_hf_to_nemo_conversion_impulse_response_fp16_on_gpu(esm2_650m_hf_model_setup, atol=3e-1):
    hf_model_name, hf_model_size, hf_model_dir = esm2_650m_hf_model_setup
    args = get_hf_to_nemo_conversion_args(hf_model_name, hf_model_dir, hf_model_size, precision=16)
    nemo_model, hf_model = convert_esm_hf_model_to_nemo(args)
    # assert the NeMo model was saved as `output_nemo_model_file`
    assert os.path.exists(args.output_nemo_model_file)

    # check outputs from HF ESM2 model and ESM2nv converted model are matching with atol=ATOL
    # Note: exact matching, i.e. mean_abs == 0 is possible on GPU devices such as V100 and A6000,
    # but this result is inconsistent and may fail occasionally when using A100 GPU devices.
    # less stringent atol since on FP16 and on GPU
    sample_data = load_sample_protein_sequence_data()
    max_absolute_diff, mean_relative_diff = run_impulse_sanity_check(
        nemo_model, hf_model, sample_data, compare_outputs, model_hf_dtype=args.hf_precision, device="cuda"
    )
    assert max_absolute_diff <= atol


@pytest.mark.internal
@pytest.mark.skip(reason="Limited capacity on CI runners, should work locally")
def test_esm2_3b_model_hf_to_nemo_conversion_impulse_response_fp16_on_gpu(esm2_3b_hf_model_setup, atol=3e-1):
    hf_model_name, hf_model_size, hf_model_dir = esm2_3b_hf_model_setup
    args = get_hf_to_nemo_conversion_args(hf_model_name, hf_model_dir, hf_model_size, precision=16)
    nemo_model, hf_model = convert_esm_hf_model_to_nemo(args)
    # assert the NeMo model was saved in `output_nemo_model_file`
    assert os.path.exists(args.output_nemo_model_file)

    # check outputs from HF ESM2 model and ESM2nv converted model are matching with atol=ATOL
    # Note: exact matching, i.e. mean_abs == 0 is possible on GPU devices such as V100 and A6000,
    # but this result is inconsistent and may fail occasionally when using A100 GPU devices.
    # less stringent atol since on FP16 and on GPU
    sample_data = load_sample_protein_sequence_data()
    max_absolute_diff, mean_relative_diff = run_impulse_sanity_check(
        nemo_model, hf_model, sample_data, compare_outputs, model_hf_dtype=args.hf_precision, device="cuda"
    )
    assert max_absolute_diff <= atol


@pytest.mark.internal
def test_esm2_650m_model_hf_to_nemo_conversion_impulse_response_wo_apex_on_cpu(esm2_650m_hf_model_setup, atol=3e-4):
    hf_model_name, hf_model_size, hf_model_dir = esm2_650m_hf_model_setup
    args = get_hf_to_nemo_conversion_args(hf_model_name, hf_model_dir, hf_model_size, use_nemo_apex=False)
    nemo_model, hf_model = convert_esm_hf_model_to_nemo(args)
    # assert the NeMo model was saved in `output_nemo_model_file`
    assert os.path.exists(args.output_nemo_model_file)

    # check outputs from HF ESM2 model and ESM2nv converted model are matching with atol=ATOL
    # more stringent atol since on CPU we can get better match due to ECC RAM etc
    sample_data = load_sample_protein_sequence_data()
    max_absolute_diff, mean_relative_diff = run_impulse_sanity_check(
        nemo_model, hf_model, sample_data, compare_outputs, model_hf_dtype=args.hf_precision, device="cpu"
    )
    assert max_absolute_diff <= atol


@pytest.mark.internal
@pytest.mark.skip(reason="Limited capacity on CI runners, should work locally")
def test_esm2_3b_model_hf_to_nemo_conversion_impulse_response_wo_apex_on_cpu(esm2_3b_hf_model_setup, atol=3e-4):
    hf_model_name, hf_model_size, hf_model_dir = esm2_3b_hf_model_setup
    args = get_hf_to_nemo_conversion_args(hf_model_name, hf_model_dir, hf_model_size, use_nemo_apex=False)
    nemo_model, hf_model = convert_esm_hf_model_to_nemo(args)
    # assert the NeMo model was saved in `output_nemo_model_file`
    assert os.path.exists(args.output_nemo_model_file)

    # check outputs from HF ESM2 model and ESM2nv converted model are matching with atol=ATOL
    # more stringent atol since on CPU we can get better match due to ECC RAM etc
    sample_data = load_sample_protein_sequence_data()
    max_absolute_diff, mean_relative_diff = run_impulse_sanity_check(
        nemo_model, hf_model, sample_data, compare_outputs, model_hf_dtype=args.hf_precision, device="cpu"
    )
    assert max_absolute_diff <= atol


@pytest.mark.skip(reason="Limited capacity on CI runners, should work locally")
@pytest.mark.internal
@pytest.mark.needs_gpu
def test_esm2_650m_model_hf_to_nemo_conversion_parameter_gradient_response_fp16_on_gpu(
    esm2_650m_hf_model_setup, atol=3e-4
):
    hf_model_name, hf_model_size, hf_model_dir = esm2_650m_hf_model_setup
    args = get_hf_to_nemo_conversion_args(hf_model_name, hf_model_dir, hf_model_size, precision=16)
    nemo_model, hf_model = convert_esm_hf_model_to_nemo(args)
    # assert the NeMo model was saved in `output_nemo_model_file`
    assert os.path.exists(args.output_nemo_model_file)

    sample_data = load_sample_protein_sequence_data()
    max_absolute_diff, mean_relative_diff = run_parameter_gradient_sanity_check(
        nemo_model, hf_model, sample_data, compare_outputs, model_hf_dtype=args.hf_precision, device="cuda"
    )
    assert max_absolute_diff <= atol


@pytest.mark.internal
@pytest.mark.skip(reason="Limited capacity on CI runners, should work locally")
def test_esm2_3b_model_hf_to_nemo_conversion_parameter_gradient_response_fp16_on_gpu(
    esm2_3b_hf_model_setup, atol=3e-4
):
    # test needs large GPU due to backprop memory requirements
    hf_model_name, hf_model_size, hf_model_dir = esm2_3b_hf_model_setup
    args = get_hf_to_nemo_conversion_args(hf_model_name, hf_model_dir, hf_model_size, precision=16)
    nemo_model, hf_model = convert_esm_hf_model_to_nemo(args)
    # assert the NeMo model was saved in `output_nemo_model_file`
    assert os.path.exists(args.output_nemo_model_file)

    sample_data = load_sample_protein_sequence_data()
    max_absolute_diff, mean_relative_diff = run_parameter_gradient_sanity_check(
        nemo_model, hf_model, sample_data, compare_outputs, model_hf_dtype=args.hf_precision, device="cuda"
    )
    assert max_absolute_diff <= atol


@pytest.mark.xfail(
    reason="Difference in parameters after conversion, probably due to missing accounting of some layers"
)
@pytest.mark.internal
def test_esm2_650m_model_hf_to_nemo_conversion_trainable_parameter_count(esm2_650m_hf_model_setup):
    hf_model_name, hf_model_size, hf_model_dir = esm2_650m_hf_model_setup
    args = get_hf_to_nemo_conversion_args(hf_model_name, hf_model_dir, hf_model_size, use_nemo_apex=False)
    nemo_model, hf_model = convert_esm_hf_model_to_nemo(args)
    # assert the NeMo model was saved in `output_nemo_model_file`
    assert os.path.exists(args.output_nemo_model_file)
    # get number of trainable parameters of both models and compare
    hf_trainable_params = sum(p.numel() for p in hf_model.parameters() if p.requires_grad)
    nemo_trainable_params = sum(p.numel() for p in nemo_model.parameters() if p.requires_grad)

    # correct for rotary_pos_emb.inv_freq parameters that are stored in HF for each layer
    # but in NeMo one for the whole model since they do not change
    rotary_embedding_param = nemo_model.model.language_model.rotary_pos_emb.inv_freq.numel()
    hf_config = vars(hf_model.config)
    num_layers = hf_config["num_hidden_layers"]
    additional_rope_params = rotary_embedding_param * (num_layers - 1)
    # correct for zero padding tensors added to prediction head, embedding layers and lm head decoder
    # to map vocabulary size from 33 to 128 in order to conform with NeMo requirements
    hf_weights = hf_model.state_dict()
    num_padding_rows = (
        128 - hf_config["vocab_size"]
    )  # 128 is make_vocab_size_divisible_by attribute from esm nemo config file
    num_padding_weight_cols = hf_weights["esm.embeddings.word_embeddings.weight"].size(1)
    padding_weight_params = num_padding_rows * num_padding_weight_cols
    padding_bias_params = num_padding_rows

    num_padding_weight_lm_head_decoder_cols = hf_weights["lm_head.decoder.weight"].size(1)
    padding_weight_lm_head_decoder_params = num_padding_rows * num_padding_weight_lm_head_decoder_cols

    # correct for parameters not included in NeMo ESM model (positional embeddings, contact head)
    contact_head_params = (
        hf_weights["esm.contact_head.regression.weight"].numel()
        + hf_weights["esm.contact_head.regression.bias"].numel()
    )
    positional_embedding_params = hf_weights["esm.embeddings.position_embeddings.weight"].numel()

    additional_params = (
        additional_rope_params
        + contact_head_params
        + positional_embedding_params
        - padding_weight_params
        - padding_bias_params
        - padding_weight_lm_head_decoder_params
    )
    assert hf_trainable_params == nemo_trainable_params + additional_params


@pytest.mark.xfail(
    reason="Difference in parameters after conversion, probably due to missing accounting of some layers"
)
@pytest.mark.internal
@pytest.mark.skip(reason="Limited capacity on CI runners, should work locally")
def test_esm2_3b_model_hf_to_nemo_conversion_trainable_parameter_count(esm2_3b_hf_model_setup):
    hf_model_name, hf_model_size, hf_model_dir = esm2_3b_hf_model_setup
    args = get_hf_to_nemo_conversion_args(hf_model_name, hf_model_dir, hf_model_size, use_nemo_apex=False)
    nemo_model, hf_model = convert_esm_hf_model_to_nemo(args)
    # assert the NeMo model was saved in `output_nemo_model_file`
    assert os.path.exists(args.output_nemo_model_file)
    hf_trainable_params = sum(p.numel() for p in hf_model.parameters() if p.requires_grad)
    nemo_trainable_params = sum(p.numel() for p in nemo_model.parameters() if p.requires_grad)

    # correct for rotary_pos_emb.inv_freq parameters that are stored in HF for each layer
    # but in NeMo one for the whole model since they do not change
    rotary_embedding_param = nemo_model.model.language_model.rotary_pos_emb.inv_freq.numel()
    hf_config = vars(hf_model.config)
    num_layers = hf_config["num_hidden_layers"]
    additional_rope_params = rotary_embedding_param * (num_layers - 1)
    # correct for zero padding tensors added to prediction head, embedding layers and lm head decoder
    # to map vocabulary size from 33 to 128 in order to conform with NeMo requirements
    hf_weights = hf_model.state_dict()
    num_padding_rows = (
        128 - hf_config["vocab_size"]
    )  # 128 is make_vocab_size_divisible_by attribute from esm nemo config file
    num_padding_weight_cols = hf_weights["esm.embeddings.word_embeddings.weight"].size(1)
    padding_weight_params = num_padding_rows * num_padding_weight_cols
    padding_bias_params = num_padding_rows

    num_padding_weight_lm_head_decoder_cols = hf_weights["lm_head.decoder.weight"].size(1)
    padding_weight_lm_head_decoder_params = num_padding_rows * num_padding_weight_lm_head_decoder_cols

    # correct for parameters not included in NeMo ESM model (positional embeddings, contact head)
    contact_head_params = (
        hf_weights["esm.contact_head.regression.weight"].numel()
        + hf_weights["esm.contact_head.regression.bias"].numel()
    )
    positional_embedding_params = hf_weights["esm.embeddings.position_embeddings.weight"].numel()

    additional_params = (
        additional_rope_params
        + contact_head_params
        + positional_embedding_params
        - padding_weight_params
        - padding_bias_params
        - padding_weight_lm_head_decoder_params
    )
    assert hf_trainable_params == nemo_trainable_params + additional_params
