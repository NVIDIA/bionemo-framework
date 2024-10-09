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


# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except i fn compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""
Conversion script to convert Huggingface ESM2 checkpoints into nemo checkpoints.
  Example to run this conversion script:
    python convert_hf_esm2_to_nemo.py \
     --input_hf_model_file <path_to_hf_checkpoints_folder> \
     --output_nemo_model_file <path_to_output_nemo_file> \

This scrips assumes HuggingFace checkpoints of ESM2 model are already downloaded.
"""

import argparse
import gc
from argparse import ArgumentParser
from collections import OrderedDict
from typing import Callable, Dict, List, Tuple, Union

import torch
from hydra import compose, initialize
from nemo.collections.nlp.models.language_modeling.megatron.bert_model import bert_extended_attention_mask
from nemo.collections.nlp.modules.common.megatron.utils import build_position_ids
from nemo.collections.nlp.parts.nlp_overrides import (
    GradScaler,
    MegatronHalfPrecisionPlugin,
    NLPDDPStrategy,
    PipelineMixedPrecisionPlugin,
)
from nemo.utils import logging
from omegaconf import DictConfig
from pytorch_lightning.core.saving import _load_state as ptl_load_state
from pytorch_lightning.trainer.trainer import Trainer
from transformers import EsmForMaskedLM

from bionemo.model.protein.esm1nv import ESM1nvModel, ESM2nvModel
from bionemo.model.protein.esm1nv.base import ESMnvMegatronBertModel
from bionemo.utils.connectors import BioNeMoSaveRestoreConnector
from tests.esm2_hf_to_nemo_conversion.compare import compare_outputs


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--input_hf_model_file",
        type=str,
        default=None,
        required=True,
        help="Path to Huggingface ESM2 checkpoints",
    )
    parser.add_argument(
        "--output_nemo_model_file",
        type=str,
        default=None,
        required=True,
        help="Path to output .nemo file.",
    )
    parser.add_argument("--run_impulse_sanity_check", action="store_true")

    parser.add_argument("--precision", type=str, default="32", help="Model precision")

    parser.add_argument("--hf-precision", type=str, default=argparse.SUPPRESS, help="HF Model precision")

    parser.add_argument(
        "--model_size",
        type=str,
        default="650M",
        help="ESM2 model size",
        choices={"8M", "650M", "3B", "15B"},
    )

    parser.add_argument(
        "--use_nemo_apex",
        action="store_true",
        help="Whether to use optimized apex layers: FusedLayerNorm and RowParallelLinear",
    )

    parser.add_argument(
        "--model_class",
        default="esm1",
        choices={"esm1", "esm2"},
    )

    args = parser.parse_args()
    return args


def load_config(args, esm_config: dict) -> DictConfig:
    """
    Updates the default BioNeMo ESM2nv configuration from commandline arguments `args`
    and additional model architecture settings from the
    HuggingFace configuration `esm2_config`

    Parameters:
    ----------
    args (Namespace): A Namespace object containing command line arguments.
                      It includes model size, and flags to determine certain
                      model behaviors such as the use of Nemo Apex layers.
    esm_config: A dictionary containing HuggingFace ESM2 configuration parameters.
                       It includes keys like 'num_hidden_layers', 'hidden_size'
                       to set respective ESM2 model parameters.

    Note:
    The function assumes that the Hydra configuration files are located in
    "../../examples/protein/esm2nv/conf" relative to the current working directory.

    Returns:
    -------
    OmegaConf.DictConfig: A Hydra configuration object for the Nemo model,
                          customized for the ESM2 model to convert.
    """
    initialize(
        version_base=None,
        config_path="../../examples/protein/esm2nv/conf",
        job_name="esm2_hf_to_nemo_conversion",
    )
    nemo_config = compose(
        config_name=f"pretrain_esm2_{args.model_size}",
        overrides=[
            "++model.use_cpu_initialization=True",
            "++position_embedding_type=rope",  # ESM2 uses relative positional encoding 'ROPE' to extrapolate to longer sequences unseen during training
            "++hidden_dropout=0",  # ESM2 removes dropout from hidden layers and attention
            "++embedding_use_attention_mask=True",  # ESM2 uses attention masking on the embeddings
            "++embedding_token_dropout=True",  # ESM2 rescales embeddings based on masked tokens
            "++mask_token_id=32",  # Needed for token dropout rescaling
            "++attention_dropout=0.0",  # ESM2 does not use attention dropout
            "++use_esm_attention=True",  # Use specific attention modifications for ESM2
            "++normalize_attention_scores=False",  # ESM2 does not use normalized attention scores
            "++esm_gelu=True",  # ESM2 uses custom gelu in the ML layer
            "++bias_gelu_fusion=False",  # disable bias_gelu_fusion to use ESM2's custom GELU function
            "++model.tensor_model_parallel_size=1",  # disable tensor model parallelism for the conversion
            "++model.share_embeddings_and_output_weights=True",  # to decide whether use a new linear layer to compute final scores (False) or reuse embedding weights (True), here true since in ESM2 swarm this conversion was forgotten
        ],
    )

    nemo_config.model.num_layers = int(esm_config["num_hidden_layers"])
    nemo_config.model.hidden_size = esm_config["hidden_size"]
    nemo_config.model.ffn_hidden_size = esm_config["intermediate_size"]
    nemo_config.model.num_attention_heads = esm_config["num_attention_heads"]
    nemo_config.model.init_method_std = esm_config["initializer_range"]

    if args.use_nemo_apex:
        logging.info("Use optimized Apex Layers for LayerNorm and Linear projections")
        nemo_config.model.use_pt_layernorm = False
        nemo_config.model.use_pt_mlp_out = False
    else:
        # Use base PyTorch Layers for LayerNorm and Linear projections
        nemo_config.model.use_pt_layernorm = True
        nemo_config.model.use_pt_mlp_out = True

    return nemo_config


def load_model(cls: ESMnvMegatronBertModel, checkpoint: Dict, strict: bool, **kwargs) -> ESMnvMegatronBertModel:
    """
    Loads an ESMnvMegatronBertModel model from a checkpoint with optional configuration overrides.

    The function also handles the registration of tokenizer artifacts if
    they are specified in the model configuration.

    Parameters:
    ----------
    cls: The class on which this method is called. It should be a
        ESMnvMegatronBertModel class that is capable of loading from a checkpoint.
    checkpoint: A dictionary containing the model's weights.
    strict: A flag indicating whether the loading should be strict, i.e.,
        the checkpoint and the model must exactly match.

    The method performs the following operations:
        - Loads the model state from the checkpoint.
        - Aligns model parameters and buffers with those in the checkpoint.
        - Handles unexpected keys in the model and checkpoint.
        - Registers tokenizer artifacts if they are specified in the model's
        configuration.
        - Sets the model to evaluation mode before returning it.

    Returns:
    -------
        ESMnvMegatronBertModel: The loaded model instance, set to evaluation mode.
    """
    try:
        if "cfg" in kwargs:
            model = ptl_load_state(cls, checkpoint, strict=strict, **kwargs)
        else:
            cfg = checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY]
            model = cls(cfg=cfg.model, **kwargs)

            for name, module in model.named_parameters():
                if name in checkpoint["state_dict"]:
                    module.data = checkpoint["state_dict"][name]
                    checkpoint["state_dict"].pop(name)
                else:
                    logging.warning(f"Unexpected key: {name} not in checkpoint but in model.")

            for name, buffer in model.named_buffers():
                if name in checkpoint["state_dict"]:
                    buffer.data = checkpoint["state_dict"][name]
                    checkpoint["state_dict"].pop(name)

            # check if the state_dict of the model has any keys
            if checkpoint["state_dict"]:
                raise RuntimeError(
                    f"Additional keys: {checkpoint['state_dict'].keys()} in checkpoint but not in model."
                )

            # register the artifacts
            if cfg.model.tokenizer.model is not None:
                model.register_artifact("tokenizer.tokenizer_model", cfg.tokenizer.model)
            if cfg.model.tokenizer.vocab_file is not None:
                model.register_artifact("tokenizer.vocab_file", cfg.tokenizer.vocab_file)
            if cfg.model.tokenizer.merge_file is not None:
                model.register_artifact("tokenizer.merge_file", cfg.tokenizer.merge_file)
    finally:
        cls._set_model_restore_state(is_being_restored=False)
    return model.eval()


def load_sample_protein_sequence_data(max_length: int = 1022) -> List[Tuple[str, str]]:
    """
    Generates sample protein sequences for sanity checks, including mask tokens.

    Parameters:
    ----------
    max_length: The maximum length of the protein sequences to be considered.
            Defaults to 1022.

    Returns:
    -------
    List[Tuple[str, str]]: list of sample proteins with a name and a sequence.
    """

    sample_data = [
        (
            "protein1",
            "MNGTEGPNFYVPFSNATGVVRSPFEYPQYYLAEPWQFSMLAAYMFLLIVLGFPINFLTLYVTVQHKKLRTPLNYILLNLAVADLFMVLGGFTSTLYTSLHGYFVFGPTGCNLEGFFATLGGEIALWSLVVLAIERYVVVCKPMSNFRFGENHAIMGVAFTWVMALACAAPPLAGWSRYIPEGLQCSCGIDYYTLKPEVNNESFVIYMFVVHFTIPMIIIFFCYGQLVFTVKEAAAQQQESATTQKAEKEVTRMVIIMVIAFLICWVPYASVAFYIFTHQGSNFGPIFMTIPAFFAKSAAIYNPVIYIMMNKQFRNCMLTTICCGKNPLGDDEASATVSKTETSQVAPA",
        ),
        ("protein2", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLA"),
        (
            "protein3",
            "MKTVRQERLKSI<mask>RILERSKEPVSGAQLAEELS<mask>SRQVIVQDIAYLRSLGYN<mask>VATPRGYVLAGG",
        ),
        (
            "protein4",
            "MKTVRQERLKSI<mask>RILERSKEPVSGAQLAEELS<mask>SRQVIVQDIAYLRSLGYN<mask>VATPRGYVLA",
        ),
    ]

    # add another sample protein that uses the maximum length to test this edge case
    sample_data.append(("protein5", (sample_data[0][1] * 3)[:max_length]))
    return sample_data


def run_impulse_sanity_check(
    model_nemo: Union[ESM1nvModel, ESM2nvModel],
    model_hf: EsmForMaskedLM,
    sample_data: List[Tuple[str, str]],
    comparison_function: Callable[[torch.Tensor, torch.Tensor, float], Tuple[float, float]] = compare_outputs,
    model_hf_dtype: torch.dtype = None,
    device: str = "cpu",
) -> Tuple[float, float]:
    """
    Performs an impulse sanity check by comparing the output embeddings from two ESM2 models:
    The converted ESM2nv BioNeMo model and the corresponding Hugging Face (HF) model.

    The function performs the following steps:
    - Tokenizes input sequences using the BioNemo model's tokenizer.
    - Generates hidden embeddings of residues from both the BioNemo and HF models.
    - Compares the outputs from both models to check for consistency.

    Parameters:
    ----------
    model_nemo: A BioNeMo ESM2 model instance for generate embeddings.
    model_hf: A Hugging Face ESM model instance for generating embeddings.
    sample_data: Fake protein sequence data including mask tokens.
    model_hf_dtype: The data type to which the Hugging Face model should
            be converted before generating embeddings.
            If None, the model's current data type is used.
            Defaults to None.

    Returns:
    -------
    Tuple[float, float]: A tuple containing the maximum absolute difference and
        the mean relative difference between the two tensors.

    Note:
    - The 'compare_outputs' function, used for comparing embeddings, is not defined within
      this function and should be available in the scope where this function is used.
    """
    # create tokenizer and create attention masks
    tokens = model_nemo.tokenizer.tokenizer([row[1] for row in sample_data], return_tensors="pt", padding=True)
    tokens["input_ids"] = tokens["input_ids"].to(device)
    tokens["attention_mask"] = tokens["attention_mask"].to(device)
    # position ids from 0 to sequence length are created
    position_ids = build_position_ids(tokens["input_ids"])
    extended_attention_mask = bert_extended_attention_mask(tokens["attention_mask"])

    if model_hf_dtype is None:
        raise ValueError("model_hf_dtype is None, but needs to be specified.")

    model_hf = model_hf.to(device=device, dtype=dtype_from_precision(model_hf_dtype))

    with torch.no_grad():
        # Get hidden embeddings from the HF model
        hf_output_all = model_hf(**tokens, output_hidden_states=True)
        hf_embeddings = hf_output_all.hidden_states[-1]

        # now delete the huggingface model to free GPU RAM for the NeMo model
        del model_hf
        gc.collect()
        torch.cuda.empty_cache()

        # Get hidden embeddings from the converted BioNeMo model
        model_nemo = model_nemo.to(device)
        nemo_embeddings = model_nemo.model.language_model(tokens["input_ids"], position_ids, extended_attention_mask)
    return comparison_function(hf_embeddings.float(), torch.transpose(nemo_embeddings, 0, 1).float())


def run_parameter_gradient_sanity_check(
    model_nemo: Union[ESM1nvModel, ESM2nvModel],
    model_hf: EsmForMaskedLM,
    sample_data: List[Tuple[str, str]],
    comparison_function: Callable[[torch.Tensor, torch.Tensor, float], Tuple[float, float]] = compare_outputs,
    model_hf_dtype: torch.dtype = None,
    device: str = "cpu",
) -> Tuple[float, float]:
    """
    Performs a gradient sanity check by comparing the gradients from two ESM2 models:
    The converted ESM2nv BioNeMo model and the corresponding Hugging Face (HF) model.

    The function performs the following steps:
    - Tokenizes input sequences using the BioNemo model's tokenizer.
    - Generates hidden embeddings of residues from both the BioNemo and HF models.
    - Computes the L2 loss of the embedding wrt to a zero tensor and backprops
    - Compares the gradients from both models to check for consistency.

    Parameters:
    ----------
    model_nemo: A BioNeMo ESM2 model instance for generate embeddings.
    model_hf: A Hugging Face ESM model instance for generating embeddings.
    sample_data: Fake protein sequence data including mask tokens.
    model_hf_dtype: The data type to which the Hugging Face model should
            be converted before generating embeddings.
            If None, the model's current data type is used.
            Defaults to None.

    Returns:
    -------
    Tuple[float, float]: A tuple containing the maximum absolute difference and
        the mean relative difference between the two tensors.

    Note:
    - The 'compare_outputs' function, used for comparing gradients, is not defined within
      this function and should be available in the scope where this function is used.
    """
    # create tokenizer and create attention masks
    tokens = model_nemo.tokenizer.tokenizer([row[1] for row in sample_data], return_tensors="pt", padding=True)
    tokens["input_ids"] = tokens["input_ids"].to(device)
    tokens["attention_mask"] = tokens["attention_mask"].to(device)
    # position ids from 0 to sequence length are created
    position_ids = build_position_ids(tokens["input_ids"])
    extended_attention_mask = bert_extended_attention_mask(tokens["attention_mask"])

    if model_hf_dtype is None:
        raise ValueError("model_hf_dtype is None, but needs to be specified.")

    model_hf = model_hf.to(device=device, dtype=dtype_from_precision(model_hf_dtype))

    # Get hidden embeddings from the HF model
    hf_output_all = model_hf(**tokens, output_hidden_states=True)
    hf_embeddings = hf_output_all.hidden_states[-1]
    # create zero_embeddings and calculate l2 loss between it and both nemo_embeddings and hf_embeddings
    zero_embeddings = torch.zeros_like(hf_embeddings)
    mse_loss = torch.nn.MSELoss()
    l2_loss_hf = mse_loss(zero_embeddings, hf_embeddings)
    l2_loss_hf.backward()
    # extract a suitable subset of the gradients in the model for gradient checking
    num_layers = len(model_nemo.model.language_model.encoder.layers)
    hf_grads_dense_attention_weights = [
        model_hf.esm.encoder.layer[layer_nr].attention.output.dense.weight.grad for layer_nr in range(num_layers)
    ]
    hf_grads_dense_attention_weights = torch.stack(hf_grads_dense_attention_weights, dim=0)

    # now delete the huggingface model to free GPU RAM for the NeMo model
    del model_hf
    gc.collect()
    torch.cuda.empty_cache()

    # Get hidden embeddings from the converted BioNeMo model
    model_nemo = model_nemo.to(device)
    nemo_embeddings = model_nemo.model.language_model(tokens["input_ids"], position_ids, extended_attention_mask)
    nemo_embeddings = torch.transpose(nemo_embeddings, 0, 1)
    # calculate l2 loss between zero embedding and nemo_embeddings
    l2_loss_nemo = mse_loss(zero_embeddings, nemo_embeddings)
    l2_loss_nemo.backward()
    # extract a suitable subset of the gradients in the model for gradient checking
    nemo_grads_dense_attention_weights = [
        model_nemo.model.language_model.encoder.layers[layer_nr].self_attention.dense.weight.grad
        for layer_nr in range(num_layers)
    ]
    nemo_grads_dense_attention_weights = torch.stack(nemo_grads_dense_attention_weights, dim=0)

    return comparison_function(hf_grads_dense_attention_weights, nemo_grads_dense_attention_weights)


def dtype_from_precision(precision: Union[str, int]) -> torch.dtype:
    """
    Determines the appropriate PyTorch data type (dtype) based on the given precision.

    The function handles the following cases:
        - 32: Maps to torch.float32.
        - 16 or "16" or "16-mixed": Maps to torch.float16.
        - "bf16" or "bf16-mixed": Maps to torch.bfloat16.
        - For any other values, the function defaults to torch.float32 as a fallback.

    Parameters:
    ----------
    precision: The desired float precision for setting model's weights. This can be an integer (e.g., 32, 16)
            or a string (e.g., "16", "16-mixed", "bf16", "bf16-mixed").

    Returns:
    --------
        torch.dtype: The corresponding PyTorch data type for the specified precision level.
    """
    if precision == 32:
        dtype = torch.float32
    elif precision in [16, "16", "16-mixed"]:
        dtype = torch.float16
    elif precision in ["bf16", "bf16-mixed"]:
        dtype = torch.bfloat16
    else:
        dtype = torch.float32  # fallback
    return dtype


def convert_esm_hf_model_to_nemo(args):
    """
    Converts a Hugging Face ESM2 model checkpoint to a BioNeMo ESMnvMegatronBertModel model.

    This function performs the following steps:
        - Loads the HF ESM2 model weights,
        - sets the float precision of model's weights,
        - maps the HF weights to NeMo ones,
        - saves the converted checkpoints as a NeMo model (.nemo checkpoints).
        - Additionally, the function can perform a sanity check to verify the
        outputs of the conveted model.

    Parameters:
    ----------
    - args (Namespace): A Namespace object containing the following attributes:
        - input_hf_model_file (str): Path to the input Hugging Face model checkpoint.
        - output_nemo_model_file (str): Path where the converted NeMo model will be saved.
        - precision (str): Desired precision for the model ('32', '16', 'bf16', 'bf16-mixed').
        - model_class (str): Type of model class to use ('esm1' or 'esm2').
        - hf_precision (str, optional): Precision setting for the Hugging Face model.
        - run_impulse_sanity_check (bool): Flag to indicate whether to perform a sanity check after conversion.

    The function adjusts the model's parameters and configurations based on the provided arguments. It
    also handles the conversion of model weights and biases to match the NeMo model specifications.

    """
    logging.info(f"loading checkpoint {args.input_hf_model_file}")
    model = EsmForMaskedLM.from_pretrained(args.input_hf_model_file)
    hf_config = vars(model.config)

    nemo_config = load_config(args, hf_config)

    # determine correct precision to use
    if args.precision in ["32", "16"]:
        precision = int(float(args.precision))
    elif args.precision in ["bf16", "bf16-mixed"]:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            precision = args.precision
        else:
            logging.warning("BF16 is not supported on this device. Using FP16 instead.")
            precision = args.precision[2:]  # prune bf in string
    else:
        precision = args.precision

    plugins = []
    if precision in [16, "16", "bf16", "16-mixed", "bf16-mixed"]:
        scaler = None
        if precision in [16, "16", "16-mixed"]:
            scaler = GradScaler(
                init_scale=nemo_config.get("native_amp_init_scale", 2**32),
                growth_interval=nemo_config.get("native_amp_growth_interval", 1000),
                hysteresis=nemo_config.get("hysteresis", 2),
            )
            # MixedPrecisionPlugin in PTL >= 2.0 requires precision to be 16-mixed or bf16-mixed
            plugin_precision = "16-mixed"
        else:
            plugin_precision = "bf16-mixed"
        if nemo_config.get("megatron_amp_O2", False):
            plugins.append(MegatronHalfPrecisionPlugin(precision=plugin_precision, device="cuda", scaler=scaler))
        else:
            plugins.append(PipelineMixedPrecisionPlugin(precision=plugin_precision, device="cuda", scaler=scaler))

    dtype = dtype_from_precision(precision)
    nemo_config.trainer.precision = precision

    # instantiate trainer since this is needed for the NeMo model
    trainer = Trainer(
        plugins=plugins,
        accelerator="cpu",
        precision=precision,
        strategy=NLPDDPStrategy(),
    )

    num_layers = hf_config["num_hidden_layers"]
    hf_weights = model.state_dict()

    def convert_to_float(x):
        return x.float()

    checkpoint = OrderedDict()
    checkpoint["state_dict"] = OrderedDict()

    # pad the embeddings weights (33 tokens) to 128 to match with the numerical optimization of of NeMo bert class
    vocab_size = hf_config["vocab_size"]
    num_padding_rows = nemo_config.model.make_vocab_size_divisible_by - vocab_size
    padding_rows = torch.zeros(num_padding_rows, hf_weights["esm.embeddings.word_embeddings.weight"].size(1))
    new_matrix_weights = torch.cat((hf_weights["esm.embeddings.word_embeddings.weight"], padding_rows), dim=0)
    checkpoint["state_dict"]["model.language_model.embedding.word_embeddings.weight"] = convert_to_float(
        new_matrix_weights
    )

    # pad the bias output shape to config.make_vocab_size_divisible_by to match with the numerical optimization of NeMo bert class
    padding_rows = torch.zeros(num_padding_rows)
    new_matrix_bias = torch.cat((hf_weights["lm_head.bias"], padding_rows))

    checkpoint["state_dict"]["model.lm_head.bias"] = convert_to_float(new_matrix_bias)

    for layer_number in range(num_layers):
        # Layer norm before attention
        checkpoint["state_dict"][f"model.language_model.encoder.layers.{layer_number}.input_layernorm.weight"] = (
            convert_to_float(hf_weights[f"esm.encoder.layer.{layer_number}.attention.LayerNorm.weight"])
        )
        checkpoint["state_dict"][f"model.language_model.encoder.layers.{layer_number}.input_layernorm.bias"] = (
            convert_to_float(hf_weights[f"esm.encoder.layer.{layer_number}.attention.LayerNorm.bias"])
        )

        # Attention weights are stored in three vectors in hf, but one vector in NeMo
        k_weight = hf_weights[f"esm.encoder.layer.{layer_number}.attention.self.key.weight"]
        v_weight = hf_weights[f"esm.encoder.layer.{layer_number}.attention.self.value.weight"]
        q_weight = hf_weights[f"esm.encoder.layer.{layer_number}.attention.self.query.weight"]
        concat_weights = torch.cat([q_weight, k_weight, v_weight], dim=0)
        # transpose weights
        # [sequence length, batch size, num_splits_model_parallel * attention head size * #attention heads]
        # --> [sequence length, batch size, attention head size * num_splits_model_parallel * #attention heads]
        input_shape = concat_weights.size()
        np = nemo_config.model.num_attention_heads
        concat_weights = concat_weights.view(3, np, -1, q_weight.size()[-1])
        concat_weights = concat_weights.transpose(0, 1).contiguous()
        concat_weights = concat_weights.view(*input_shape)
        checkpoint["state_dict"][
            f"model.language_model.encoder.layers.{layer_number}.self_attention.query_key_value.weight"
        ] = convert_to_float(concat_weights)

        # Attention bias
        k_bias = hf_weights[f"esm.encoder.layer.{layer_number}.attention.self.key.bias"]
        v_bias = hf_weights[f"esm.encoder.layer.{layer_number}.attention.self.value.bias"]
        q_bias = hf_weights[f"esm.encoder.layer.{layer_number}.attention.self.query.bias"]
        concat_biases = torch.cat([q_bias, k_bias, v_bias], dim=0)
        # transpose biases
        # [num_splits_model_parallel * attention head size * #attention heads]
        # --> [attention head size * num_splits_model_parallel * #attention heads]
        input_shape = concat_biases.size()
        np = nemo_config.model.num_attention_heads
        concat_biases = concat_biases.view(3, np, -1)
        concat_biases = concat_biases.transpose(0, 1).contiguous()
        concat_biases = concat_biases.view(*input_shape)
        checkpoint["state_dict"][
            f"model.language_model.encoder.layers.{layer_number}.self_attention.query_key_value.bias"
        ] = convert_to_float(concat_biases)

        # Attention dense layer
        checkpoint["state_dict"][f"model.language_model.encoder.layers.{layer_number}.self_attention.dense.weight"] = (
            convert_to_float(hf_weights[f"esm.encoder.layer.{layer_number}.attention.output.dense.weight"])
        )
        checkpoint["state_dict"][f"model.language_model.encoder.layers.{layer_number}.self_attention.dense.bias"] = (
            convert_to_float(hf_weights[f"esm.encoder.layer.{layer_number}.attention.output.dense.bias"])
        )

        # Layer norm after Attention
        checkpoint["state_dict"][
            f"model.language_model.encoder.layers.{layer_number}.post_attention_layernorm.weight"
        ] = convert_to_float(hf_weights[f"esm.encoder.layer.{layer_number}.LayerNorm.weight"])
        checkpoint["state_dict"][
            f"model.language_model.encoder.layers.{layer_number}.post_attention_layernorm.bias"
        ] = convert_to_float(hf_weights[f"esm.encoder.layer.{layer_number}.LayerNorm.bias"])

        # MLP: Intermediate + ESMOuput
        checkpoint["state_dict"][f"model.language_model.encoder.layers.{layer_number}.mlp.dense_h_to_4h.weight"] = (
            convert_to_float(hf_weights[f"esm.encoder.layer.{layer_number}.intermediate.dense.weight"])
        )
        checkpoint["state_dict"][f"model.language_model.encoder.layers.{layer_number}.mlp.dense_h_to_4h.bias"] = (
            convert_to_float(hf_weights[f"esm.encoder.layer.{layer_number}.intermediate.dense.bias"])
        )

        checkpoint["state_dict"][f"model.language_model.encoder.layers.{layer_number}.mlp.dense_4h_to_h.weight"] = (
            convert_to_float(hf_weights[f"esm.encoder.layer.{layer_number}.output.dense.weight"])
        )
        checkpoint["state_dict"][f"model.language_model.encoder.layers.{layer_number}.mlp.dense_4h_to_h.bias"] = (
            convert_to_float(hf_weights[f"esm.encoder.layer.{layer_number}.output.dense.bias"])
        )

    # Inv freq tensor for ROPE class is the same for all layers,
    # but is only saved once in NeMo, we therefore choose the ones from layer 0
    checkpoint["state_dict"]["model.language_model.rotary_pos_emb.inv_freq"] = convert_to_float(
        hf_weights["esm.encoder.layer.0.attention.self.rotary_embeddings.inv_freq"]
    )

    # Final Later norm after the list of ESMLayers
    checkpoint["state_dict"]["model.language_model.encoder.final_layernorm.weight"] = convert_to_float(
        hf_weights["esm.encoder.emb_layer_norm_after.weight"]
    )
    checkpoint["state_dict"]["model.language_model.encoder.final_layernorm.bias"] = convert_to_float(
        hf_weights["esm.encoder.emb_layer_norm_after.bias"]
    )

    # Weights for logits head
    checkpoint["state_dict"]["model.lm_head.dense.weight"] = convert_to_float(hf_weights["lm_head.dense.weight"])
    checkpoint["state_dict"]["model.lm_head.dense.bias"] = convert_to_float(hf_weights["lm_head.dense.bias"])
    checkpoint["state_dict"]["model.lm_head.layernorm.weight"] = convert_to_float(
        hf_weights["lm_head.layer_norm.weight"]
    )
    checkpoint["state_dict"]["model.lm_head.layernorm.bias"] = convert_to_float(hf_weights["lm_head.layer_norm.bias"])
    # convert final decoder layer to logprobs (needed for continuing training)
    # pad  to config.make_vocab_size_divisible_by to match with the numerical optimization of NeMo bert class
    # outcommented for now since the original ESM2 model shares this output with the embedding layer
    # num_padding_rows = nemo_config.model.make_vocab_size_divisible_by - vocab_size
    # padding_rows = torch.zeros(num_padding_rows, hf_weights["lm_head.decoder.weight"].size(1))
    # new_matrix_weights = torch.cat((hf_weights["lm_head.decoder.weight"], padding_rows), dim=0)
    # checkpoint["state_dict"]["model.language_model.output_layer.weight"] = convert_to_float(new_matrix_weights)

    model_classes = {
        "esm1": ESM1nvModel,
        "esm2": ESM2nvModel,
    }
    model_cls = model_classes[args.model_class]
    checkpoint[model_cls.CHECKPOINT_HYPER_PARAMS_KEY] = nemo_config
    nemo_model = load_model(
        model_cls,
        checkpoint,
        strict=False,
        trainer=trainer,
    )
    nemo_model._save_restore_connector = BioNeMoSaveRestoreConnector()

    # cast to target precision and disable cpu init
    nemo_model = nemo_model.to(dtype=dtype)
    nemo_model.cfg.use_cpu_initialization = False
    nemo_model.save_to(args.output_nemo_model_file)

    logging.info(f"NeMo model saved to: {args.output_nemo_model_file}")
    if "hf_precision" in args:
        hf_dtype = dtype_from_precision(args.hf_precision)
    else:
        hf_dtype = dtype

    if args.run_impulse_sanity_check:
        sample_data = load_sample_protein_sequence_data()
        max_absolute_difference, mean_relative_difference = run_impulse_sanity_check(
            nemo_model.cuda(), model.cuda(), sample_data, compare_outputs, model_hf_dtype=hf_dtype, device="cuda"
        )
        if max_absolute_difference > 3e-1:
            raise ValueError(
                f"Discrepancy of model outputs reached {max_absolute_difference}, but should not be bigger than 3e-1"
            )
    return nemo_model, model


if __name__ == "__main__":
    args = get_args()
    convert_esm_hf_model_to_nemo(args)
