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


from pathlib import Path

import torch
from nemo.lightning import io, teardown
from nemo.lightning.pytorch.utils import dtype_from_hf
from transformers import AutoConfig as HFAutoConfig
from transformers import AutoModelForMaskedLM

from bionemo.esm2.data.tokenizer import BioNeMoESMTokenizer, get_tokenizer
from bionemo.esm2.model.model import ESM2Config
from bionemo.llm.lightning import BionemoLightningModule
from bionemo.llm.model.biobert.lightning import biobert_lightning_module


@io.model_importer(BionemoLightningModule, "hf")
class HFESM2Importer(io.ModelConnector[AutoModelForMaskedLM, BionemoLightningModule]):
    """Converts a Hugging Face ESM-2 model to a NeMo ESM-2 model."""

    def init(self) -> BionemoLightningModule:
        """Initialize the converted model."""
        return biobert_lightning_module(self.config, tokenizer=self.tokenizer)

    def apply(self, output_path: Path) -> Path:
        """Applies the transformation.

        Largely inspired by
        https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/features/hf-integration.html
        """
        source = AutoModelForMaskedLM.from_pretrained(str(self), trust_remote_code=True, torch_dtype="auto")
        target = self.init()
        trainer = self.nemo_setup(target)
        self.convert_state(source, target)
        self.nemo_save(output_path, trainer)

        print(f"Converted ESM-2 model to Nemo, model saved to {output_path}")

        teardown(trainer, target)
        del trainer, target

        return output_path

    def convert_state(self, source, target):
        """Converting HF state dict to NeMo state dict."""
        mapping = {
            # "esm.encoder.layer.0.attention.self.rotary_embeddings.inv_freq": "rotary_pos_emb.inv_freq",
            "esm.encoder.layer.*.attention.output.dense.weight": "encoder.layers.*.self_attention.linear_proj.weight",
            "esm.encoder.layer.*.attention.output.dense.bias": "encoder.layers.*.self_attention.linear_proj.bias",
            "esm.encoder.layer.*.attention.LayerNorm.weight": "encoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
            "esm.encoder.layer.*.attention.LayerNorm.bias": "encoder.layers.*.self_attention.linear_qkv.layer_norm_bias",
            "esm.encoder.layer.*.intermediate.dense.weight": "encoder.layers.*.mlp.linear_fc1.weight",
            "esm.encoder.layer.*.intermediate.dense.bias": "encoder.layers.*.mlp.linear_fc1.bias",
            "esm.encoder.layer.*.output.dense.weight": "encoder.layers.*.mlp.linear_fc2.weight",
            "esm.encoder.layer.*.output.dense.bias": "encoder.layers.*.mlp.linear_fc2.bias",
            "esm.encoder.layer.*.LayerNorm.weight": "encoder.layers.*.mlp.linear_fc1.layer_norm_weight",
            "esm.encoder.layer.*.LayerNorm.bias": "encoder.layers.*.mlp.linear_fc1.layer_norm_bias",
            "esm.encoder.emb_layer_norm_after.weight": "encoder.final_layernorm.weight",
            "esm.encoder.emb_layer_norm_after.bias": "encoder.final_layernorm.bias",
            "lm_head.dense.weight": "lm_head.dense.weight",
            "lm_head.dense.bias": "lm_head.dense.bias",
            "lm_head.layer_norm.weight": "lm_head.layer_norm.weight",
            "lm_head.layer_norm.bias": "lm_head.layer_norm.bias",
        }

        # lm_head.bias
        return io.apply_transforms(
            source,
            target,
            mapping=mapping,
            transforms=[_pad_embeddings, _pad_bias, _import_qkv_weight, _import_qkv_bias],
        )

    @property
    def tokenizer(self) -> BioNeMoESMTokenizer:
        """We just have the one tokenizer for ESM-2."""
        return get_tokenizer()

    @property
    def config(self) -> ESM2Config:
        """Returns the transformed ESM-2 config given the model tag."""
        source = HFAutoConfig.from_pretrained(str(self), trust_remote_code=True)
        output = ESM2Config(
            num_layers=source.num_hidden_layers,
            hidden_size=source.hidden_size,
            ffn_hidden_size=source.intermediate_size,
            position_embedding_type="rope",
            num_attention_heads=source.num_attention_heads,
            seq_length=source.max_position_embeddings,
            fp16=(dtype_from_hf(source) == torch.float16),
            bf16=(dtype_from_hf(source) == torch.bfloat16),
            params_dtype=dtype_from_hf(source),
        )

        return output


@io.state_transform(
    source_key="esm.embeddings.word_embeddings.weight",
    target_key="embedding.word_embeddings.weight",
)
def _pad_embeddings(ctx: io.TransformCTX, source_embed):
    """Pad the embedding layer to the new input dimension."""
    nemo_embedding_dimension = ctx.target.config.make_vocab_size_divisible_by
    hf_embedding_dimension = source_embed.size(0)
    num_padding_rows = nemo_embedding_dimension - hf_embedding_dimension
    padding_rows = torch.zeros(num_padding_rows, source_embed.size(1))
    return torch.cat((source_embed, padding_rows), dim=0)


@io.state_transform(
    source_key="lm_head.bias",
    target_key="output_layer.bias",
)
def _pad_bias(ctx: io.TransformCTX, source_bias):
    """Pad the embedding layer to the new input dimension."""
    nemo_embedding_dimension = ctx.target.config.make_vocab_size_divisible_by
    hf_embedding_dimension = source_bias.size(0)
    output_bias = torch.zeros(nemo_embedding_dimension, dtype=source_bias.dtype, device=source_bias.device)
    output_bias[:hf_embedding_dimension] = source_bias
    return output_bias


@io.state_transform(
    source_key=(
        "esm.encoder.layer.*.attention.self.query.weight",
        "esm.encoder.layer.*.attention.self.key.weight",
        "esm.encoder.layer.*.attention.self.value.weight",
    ),
    target_key="encoder.layers.*.self_attention.linear_qkv.weight",
)
def _import_qkv_weight(ctx: io.TransformCTX, query, key, value):
    """Pad the embedding layer to the new input dimension."""
    concat_weights = torch.cat((query, key, value), dim=0)
    input_shape = concat_weights.size()
    np = ctx.target.config.num_attention_heads
    concat_weights = concat_weights.view(3, np, -1, query.size()[-1])
    concat_weights = concat_weights.transpose(0, 1).contiguous()
    concat_weights = concat_weights.view(*input_shape)
    return concat_weights


@io.state_transform(
    source_key=(
        "esm.encoder.layer.*.attention.self.query.bias",
        "esm.encoder.layer.*.attention.self.key.bias",
        "esm.encoder.layer.*.attention.self.value.bias",
    ),
    target_key="encoder.layers.*.self_attention.linear_qkv.bias",
)
def _import_qkv_bias(ctx: io.TransformCTX, query, key, value):
    """Pad the embedding layer to the new input dimension."""
    concat_biases = torch.cat((query, key, value), dim=0)
    input_shape = concat_biases.size()
    np = ctx.target.config.num_attention_heads
    concat_biases = concat_biases.view(3, np, -1)
    concat_biases = concat_biases.transpose(0, 1).contiguous()
    concat_biases = concat_biases.view(*input_shape)
    return concat_biases
