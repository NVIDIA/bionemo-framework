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
from transformers import AutoModel

from bionemo.amplify.model import AMPLIFYConfig
from bionemo.amplify.tokenizer import BioNeMoAMPLIFYTokenizer
from bionemo.llm.lightning import BionemoLightningModule
from bionemo.llm.model.biobert.lightning import biobert_lightning_module


@io.model_importer(BionemoLightningModule, "hf")
class HFAMPLIFYImporter(io.ModelConnector[AutoModel, BionemoLightningModule]):
    """Converts a Hugging Face ESM-2 model to a NeMo ESM-2 model."""

    def init(self) -> BionemoLightningModule:
        """Initialize the converted model."""
        return biobert_lightning_module(self.config, tokenizer=self.tokenizer)

    def apply(self, output_path: Path) -> Path:
        """Applies the transformation."""
        source = AutoModel.from_pretrained(str(self), trust_remote_code=True, torch_dtype="auto")
        target = self.init()
        trainer = self.nemo_setup(target)
        self.convert_state(source, target)
        self.nemo_save(output_path, trainer)
        teardown(trainer, target)
        return output_path

    def convert_state(self, source, target):
        """Converting HF state dict to NeMo state dict."""
        mapping = {
            "encoder.weight": "embedding.word_embeddings.weight",
            "transformer_encoder.*.wo.weight": "encoder.layers.*.self_attention.linear_proj.weight",
            "transformer_encoder.*.ffn.w12.weight": "encoder.layers.*.mlp.linear_fc1.weight",
            "transformer_encoder.*.ffn.w3.weight": "encoder.layers.*.mlp.linear_fc2.weight",
            "transformer_encoder.*.attention_norm.weight": "encoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
            "transformer_encoder.*.ffn_norm.weight": "encoder.layers.*.mlp.linear_fc1.layer_norm_weight",
            "layer_norm_2.weight": "encoder.final_layernorm.weight",
            "decoder.weight": "output_layer.weight",
            "decoder.bias": "output_layer.bias",
        }

        # lm_head.bias
        return io.apply_transforms(
            source,
            target,
            mapping=mapping,
            transforms=[_import_qkv_weight],
            # transforms=[_pad_embeddings, _pad_bias, _import_qkv_weight],
        )

    @property
    def tokenizer(self) -> BioNeMoAMPLIFYTokenizer:
        """We just have the one tokenizer for ESM-2."""
        return BioNeMoAMPLIFYTokenizer()

    @property
    def config(self) -> AMPLIFYConfig:
        """Returns the transformed ESM-2 config given the model tag."""
        source = HFAutoConfig.from_pretrained(str(self), trust_remote_code=True)
        output = AMPLIFYConfig(
            num_layers=source.num_hidden_layers,
            hidden_size=source.hidden_size,
            ffn_hidden_size=source.intermediate_size,
            position_embedding_type="rope",
            num_attention_heads=source.num_attention_heads,
            seq_length=source.max_length,
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
        "transformer_encoder.*.q.weight",
        "transformer_encoder.*.k.weight",
        "transformer_encoder.*.v.weight",
    ),
    target_key="encoder.layers.*.self_attention.linear_qkv.weight",
)
def _import_qkv_weight(ctx: io.TransformCTX, query, key, value):
    """Pad the embedding layer to the new input dimension."""
    concat_weights = torch.cat((query, key, value), dim=0)
    input_shape = concat_weights.size()
    np = ctx.target.config.num_attention_heads
    # transpose weights
    # [sequence length, batch size, num_splits_model_parallel * attention head size * #attention heads]
    # --> [sequence length, batch size, attention head size * num_splits_model_parallel * #attention heads]
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
    # transpose biases
    # [num_splits_model_parallel * attention head size * #attention heads]
    # --> [attention head size * num_splits_model_parallel * #attention heads]
    concat_biases = concat_biases.view(3, np, -1)
    concat_biases = concat_biases.transpose(0, 1).contiguous()
    concat_biases = concat_biases.view(*input_shape)
    return concat_biases
