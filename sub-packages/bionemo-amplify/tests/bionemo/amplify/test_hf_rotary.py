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


import torch
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from transformers import AutoConfig

from bionemo.amplify.hf_rotary import apply_rotary_emb, precompute_freqs_cis
from bionemo.amplify.model import AMPLIFYConfig


def test_rope_embeddings():
    rng = torch.Generator().manual_seed(42)
    q = torch.randn([2, 72, 10, 64], dtype=torch.float32, generator=rng)
    k = torch.randn([2, 72, 10, 64], dtype=torch.float32, generator=rng)

    # AMPLIFY HF Rope
    hf_config = AutoConfig.from_pretrained("chandar-lab/AMPLIFY_120M", trust_remote_code=True)
    freqs_cis = precompute_freqs_cis(hf_config.hidden_size // hf_config.num_attention_heads, hf_config.max_length)
    freqs_cis = freqs_cis[: q.shape[1]]
    q_post, k_post = apply_rotary_emb(q, k, freqs_cis)

    # NeMo Rope
    nemo_config = AMPLIFYConfig(apply_rope_fusion=False, rotary_interleaved=True)
    rotary_pos_layer = RotaryEmbedding(
        kv_channels=nemo_config.kv_channels,
        rotary_percent=nemo_config.rotary_percent,
        rotary_interleaved=nemo_config.rotary_interleaved,
        seq_len_interpolation_factor=nemo_config.seq_len_interpolation_factor,
    )
    rotary_pos_emb = rotary_pos_layer(q.shape[1])
    q_post_nemo = apply_rotary_pos_emb(q.transpose(0, 1).cuda(), rotary_pos_emb.cuda(), config=nemo_config).cpu()
    k_post_nemo = apply_rotary_pos_emb(k.transpose(0, 1).cuda(), rotary_pos_emb.cuda(), config=nemo_config).cpu()

    torch.testing.assert_close(q_post, q_post_nemo.transpose(0, 1))
    torch.testing.assert_close(k_post, k_post_nemo.transpose(0, 1))


# TODO: extend this test to try the DotProductAttention and TEDotProductAttention layers and compare how close the
# outputs are; that seems to be where the outputs between the HF and NeMo implementations are diverging.

# def test_multi_head_attention():
#     rng = torch.Generator().manual_seed(42)
#     q = torch.randn([2, 72, 10, 64], dtype=torch.float32, generator=rng)
#     k = torch.randn([2, 72, 10, 64], dtype=torch.float32, generator=rng)
#     v = torch.randn([2, 72, 10, 64], dtype=torch.float32, generator=rng)

#     attention_mask = torch.ones([2, 72], dtype=torch.float32).bool()
#     attention_mask[0, -7:] = False
#     attention_mask[1, -5:] = False

#     q_new = torch.randn([2, 72, 10, 64], dtype=torch.float32, generator=rng)
#     k_new = torch.randn([2, 72, 10, 64], dtype=torch.float32, generator=rng)
#     v_new = torch.randn([2, 72, 10, 64], dtype=torch.float32, generator=rng)

#     q_new[attention_mask] = q[attention_mask]
#     k_new[attention_mask] = k[attention_mask]
#     v_new[attention_mask] = v[attention_mask]

#     attention_mask_rep = attention_mask.unsqueeze(1).unsqueeze(1).repeat(1, 10, attention_mask.size(-1), 1)

#     attn_output = torch.nn.functional.scaled_dot_product_attention(
#         query=q.transpose(1, 2),
#         key=k.transpose(1, 2),
#         value=v.transpose(1, 2),
#         attn_mask=attention_mask_rep,
#         dropout_p=0,
#     ).transpose(1, 2)

#     attn_output_new = torch.nn.functional.scaled_dot_product_attention(
#         query=q_new.transpose(1, 2),
#         key=k_new.transpose(1, 2),
#         value=v_new.transpose(1, 2),
#         attn_mask=attention_mask_rep,
#         dropout_p=0,
#     ).transpose(1, 2)

#     torch.testing.assert_close(attn_output[attention_mask], attn_output_new[attention_mask])
