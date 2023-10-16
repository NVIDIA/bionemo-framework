# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from typing import List

import torch
from torch.nn.modules.loss import _WeightedLoss


class PerTokenMaskedCrossEntropyLoss(_WeightedLoss):
    def __init__(self, **kwargs):
        super(PerTokenMaskedCrossEntropyLoss, self).__init__(**kwargs)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        masks = target[1]
        target = target[0]
        assert len(input) == len(target)
        loss = 0
        for i in range(len(input)):
            masked_out = mask_tensor(masks[i], input[i]).permute(0, 2, 1)
            cur_loss = self.loss_fn(masked_out, target[i].permute(0, 2, 1))
            loss += cur_loss
        return loss


class ConvNet(torch.nn.Module):
    def __init__(self, embed_dim: int, output_sizes: List[int]):
        super(ConvNet, self).__init__()
        # This is only called "elmo_feature_extractor" for historic reason
        # CNN weights are trained on ProtT5 embeddings
        self.elmo_feature_extractor = torch.nn.Sequential(
            torch.nn.Conv2d(embed_dim, 32, kernel_size=(7, 1), padding=(3, 0)),  # 7x32
            torch.nn.ReLU(),
            torch.nn.Dropout(0.25),
        )
        n_final_in = 32
        self.class_heads = torch.nn.ModuleList([])
        for head_size in output_sizes:
            self.class_heads.append(torch.nn.Conv2d(n_final_in, head_size, kernel_size=(7, 1), padding=(3, 0)))

    def forward(self, x):
        # IN: X = (B x L x F); OUT: (B x F x L, 1)
        x = x.permute(0, 2, 1).unsqueeze(dim=-1)
        x = self.elmo_feature_extractor(x)  # OUT: (B x 32 x L x 1)
        outputs = []
        for head in self.class_heads:
            output = head(x)
            outputs.append(output.squeeze(dim=-1).permute(0, 2, 1))  # OUT: (B x L x output_size)
        return outputs


def mask_tensor(mask, tensor):
    dims = tensor.size()
    mask = torch.repeat_interleave(mask, dims[2]).reshape(dims[0], -1, dims[2]).to("cuda")
    output_tensor = torch.mul(mask, tensor)

    return output_tensor
