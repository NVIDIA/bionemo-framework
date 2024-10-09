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


from typing import List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, mean_squared_error


def mse(output, target):
    if torch.is_tensor(output):
        output = output.detach().to("cpu").numpy()
    if torch.is_tensor(target):
        target = target.detach().to("cpu").numpy()
    return mean_squared_error(output, target)


def accuracy(output, target):
    if torch.is_tensor(output):
        output = output.detach().to("cpu").numpy()
    if torch.is_tensor(target):
        target = target.detach().to("cpu").numpy()
    return accuracy_score(output.argmax(axis=1), target)


def calculate_accuracy(predicted, labels, mask):
    matches = [int(p == l) if m == 1 else 0 for p, l, m in zip(predicted, labels, mask)]
    num_correct = sum(matches)
    accuracy = 100 * num_correct / sum(mask)
    return accuracy


def per_token_accuracy(outputs: List, targets: List, label_id: int):
    output = outputs[label_id]
    target = targets[0][label_id]
    mask = targets[1][label_id]
    if torch.is_tensor(mask):
        mask = mask.detach().to("cpu").numpy()
    seq_len = list(target.sum(axis=2).sum(axis=1).cpu().numpy().astype("int"))
    target_seq = target.argmax(2)
    if torch.is_tensor(target_seq):
        target_seq = target_seq.detach().to("cpu").numpy()
    if torch.is_tensor(output):
        output = output.detach().to("cpu").numpy()
    pred_seq = output.argmax(2)
    acc = [calculate_accuracy(p[:l], t[:l], m[:l]) for (l, p, t, m) in zip(seq_len, pred_seq, target_seq, mask)]
    return torch.tensor(np.mean(acc), device="cuda")
