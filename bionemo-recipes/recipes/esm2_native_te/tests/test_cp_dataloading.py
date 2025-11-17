# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import copy
import unittest
from typing import Dict, Iterator, List

from unittest import mock

import torch

from dataset import CPAwareDataloader
from utils import split_batch_by_cp_rank
from transformer_engine.pytorch.attention.dot_product_attention.context_parallel import pad_thd_sequences_for_cp



def get_dummy_data_thd_with_padding_dp0(cp_size: int):
    pid = 1 # The pad token id.
    label_pad = -100 # The label pad id.

    # Make some fake data.
    input_ids = torch.tensor([
                1, 2, 3, 5, 6
            ])
    labels = torch.tensor([
        10, 20, 30, 50, 60,
    ])
    cu_seqlens_q = torch.tensor([0, 3, 5])
    divisibility_factor = 2 * cp_size

    input_ids_padded, labels_padded, cu_seqlens_q_padded = \
                pad_thd_sequences_for_cp(
                    input_ids.unsqueeze(0),
                    labels.unsqueeze(0),
                    cu_seqlens_q,
                    divisibility_factor,
                    padding_token_id=pid,
                    padding_label_id=label_pad
                )

    batch = {
        "input_ids": input_ids_padded.unsqueeze(0).to(torch.int64), # Add batch dim: [1, seq_len]
        "labels": labels_padded.unsqueeze(0).to(torch.int64), # [1, seq_len]
        "cu_seq_lens_q_padded": cu_seqlens_q_padded.to(torch.int32), # Keep 1D - int32
        "cu_seq_lens_k_padded": cu_seqlens_q_padded.to(torch.int32), # Keep 1D - int32
        "cu_seq_lens_q": cu_seqlens_q.to(torch.int32), # Keep 1D - int32
        "cu_seq_lens_k": cu_seqlens_q.to(torch.int32), # Keep 1D - int32
        "max_length_q": 8,
        "max_length_k": 8,
    }
    return batch


def get_dummy_data_thd_with_padding_dp1(cp_size: int):
    pid = 1 # The pad token id.
    label_pad = -100 # The label pad id.

    # Make some fake data.
    input_ids = torch.tensor([
                9, 10, 11, 13, 14, 15,
            ])
    labels = torch.tensor([
        90, 100, 110, 130, 140, 150,
    ])
    cu_seqlens_q = torch.tensor([0, 3, 6])
    divisibility_factor = 2 * cp_size

    input_ids_padded, labels_padded, cu_seqlens_q_padded = \
                pad_thd_sequences_for_cp(
                    input_ids.unsqueeze(0),
                    labels.unsqueeze(0),
                    cu_seqlens_q,
                    divisibility_factor,
                    padding_token_id=pid,
                    padding_label_id=label_pad
                )

    batch = {
        "input_ids": input_ids_padded.unsqueeze(0).to(torch.int64), # Add batch dim: [1, seq_len]
        "labels": labels_padded.unsqueeze(0).to(torch.int64), # [1, seq_len]
        "cu_seq_lens_q_padded": cu_seqlens_q_padded.to(torch.int32), # Keep 1D - int32
        "cu_seq_lens_k_padded": cu_seqlens_q_padded.to(torch.int32), # Keep 1D - int32
        "cu_seq_lens_q": cu_seqlens_q.to(torch.int32), # Keep 1D - int32
        "cu_seq_lens_k": cu_seqlens_q.to(torch.int32), # Keep 1D - int32
        "max_length_q": 8,
        "max_length_k": 8,
    }
    return batch


def get_dummy_data_thd_dp0_nopadding():
    # Make some fake data.
    input_ids = torch.tensor([
                1, 2, 3, 4, 5, 6, 7, 8,  # 8 tokens
            ])
    labels = torch.tensor([
        10, 20, 30, 40, 50, 60, 70, 80,
    ])
    cu_seqlens_q = torch.tensor([0, 8])
    batch = {
        "input_ids": input_ids.unsqueeze(0).to(torch.int64), # Add batch dim: [1, seq_len]
        "labels": labels.unsqueeze(0).to(torch.int64), # [1, seq_len]
        "cu_seq_lens_q_padded": cu_seqlens_q.to(torch.int32), # Keep 1D - int32
        "cu_seq_lens_k_padded": cu_seqlens_q.to(torch.int32), # Keep 1D - int32
        "cu_seq_lens_q": cu_seqlens_q.to(torch.int32), # Keep 1D - int32
        "cu_seq_lens_k": cu_seqlens_q.to(torch.int32), # Keep 1D - int32
        "max_length_q": 8,
        "max_length_k": 8,
    }
    return batch


def get_dummy_data_thd_dp1_nopadding():
    # Make some fake data.
    input_ids = torch.tensor([
                9, 10, 11, 12, 13, 14, 15, 16,  # 8 tokens
            ])
    labels = torch.tensor([
        90, 100, 110, 120, 130, 140, 150, 160,
    ])
    cu_seqlens_q = torch.tensor([0, 8])
    batch = {
        "input_ids": input_ids.unsqueeze(0).to(torch.int64), # Add batch dim: [1, seq_len]
        "labels": labels.unsqueeze(0).to(torch.int64), # [1, seq_len]
        "cu_seq_lens_q_padded": cu_seqlens_q.to(torch.int32), # Keep 1D - int32
        "cu_seq_lens_k_padded": cu_seqlens_q.to(torch.int32), # Keep 1D - int32
        "cu_seq_lens_q": cu_seqlens_q.to(torch.int32), # Keep 1D - int32
        "cu_seq_lens_k": cu_seqlens_q.to(torch.int32), # Keep 1D - int32
        "max_length_q": 8,
        "max_length_k": 8,
    }
    return batch

class _DummyLoader:
    """Minimal iterable that always yields the same object (batch or list)."""

    def __init__(self, batch):
        self._batch = batch

    def __iter__(self) -> Iterator:
        return self

    def __next__(self):
        return copy.deepcopy(self._batch)


class _DummyCPGroup:
    def __init__(self, size: int):
        self._size = size

    def size(self) -> int:
        return self._size


def _fake_get_batch(
    cu_seqlens_padded,
    input_ids_padded,
    labels_padded,
    cp_group,
    qvk_format,
    cp_rank,
):
    cp_size = cp_group.size()
    total_slices = 2 * cp_size
    seq_tokens = input_ids_padded.view(-1)
    seq_labels = labels_padded.view(-1)
    shard_tokens: List[torch.Tensor] = []
    shard_labels: List[torch.Tensor] = []

    for start, end in zip(cu_seqlens_padded[:-1], cu_seqlens_padded[1:]):
        start_idx = int(start)
        end_idx = int(end)
        slice_size = (end_idx - start_idx) // total_slices

        first_start = start_idx + (cp_rank * slice_size)
        first_end = first_start + slice_size
        second_start = start_idx + ((total_slices - cp_rank - 1) * slice_size)
        second_end = second_start + slice_size

        shard_tokens.append(torch.cat([seq_tokens[first_start:first_end], seq_tokens[second_start:second_end]]))
        shard_labels.append(torch.cat([seq_labels[first_start:first_end], seq_labels[second_start:second_end]]))

    return (
        torch.cat(shard_tokens).unsqueeze(0),
        torch.cat(shard_labels).unsqueeze(0),
    )


def _make_cp_shards(base_batch: Dict[str, torch.Tensor], cp_group: _DummyCPGroup):
    combined_batch = []
    for cp_rank in range(cp_group.size()):
        input_ids_sharded, labels_sharded = _fake_get_batch(
            cu_seqlens_padded=base_batch["cu_seq_lens_q_padded"],
            input_ids_padded=base_batch["input_ids"],
            labels_padded=base_batch["labels"],
            cp_group=cp_group,
            qvk_format="thd",
            cp_rank=cp_rank,
        )
        batch_shard = dict(base_batch)
        batch_shard["input_ids"] = input_ids_sharded
        batch_shard["labels"] = labels_sharded
        combined_batch.append(batch_shard)
    return combined_batch

def test_pad_thd_sequences_for_cp():
    pid = 1 # The pad token id.
    label_pad = -100 # The label pad id.

    # Make some fake data.
    input_ids = torch.tensor([
                1, 1, 1, 1, 1, 1, 1,  # 7 tokens
                2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,  # 11 tokens
                3, 3, 3, 3, 3  # 5 tokens
            ])
    labels = torch.tensor([
        10, 11, 12, 13, 14, 15, 16,
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        5, 6, 7, 8, 9
    ])
    cu_seqlens_q = torch.tensor([0, 7, 18, 23])
    divisibility_factor = 4

    input_ids_padded, labels_padded, cu_seqlens_q_padded = \
                pad_thd_sequences_for_cp(
                    input_ids.unsqueeze(0),
                    labels.unsqueeze(0),
                    cu_seqlens_q,
                    divisibility_factor,
                    padding_token_id=pid,
                    padding_label_id=label_pad
                )
    expected_input_ids = torch.tensor([
                1, 1, 1, 1, 1, 1, 1, pid,
                2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, pid,
                3, 3, 3, 3, 3, pid, pid, pid
            ])

    expected_labels = torch.tensor([
        10, 11, 12, 13, 14, 15, 16, label_pad,
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, label_pad,
        5, 6, 7, 8, 9, label_pad, label_pad, label_pad
    ])

    expected_cu_seqlens_padded = torch.tensor([0, 8, 20, 28])

    assert torch.equal(input_ids_padded, expected_input_ids)
    assert torch.equal(labels_padded, expected_labels)
    assert torch.equal(cu_seqlens_q_padded, expected_cu_seqlens_padded)



def test_dataloader_scatter_nopadding():
    """
    Test single sequence on two dataloader ranks with no additional padding, with CP=2, DP=2, ensure that the data is scattered correctly.
    There are going to be 4 shards. CP0, CP1 (for context parallel) and DP0, DP1 (for data parallel).
    DP0 will return [1,2,3,4,5,6,7,8] and DP1 will return [9,10,11,12,13,14,15,16].
    CP0 will receive [1,2,7,8] and CP1 will receive [3,4,5,6]. (from DP0)
    CP1 will receive [9,10,15,16] and CP0 will receive [11,12,13,14]. (from DP1)

        |   DP0   |    DP1        |
    CP0 | 1,2,7,8 | 9, 10, 15, 16 |
    CP1 | 3,4,5,6 | 11, 12, 13, 14|
    """
    cp_group = _DummyCPGroup(size=2)

    def run_roundtrip(base_batch):
        combined_batch = [
            dict(
                base_batch,
                **{
                    "input_ids": split_batch_by_cp_rank(
                        cu_seqlens_padded=base_batch["cu_seq_lens_q_padded"],
                        input_ids_padded=base_batch["input_ids"],
                        labels_padded=base_batch["labels"],
                        qvk_format="thd",
                        cp_rank=cp_rank,
                        cp_world_size=cp_group.size(),
                    )[0],
                    "labels": split_batch_by_cp_rank(
                        cu_seqlens_padded=base_batch["cu_seq_lens_q_padded"],
                        input_ids_padded=base_batch["input_ids"],
                        labels_padded=base_batch["labels"],
                        qvk_format="thd",
                        cp_rank=cp_rank,
                        cp_world_size=cp_group.size(),
                    )[1],
                },
            )
            for cp_rank in range(cp_group.size())
        ]
        loader_rank0 = CPAwareDataloader(_DummyLoader(combined_batch), cp_group, cp_rank=0)
        loader_rank1 = CPAwareDataloader(_DummyLoader(combined_batch), cp_group, cp_rank=1)

        scatter_payload: Dict[str, List[Dict[str, torch.Tensor]]] = {}
        current_rank = {"value": None}

        def fake_scatter(
            *,
            scatter_object_output_list,
            scatter_object_input_list,
            group,
            group_src,
        ):
            if scatter_object_input_list is not None:
                scatter_payload["data"] = scatter_object_input_list
            assert "data" in scatter_payload, "Rank 0 payload missing"
            scatter_object_output_list[0] = scatter_payload["data"][current_rank["value"]]

        with mock.patch("dataset.torch.distributed.scatter_object_list", side_effect=fake_scatter), \
             mock.patch("dataset.torch.distributed.barrier", return_value=None):
            iter(loader_rank0)
            iter(loader_rank1)

            current_rank["value"] = 0
            batch_cp0 = next(loader_rank0)

            current_rank["value"] = 1
            batch_cp1 = next(loader_rank1)

        return batch_cp0, batch_cp1

    batch_dp0_cp0, batch_dp0_cp1 = run_roundtrip(get_dummy_data_thd_dp0_nopadding())
    torch.testing.assert_close(batch_dp0_cp0["input_ids"], torch.tensor([[1, 2, 7, 8]], dtype=torch.int64))
    torch.testing.assert_close(batch_dp0_cp0["labels"], torch.tensor([[10, 20, 70, 80]], dtype=torch.int64))
    torch.testing.assert_close(batch_dp0_cp1["input_ids"], torch.tensor([[3, 4, 5, 6]], dtype=torch.int64))
    torch.testing.assert_close(batch_dp0_cp1["labels"], torch.tensor([[30, 40, 50, 60]], dtype=torch.int64))

    batch_dp1_cp0, batch_dp1_cp1 = run_roundtrip(get_dummy_data_thd_dp1_nopadding())

    torch.testing.assert_close(batch_dp1_cp0["input_ids"], torch.tensor([[9, 10, 15 ,16]], dtype=torch.int64))
    torch.testing.assert_close(batch_dp1_cp0["labels"], torch.tensor([[90, 100, 150, 160]], dtype=torch.int64))
    torch.testing.assert_close(batch_dp1_cp1["input_ids"], torch.tensor([[11, 12, 13, 14]], dtype=torch.int64))
    torch.testing.assert_close(batch_dp1_cp1["labels"], torch.tensor([[110, 120, 130, 140]], dtype=torch.int64))

def test_dataloader_scatter_with_pad_between_seqs():
    f"""
    Here we are going to test two sequences using two dataloaders with padding. We use CP=2, DP=2 and
    ensure that the data is scattered correctly.
    There are going to be 4 shards. CP0, CP1 (for context parallel) and DP0, DP1 (for data parallel).
    DP0 will return [1,2,3,<p> | 5,6,<p>,<p>]
    DP1 will return [9,10,11,<p> | 13,14,15,<p>]

    We notice that CP sharding grabs slices of each sequence. Thus, CP0 grabs the first and last slices, while CP1 grabs the middle slices.
        |   DP0       |       DP1      |
    CP0 | 1,<p>,5,<p> | 9, <p>, 13, <p>|
    CP1 | 2,3,6, <p>  | 10, 11, 14, 15 |
    """
    cp_group = _DummyCPGroup(size=2)

    def run_roundtrip(base_batch):
        combined_batch = [
            dict(
                base_batch,
                **{
                    "input_ids": split_batch_by_cp_rank(
                        cu_seqlens_padded=base_batch["cu_seq_lens_q_padded"],
                        input_ids_padded=base_batch["input_ids"],
                        labels_padded=base_batch["labels"],
                        qvk_format="thd",
                        cp_rank=cp_rank,
                        cp_world_size=cp_group.size(),
                    )[0],
                    "labels": split_batch_by_cp_rank(
                        cu_seqlens_padded=base_batch["cu_seq_lens_q_padded"],
                        input_ids_padded=base_batch["input_ids"],
                        labels_padded=base_batch["labels"],
                        qvk_format="thd",
                        cp_rank=cp_rank,
                        cp_world_size=cp_group.size(),
                    )[1],
                },
            )
            for cp_rank in range(cp_group.size())
        ]
        loader_rank0 = CPAwareDataloader(_DummyLoader(combined_batch), cp_group, cp_rank=0)
        loader_rank1 = CPAwareDataloader(_DummyLoader(combined_batch), cp_group, cp_rank=1)

        scatter_payload: Dict[str, List[Dict[str, torch.Tensor]]] = {}
        current_rank = {"value": None}

        def fake_scatter(
            *,
            scatter_object_output_list,
            scatter_object_input_list,
            group,
            group_src,
        ):
            if scatter_object_input_list is not None:
                scatter_payload["data"] = scatter_object_input_list
            assert "data" in scatter_payload, "Rank 0 payload missing"
            scatter_object_output_list[0] = scatter_payload["data"][current_rank["value"]]

        with mock.patch("dataset.torch.distributed.scatter_object_list", side_effect=fake_scatter), \
             mock.patch("dataset.torch.distributed.barrier", return_value=None):
            iter(loader_rank0)
            iter(loader_rank1)

            current_rank["value"] = 0
            batch_cp0 = next(loader_rank0)

            current_rank["value"] = 1
            batch_cp1 = next(loader_rank1)

        return batch_cp0, batch_cp1

    batch_dp0_cp0, batch_dp0_cp1 = run_roundtrip(get_dummy_data_thd_with_padding_dp0(cp_size=2))

    torch.testing.assert_close(batch_dp0_cp0["input_ids"], torch.tensor([[1, 1, 5, 1]], dtype=torch.int64))
    torch.testing.assert_close(batch_dp0_cp1["input_ids"], torch.tensor([[2, 3, 6, 1]], dtype=torch.int64))

    batch_dp1_cp0, batch_dp1_cp1 = run_roundtrip(get_dummy_data_thd_with_padding_dp1(cp_size=2))

    torch.testing.assert_close(batch_dp1_cp0["input_ids"], torch.tensor([[9, 1, 13, 1]], dtype=torch.int64))
    torch.testing.assert_close(batch_dp1_cp1["input_ids"], torch.tensor([[10, 11, 14, 15]], dtype=torch.int64))

if __name__ == "__main__":
    unittest.main()
