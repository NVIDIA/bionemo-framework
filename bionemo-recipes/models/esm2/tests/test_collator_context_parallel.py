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
from typing import Dict, Iterator, List
from unittest import mock

import torch
from transformer_engine.pytorch.attention.dot_product_attention.context_parallel import pad_thd_sequences_for_cp

from esm.collator import ContextParallelDataLoaderWrapper, _split_batch_by_cp_rank


def get_dummy_data_thd_with_padding_dp0(cp_size: int):
    pid = 1  # The pad token id.
    label_pad = -100  # The label pad id.

    # Make some fake data.
    input_ids = torch.tensor([1, 2, 3, 5, 6])
    labels = torch.tensor(
        [
            10,
            20,
            30,
            50,
            60,
        ]
    )
    cu_seqlens_q = torch.tensor([0, 3, 5])
    divisibility_factor = 2 * cp_size

    input_ids_padded, labels_padded, cu_seqlens_q_padded = pad_thd_sequences_for_cp(
        input_ids.unsqueeze(0),
        labels.unsqueeze(0),
        cu_seqlens_q,
        divisibility_factor,
        padding_token_id=pid,
        padding_label_id=label_pad,
    )

    batch = {
        "input_ids": input_ids_padded.unsqueeze(0).to(torch.int64),  # Add batch dim: [1, seq_len]
        "labels": labels_padded.unsqueeze(0).to(torch.int64),  # [1, seq_len]
        "cu_seq_lens_q_padded": cu_seqlens_q_padded.to(torch.int32),  # Keep 1D - int32
        "cu_seq_lens_k_padded": cu_seqlens_q_padded.to(torch.int32),  # Keep 1D - int32
        "cu_seq_lens_q": cu_seqlens_q.to(torch.int32),  # Keep 1D - int32
        "cu_seq_lens_k": cu_seqlens_q.to(torch.int32),  # Keep 1D - int32
        "max_length_q": 8,
        "max_length_k": 8,
    }
    return batch


def get_dummy_data_thd_with_padding_dp1(cp_size: int):
    pid = 1  # The pad token id.
    label_pad = -100  # The label pad id.

    # Make some fake data.
    input_ids = torch.tensor(
        [
            9,
            10,
            11,
            13,
            14,
            15,
        ]
    )
    labels = torch.tensor(
        [
            90,
            100,
            110,
            130,
            140,
            150,
        ]
    )
    cu_seqlens_q = torch.tensor([0, 3, 6])
    divisibility_factor = 2 * cp_size

    input_ids_padded, labels_padded, cu_seqlens_q_padded = pad_thd_sequences_for_cp(
        input_ids.unsqueeze(0),
        labels.unsqueeze(0),
        cu_seqlens_q,
        divisibility_factor,
        padding_token_id=pid,
        padding_label_id=label_pad,
    )

    batch = {
        "input_ids": input_ids_padded.unsqueeze(0).to(torch.int64),  # Add batch dim: [1, seq_len]
        "labels": labels_padded.unsqueeze(0).to(torch.int64),  # [1, seq_len]
        "cu_seq_lens_q_padded": cu_seqlens_q_padded.to(torch.int32),  # Keep 1D - int32
        "cu_seq_lens_k_padded": cu_seqlens_q_padded.to(torch.int32),  # Keep 1D - int32
        "cu_seq_lens_q": cu_seqlens_q.to(torch.int32),  # Keep 1D - int32
        "cu_seq_lens_k": cu_seqlens_q.to(torch.int32),  # Keep 1D - int32
        "max_length_q": 8,
        "max_length_k": 8,
    }
    return batch


def get_dummy_data_thd_dp0_nopadding():
    # Make some fake data.
    input_ids = torch.tensor(
        [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,  # 8 tokens
        ]
    )
    labels = torch.tensor(
        [
            10,
            20,
            30,
            40,
            50,
            60,
            70,
            80,
        ]
    )
    cu_seqlens_q = torch.tensor([0, 8])
    batch = {
        "input_ids": input_ids.unsqueeze(0).to(torch.int64),  # Add batch dim: [1, seq_len]
        "labels": labels.unsqueeze(0).to(torch.int64),  # [1, seq_len]
        "cu_seq_lens_q_padded": cu_seqlens_q.to(torch.int32),  # Keep 1D - int32
        "cu_seq_lens_k_padded": cu_seqlens_q.to(torch.int32),  # Keep 1D - int32
        "cu_seq_lens_q": cu_seqlens_q.to(torch.int32),  # Keep 1D - int32
        "cu_seq_lens_k": cu_seqlens_q.to(torch.int32),  # Keep 1D - int32
        "max_length_q": 8,
        "max_length_k": 8,
    }
    return batch


def get_dummy_data_thd_dp1_nopadding():
    # Make some fake data.
    input_ids = torch.tensor(
        [
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,  # 8 tokens
        ]
    )
    labels = torch.tensor(
        [
            90,
            100,
            110,
            120,
            130,
            140,
            150,
            160,
        ]
    )
    cu_seqlens_q = torch.tensor([0, 8])
    batch = {
        "input_ids": input_ids.unsqueeze(0).to(torch.int64),  # Add batch dim: [1, seq_len]
        "labels": labels.unsqueeze(0).to(torch.int64),  # [1, seq_len]
        "cu_seq_lens_q_padded": cu_seqlens_q.to(torch.int32),  # Keep 1D - int32
        "cu_seq_lens_k_padded": cu_seqlens_q.to(torch.int32),  # Keep 1D - int32
        "cu_seq_lens_q": cu_seqlens_q.to(torch.int32),  # Keep 1D - int32
        "cu_seq_lens_k": cu_seqlens_q.to(torch.int32),  # Keep 1D - int32
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


class _DummyDeviceMesh:
    """Dummy device mesh for testing ContextParallelDataLoaderWrapper."""

    def __init__(self, size: int, rank: int = 0):
        self._size = size
        self._rank = rank
        self._group = mock.MagicMock()  # Mock process group

    def get_local_rank(self) -> int:
        """Return the local rank within this mesh."""
        return self._rank

    def get_group(self):
        """Return the process group."""
        return self._group

    def size(self) -> int:
        """Return the size of the mesh."""
        return self._size


def test_pad_thd_sequences_for_cp():
    pid = 1  # The pad token id.
    label_pad = -100  # The label pad id.

    # Make some fake data.
    input_ids = torch.tensor(
        [
            1,
            1,
            1,
            1,
            1,
            1,
            1,  # 7 tokens
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,  # 11 tokens
            3,
            3,
            3,
            3,
            3,  # 5 tokens
        ]
    )
    labels = torch.tensor([10, 11, 12, 13, 14, 15, 16, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 5, 6, 7, 8, 9])
    cu_seqlens_q = torch.tensor([0, 7, 18, 23])
    divisibility_factor = 4

    input_ids_padded, labels_padded, cu_seqlens_q_padded = pad_thd_sequences_for_cp(
        input_ids.unsqueeze(0),
        labels.unsqueeze(0),
        cu_seqlens_q,
        divisibility_factor,
        padding_token_id=pid,
        padding_label_id=label_pad,
    )
    expected_input_ids = torch.tensor(
        [1, 1, 1, 1, 1, 1, 1, pid, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, pid, 3, 3, 3, 3, 3, pid, pid, pid]
    )

    expected_labels = torch.tensor(
        [
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            label_pad,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            label_pad,
            5,
            6,
            7,
            8,
            9,
            label_pad,
            label_pad,
            label_pad,
        ]
    )

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
    cp_size = 2

    def run_roundtrip(base_batch):
        combined_batch = [
            dict(
                base_batch,
                **{
                    "input_ids": _split_batch_by_cp_rank(
                        cu_seqlens_padded=base_batch["cu_seq_lens_q_padded"],
                        input_ids_padded=base_batch["input_ids"],
                        labels_padded=base_batch["labels"],
                        qvk_format="thd",
                        cp_rank=cp_rank,
                        cp_world_size=cp_size,
                    )[0],
                    "labels": _split_batch_by_cp_rank(
                        cu_seqlens_padded=base_batch["cu_seq_lens_q_padded"],
                        input_ids_padded=base_batch["input_ids"],
                        labels_padded=base_batch["labels"],
                        qvk_format="thd",
                        cp_rank=cp_rank,
                        cp_world_size=cp_size,
                    )[1],
                },
            )
            for cp_rank in range(cp_size)
        ]
        cp_mesh_rank0 = _DummyDeviceMesh(size=cp_size, rank=0)
        cp_mesh_rank1 = _DummyDeviceMesh(size=cp_size, rank=1)
        loader_rank0 = ContextParallelDataLoaderWrapper(_DummyLoader(combined_batch), cp_mesh_rank0)
        loader_rank1 = ContextParallelDataLoaderWrapper(None, cp_mesh_rank1)

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

        with (
            mock.patch("esm.collator.torch.distributed.scatter_object_list", side_effect=fake_scatter),
            mock.patch("esm.collator.torch.distributed.barrier", return_value=None),
        ):
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

    torch.testing.assert_close(batch_dp1_cp0["input_ids"], torch.tensor([[9, 10, 15, 16]], dtype=torch.int64))
    torch.testing.assert_close(batch_dp1_cp0["labels"], torch.tensor([[90, 100, 150, 160]], dtype=torch.int64))
    torch.testing.assert_close(batch_dp1_cp1["input_ids"], torch.tensor([[11, 12, 13, 14]], dtype=torch.int64))
    torch.testing.assert_close(batch_dp1_cp1["labels"], torch.tensor([[110, 120, 130, 140]], dtype=torch.int64))


def test_dataloader_scatter_with_pad_between_seqs():
    """
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
    cp_size = 2

    def run_roundtrip(base_batch):
        combined_batch = [
            dict(
                base_batch,
                **{
                    "input_ids": _split_batch_by_cp_rank(
                        cu_seqlens_padded=base_batch["cu_seq_lens_q_padded"],
                        input_ids_padded=base_batch["input_ids"],
                        labels_padded=base_batch["labels"],
                        qvk_format="thd",
                        cp_rank=cp_rank,
                        cp_world_size=cp_size,
                    )[0],
                    "labels": _split_batch_by_cp_rank(
                        cu_seqlens_padded=base_batch["cu_seq_lens_q_padded"],
                        input_ids_padded=base_batch["input_ids"],
                        labels_padded=base_batch["labels"],
                        qvk_format="thd",
                        cp_rank=cp_rank,
                        cp_world_size=cp_size,
                    )[1],
                },
            )
            for cp_rank in range(cp_size)
        ]
        cp_mesh_rank0 = _DummyDeviceMesh(size=cp_size, rank=0)
        cp_mesh_rank1 = _DummyDeviceMesh(size=cp_size, rank=1)
        loader_rank0 = ContextParallelDataLoaderWrapper(_DummyLoader(combined_batch), cp_mesh_rank0)
        loader_rank1 = ContextParallelDataLoaderWrapper(None, cp_mesh_rank1)

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

        with (
            mock.patch("esm.collator.torch.distributed.scatter_object_list", side_effect=fake_scatter),
            mock.patch("esm.collator.torch.distributed.barrier", return_value=None),
        ):
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


def get_dummy_data_bshd_single_sequence(cp_size: int, seq_len: int = 8):
    """Create dummy BSHD format data with a single sequence.

    Args:
        cp_size: The size of the context parallelism group.
        seq_len: The sequence length (must be divisible by 2*cp_size).

    Returns:
        A dictionary containing input_ids and labels in BSHD format.
    """
    if seq_len % (2 * cp_size) != 0:
        raise ValueError(f"seq_len ({seq_len}) must be divisible by 2*cp_size ({2 * cp_size})")

    # Create a simple sequence: [1, 2, 3, ..., seq_len]
    input_ids = torch.arange(1, seq_len + 1, dtype=torch.int64).unsqueeze(0)  # [1, seq_len]
    labels = torch.arange(10, 10 + seq_len, dtype=torch.int64).unsqueeze(0)  # [1, seq_len]

    return {
        "input_ids": input_ids,
        "labels": labels,
    }


def get_dummy_data_bshd_multiple_sequences(cp_size: int, batch_size: int = 2, seq_len: int = 8):
    """Create dummy BSHD format data with multiple sequences.

    Args:
        cp_size: The size of the context parallelism group.
        batch_size: The batch size.
        seq_len: The sequence length (must be divisible by 2*cp_size).

    Returns:
        A dictionary containing input_ids and labels in BSHD format.
    """
    if seq_len % (2 * cp_size) != 0:
        raise ValueError(f"seq_len ({seq_len}) must be divisible by 2*cp_size ({2 * cp_size})")

    # Create sequences: each sequence starts at a different offset
    input_ids_list = []
    labels_list = []
    for i in range(batch_size):
        seq_input_ids = torch.arange(i * 100 + 1, i * 100 + seq_len + 1, dtype=torch.int64)
        seq_labels = torch.arange(i * 1000 + 10, i * 1000 + seq_len + 10, dtype=torch.int64)
        input_ids_list.append(seq_input_ids)
        labels_list.append(seq_labels)

    input_ids = torch.stack(input_ids_list)  # [batch_size, seq_len]
    labels = torch.stack(labels_list)  # [batch_size, seq_len]

    return {
        "input_ids": input_ids,
        "labels": labels,
    }


def test_split_batch_by_cp_rank_bshd_single_sequence():
    """Test BSHD format splitting for a single sequence with CP=2.

    For a sequence of length 8 with CP=2:
    - Total chunks = 2 * 2 = 4
    - Chunk size = 8 / 4 = 2
    - CP rank 0 gets chunks [0, 3]: indices [0:2] and [6:8] -> [1,2,7,8]
    - CP rank 1 gets chunks [1, 2]: indices [2:4] and [4:6] -> [3,4,5,6]
    """
    cp_size = 2
    seq_len = 8
    batch = get_dummy_data_bshd_single_sequence(cp_size=cp_size, seq_len=seq_len)

    # Test CP rank 0
    input_ids_cp0, labels_cp0 = _split_batch_by_cp_rank(
        cu_seqlens_padded=None,
        input_ids_padded=batch["input_ids"],
        labels_padded=batch["labels"],
        qvk_format="bshd",
        cp_rank=0,
        cp_world_size=cp_size,
    )

    # CP rank 0 should get chunks [0, 3]: [1,2] and [7,8]
    expected_input_ids_cp0 = torch.tensor([[1, 2, 7, 8]], dtype=torch.int64)
    expected_labels_cp0 = torch.tensor([[10, 11, 16, 17]], dtype=torch.int64)

    torch.testing.assert_close(input_ids_cp0, expected_input_ids_cp0)
    torch.testing.assert_close(labels_cp0, expected_labels_cp0)

    # Test CP rank 1
    input_ids_cp1, labels_cp1 = _split_batch_by_cp_rank(
        cu_seqlens_padded=None,
        input_ids_padded=batch["input_ids"],
        labels_padded=batch["labels"],
        qvk_format="bshd",
        cp_rank=1,
        cp_world_size=cp_size,
    )

    # CP rank 1 should get chunks [1, 2]: [3,4] and [5,6]
    expected_input_ids_cp1 = torch.tensor([[3, 4, 5, 6]], dtype=torch.int64)
    expected_labels_cp1 = torch.tensor([[12, 13, 14, 15]], dtype=torch.int64)

    torch.testing.assert_close(input_ids_cp1, expected_input_ids_cp1)
    torch.testing.assert_close(labels_cp1, expected_labels_cp1)


def test_split_batch_by_cp_rank_bshd_multiple_sequences():
    """Test BSHD format splitting for multiple sequences with CP=2.

    For batch_size=2, seq_len=8 with CP=2:
    - Each sequence is split independently
    - Sequence 0: [1,2,3,4,5,6,7,8] (i=0, starts at 0*100+1=1)
    - Sequence 1: [101,102,103,104,105,106,107,108] (i=1, starts at 1*100+1=101)
    - CP rank 0 gets chunks [0, 3] from each sequence
    - CP rank 1 gets chunks [1, 2] from each sequence
    """
    cp_size = 2
    batch_size = 2
    seq_len = 8
    batch = get_dummy_data_bshd_multiple_sequences(cp_size=cp_size, batch_size=batch_size, seq_len=seq_len)

    # Test CP rank 0
    input_ids_cp0, labels_cp0 = _split_batch_by_cp_rank(
        cu_seqlens_padded=None,
        input_ids_padded=batch["input_ids"],
        labels_padded=batch["labels"],
        qvk_format="bshd",
        cp_rank=0,
        cp_world_size=cp_size,
    )

    # CP rank 0 should get chunks [0, 3] from each sequence
    # Sequence 0: [1,2] and [7,8] -> [1,2,7,8]
    # Sequence 1: [101,102] and [107,108] -> [101,102,107,108]
    expected_input_ids_cp0 = torch.tensor([[1, 2, 7, 8], [101, 102, 107, 108]], dtype=torch.int64)
    expected_labels_cp0 = torch.tensor([[10, 11, 16, 17], [1010, 1011, 1016, 1017]], dtype=torch.int64)

    torch.testing.assert_close(input_ids_cp0, expected_input_ids_cp0)
    torch.testing.assert_close(labels_cp0, expected_labels_cp0)

    # Test CP rank 1
    input_ids_cp1, labels_cp1 = _split_batch_by_cp_rank(
        cu_seqlens_padded=None,
        input_ids_padded=batch["input_ids"],
        labels_padded=batch["labels"],
        qvk_format="bshd",
        cp_rank=1,
        cp_world_size=cp_size,
    )

    # CP rank 1 should get chunks [1, 2] from each sequence
    # Sequence 0: [3,4] and [5,6] -> [3,4,5,6]
    # Sequence 1: [103,104] and [105,106] -> [103,104,105,106]
    expected_input_ids_cp1 = torch.tensor([[3, 4, 5, 6], [103, 104, 105, 106]], dtype=torch.int64)
    expected_labels_cp1 = torch.tensor([[12, 13, 14, 15], [1012, 1013, 1014, 1015]], dtype=torch.int64)

    torch.testing.assert_close(input_ids_cp1, expected_input_ids_cp1)
    torch.testing.assert_close(labels_cp1, expected_labels_cp1)


def test_split_batch_by_cp_rank_bshd_cp4():
    """Test BSHD format splitting with CP=4.

    For a sequence of length 16 with CP=4:
    - Total chunks = 2 * 4 = 8
    - Chunk size = 16 / 8 = 2
    - CP rank 0 gets chunks [0, 7]: [1,2] and [15,16]
    - CP rank 1 gets chunks [1, 6]: [3,4] and [13,14]
    - CP rank 2 gets chunks [2, 5]: [5,6] and [11,12]
    - CP rank 3 gets chunks [3, 4]: [7,8] and [9,10]
    """
    cp_size = 4
    seq_len = 16
    batch = get_dummy_data_bshd_single_sequence(cp_size=cp_size, seq_len=seq_len)

    # Test each CP rank
    for cp_rank in range(cp_size):
        input_ids_shard, labels_shard = _split_batch_by_cp_rank(
            cu_seqlens_padded=None,
            input_ids_padded=batch["input_ids"],
            labels_padded=batch["labels"],
            qvk_format="bshd",
            cp_rank=cp_rank,
            cp_world_size=cp_size,
        )

        # Verify shape: should be [1, 4] (batch_size=1, 2 chunks * chunk_size=2)
        assert input_ids_shard.shape == (1, 4), (
            f"CP rank {cp_rank}: expected shape (1, 4), got {input_ids_shard.shape}"
        )
        assert labels_shard.shape == (1, 4), f"CP rank {cp_rank}: expected shape (1, 4), got {labels_shard.shape}"

        # Verify that all values are unique (no duplicates)
        unique_values = torch.unique(input_ids_shard)
        assert len(unique_values) == 4, f"CP rank {cp_rank}: expected 4 unique values, got {len(unique_values)}"

    # Verify that all ranks together reconstruct the original sequence
    all_shards = []
    for cp_rank in range(cp_size):
        input_ids_shard, _ = _split_batch_by_cp_rank(
            cu_seqlens_padded=None,
            input_ids_padded=batch["input_ids"],
            labels_padded=batch["labels"],
            qvk_format="bshd",
            cp_rank=cp_rank,
            cp_world_size=cp_size,
        )
        all_shards.append(input_ids_shard.squeeze(0))

    # Concatenate all shards
    reconstructed = torch.cat(all_shards)
    # Sort to compare with original (chunks are interleaved)
    reconstructed_sorted = torch.sort(reconstructed)[0]
    original_sorted = torch.sort(batch["input_ids"].squeeze(0))[0]

    torch.testing.assert_close(reconstructed_sorted, original_sorted)


def test_split_batch_by_cp_rank_bshd_3d_tensor():
    """Test BSHD format splitting for 3D tensors (e.g., [batch, seq_len, hidden_dim]).

    This tests that the function works correctly for tensors with more than 2 dimensions.
    """
    cp_size = 2
    batch_size = 2
    seq_len = 8
    hidden_dim = 128

    # Create 3D tensors
    input_ids = torch.randn(batch_size, seq_len, hidden_dim)
    labels = torch.randn(batch_size, seq_len, hidden_dim)

    # Test CP rank 0
    input_ids_cp0, labels_cp0 = _split_batch_by_cp_rank(
        cu_seqlens_padded=None,
        input_ids_padded=input_ids,
        labels_padded=labels,
        qvk_format="bshd",
        cp_rank=0,
        cp_world_size=cp_size,
    )

    # Should split along seq_len dimension (dim=1)
    # CP rank 0 gets chunks [0, 3]: indices [0:2] and [6:8]
    expected_shape = (batch_size, 4, hidden_dim)  # 2 chunks * chunk_size=2
    assert input_ids_cp0.shape == expected_shape, f"Expected shape {expected_shape}, got {input_ids_cp0.shape}"
    assert labels_cp0.shape == expected_shape, f"Expected shape {expected_shape}, got {labels_cp0.shape}"

    # Verify the chunks are correct by checking indices
    # First chunk should be original[:, 0:2, :]
    torch.testing.assert_close(input_ids_cp0[:, 0:2, :], input_ids[:, 0:2, :])
    # Second chunk should be original[:, 6:8, :]
    torch.testing.assert_close(input_ids_cp0[:, 2:4, :], input_ids[:, 6:8, :])

    # Test CP rank 1
    input_ids_cp1, labels_cp1 = _split_batch_by_cp_rank(
        cu_seqlens_padded=None,
        input_ids_padded=input_ids,
        labels_padded=labels,
        qvk_format="bshd",
        cp_rank=1,
        cp_world_size=cp_size,
    )

    assert input_ids_cp1.shape == expected_shape
    # CP rank 1 gets chunks [1, 2]: indices [2:4] and [4:6]
    torch.testing.assert_close(input_ids_cp1[:, 0:2, :], input_ids[:, 2:4, :])
    torch.testing.assert_close(input_ids_cp1[:, 2:4, :], input_ids[:, 4:6, :])
    torch.testing.assert_close(labels_cp1[:, 0:2, :], labels[:, 2:4, :])
    torch.testing.assert_close(labels_cp1[:, 2:4, :], labels[:, 4:6, :])
