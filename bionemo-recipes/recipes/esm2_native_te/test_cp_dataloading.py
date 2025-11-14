import copy
import unittest
from typing import Dict, Iterator, List

from unittest import mock

import torch

from dataset import CPAwareDataloader
from utils import get_dummy_data_thd_dp0_nopadding, get_dummy_data_thd_dp1_nopadding

class _DummyLoader:
    """Minimal iterable that always yields the same batch."""

    def __init__(self, batch: Dict[str, torch.Tensor]):
        self._batch = batch

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        return self

    def __next__(self) -> Dict[str, torch.Tensor]:
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


def test_dataloader_scatter_nopadding():
    """
    Test 1. Using no additional padding, with CP=2, DP=2, ensure that the data is scattered correctly.
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
        loader_rank0 = CPAwareDataloader(_DummyLoader(base_batch), cp_group, cp_rank=0)
        loader_rank1 = CPAwareDataloader(_DummyLoader(base_batch), cp_group, cp_rank=1)

        scatter_payload: List[Dict[str, torch.Tensor]] | None = None
        current_rank = {"value": None}

        def fake_scatter(
            *,
            scatter_object_output_list,
            scatter_object_input_list,
            group,
            group_src,
        ):
            nonlocal scatter_payload
            if scatter_object_input_list is not None:
                scatter_payload = scatter_object_input_list
            assert scatter_payload is not None
            scatter_object_output_list[0] = scatter_payload[current_rank["value"]]

        with mock.patch("dataset.get_batch_on_this_cp_rank", side_effect=_fake_get_batch) as mock_get_batch, \
             mock.patch("dataset.torch.distributed.scatter_object_list", side_effect=fake_scatter), \
             mock.patch("dataset.torch.distributed.barrier", return_value=None):
            iter(loader_rank0)
            iter(loader_rank1)

            current_rank["value"] = 0
            batch_cp0 = next(loader_rank0)

            current_rank["value"] = 1
            batch_cp1 = next(loader_rank1)

        return batch_cp0, batch_cp1, mock_get_batch

    batch_dp0_cp0, batch_dp0_cp1, mock_dp0 = run_roundtrip(get_dummy_data_thd_dp0_nopadding())

    torch.testing.assert_close(batch_dp0_cp0["input_ids"], torch.tensor([[1, 2, 7, 8]], dtype=torch.int64))
    torch.testing.assert_close(batch_dp0_cp0["labels"], torch.tensor([[10, 20, 70, 80]], dtype=torch.int64))
    torch.testing.assert_close(batch_dp0_cp1["input_ids"], torch.tensor([[3, 4, 5, 6]], dtype=torch.int64))
    torch.testing.assert_close(batch_dp0_cp1["labels"], torch.tensor([[30, 40, 50, 60]], dtype=torch.int64))

    batch_dp1_cp0, batch_dp1_cp1, _ = run_roundtrip(get_dummy_data_thd_dp1_nopadding())

    torch.testing.assert_close(batch_dp1_cp0["input_ids"], torch.tensor([[9, 10, 15 ,16]], dtype=torch.int64))
    torch.testing.assert_close(batch_dp1_cp0["labels"], torch.tensor([[90, 100, 150, 160]], dtype=torch.int64))
    torch.testing.assert_close(batch_dp1_cp1["input_ids"], torch.tensor([[11, 12, 13, 14]], dtype=torch.int64))
    torch.testing.assert_close(batch_dp1_cp1["labels"], torch.tensor([[110, 120, 130, 140]], dtype=torch.int64))

    called_ranks = [kwargs["cp_rank"] for _, kwargs in mock_dp0.call_args_list]
    assert called_ranks == [0, 1]

if __name__ == "__main__":
    unittest.main()
