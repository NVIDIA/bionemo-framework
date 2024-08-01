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


from typing import Iterable, List, Optional, Sequence, Tuple, TypeVar, Union

import pytorch_lightning as pl
import torch
import torch.distributed
from megatron.core import parallel_state
from nemo.lightning.megatron_parallel import DataT, MegatronLossReduction, ReductionT


__all__: Sequence[str] = (
    "get_dtype_device",
    "batch_collator",
    "PassthroughLossReduction",
    "LightningPassthroughPredictionMixin",
    "LossLoggingCallback",
)


T = TypeVar("T")


def some_first(seq: Iterable[Optional[T]]) -> T:
    """Returns the first non-None value from the sequence or fails"""  # noqa: D415
    for s in seq:
        if s is not None:
            return s
    raise ValueError("non-None value not found")


def get_dtype_device(torch_object) -> Tuple[torch.dtype, torch.device]:  # noqa: D103
    match torch_object:
        case []:
            raise ValueError("Looking up dtype on an empty list")
        case {**data} if not data:
            raise ValueError("Looking up dtype on an empty dict")
        case torch.Tensor(dtype=dtype, device=device):
            return dtype, device
        case torch.nn.Module() as m:
            try:
                p = next(m.parameters())
            except StopIteration as e:
                raise ValueError("Cannot get dtype on a torch module with no parameters.") from e
            return p.dtype, p.device
        case dict(keys=_, values=values):
            val = some_first(values())
            return get_dtype_device(val)
        case list() as l:
            val = some_first(l)
            return get_dtype_device(val)
        case _:
            raise TypeError("Got something we didnt expect")


# NOTE(SKH): These types are all wrong, but are close. The inner type must always be a torch.Tensor, but the outer container should be generic.
def batch_collator(batches: Optional[Union[Tuple[ReductionT], List[ReductionT]]]) -> Optional[ReductionT]:
    """Takes a sequence of batches and collates them into a single batch.
        This is distinct from the standard pytorch default_collator since it does
        not add the batch dimension, it's assumed the batch
        dimension is already present in the input, as would be the case when
        parallelizing across minibatches.

    IMPORTANT: The underlying data primitive _must_ be a torch Tensor. The input to this function is a recurisve type,
    there can be any amount of nesting between dictionaries, tuples, and lists, as long as the inner type is a n-d torch.Tensor.

    Examples:
        Outer container = Dict:
            [{'a': torch.tensor([1]), 'b': torch.tensor([2])}, {'a': torch.tensor([2]), 'b': torch.tensor([3])}] -> {'a': torch.tensor([1, 2]), 'b': torch.tensor([2, 3])}
        Outer container = List:
            [[torch.tensor([1]), torch.tensor([2])], [torch.tensor([2]), torch.tensor([3])]] -> [torch.tensor([1, 2]), torch.tensor([2, 3])]
        Outer container = Tuple:
            ([torch.tensor([1]), torch.tensor([2])], [torch.tensor([2]), torch.tensor([3])]) -> (torch.tensor([1, 2]), torch.tensor([2, 3]))

    Args:
        batches (Optional[Sequence[ReductionT]]): sequence of batches to collate into a single batch.

    Returns:
        A single batch of the same type as the elements of your input sequence.
    """
    match batches:
        case [torch.Tensor(), *_]:
            return torch.cat(batches, dim=0)
        case [dict(), *_]:
            return {key: batch_collator([batch[key] for batch in batches]) for key in batches[0]}
        case [tuple(), *_]:
            return tuple(batch_collator([batch[i] for batch in batches]) for i in range(len(batches[0])))
        case [list(), *_]:
            return [batch_collator([batch[i] for batch in batches]) for i in range(len(batches[0]))]
        case None:
            return None
        case []:
            raise ValueError("Cannot process an empty sequence")
        case _:
            raise ValueError("Unsupported input structure in batch_collator")


# TODO(@jstjohn): Properly use the Generic for DataT and ReductionT usage. Define our own batch/output types.
# TODO(@skothenhill): Re-think the generics here- the way that `batch_collator` is expressed, `batches` should be a recursive generic type.
class PassthroughLossReduction(MegatronLossReduction):
    """Internally in NeMo2.0 the forward step is always expected to return a loss reduction class, and forward is expected to return a loss.
    This class hijacks that mechanism to instead pass through the forward output unperturbed as the loss (to enable inference in the predict step), and then the
    reduce method is used to collate the batch of forward outputs into a single batch. This supports the model forward output being a tensor, dict, tuple,
    or list of tensors. The inner type _must always be a torch.Tensor_.
    """

    def forward(self, batch: DataT, forward_out: DataT) -> Tuple[torch.Tensor, DataT]:
        """_summary_

        Args:
            batch (DataT): The batch of data that was passed through the model to generate output.
            forward_out (torch.Tensor): The output from your model's forward pass.

        Returns:
            Tuple[torch.Tensor, ReductionT]: A tuple containing the loss tensor (dummy in this case) and the forward output (unmodified).
        """  # noqa: D415
        dtype, device = get_dtype_device(forward_out)
        return torch.zeros(1, device=device, dtype=dtype), forward_out

    def reduce(self, forward_out: List[DataT]) -> DataT:
        """This overrides the standard reduce with a simplified version that just takes a list of your model's forward outputs
            and collates them togehter into a single output.

        Args:
            forward_out (List[ReductionT]): _description_

        Returns:
            ReductionT: _description_
        """
        return batch_collator(forward_out)


class LightningPassthroughPredictionMixin:
    """A mixin that allows your model to do inference on the predict step by hijacking the nemo loss
    reduction mechanism and passing the model output through.
    """

    def predict_loss_reduction(self) -> PassthroughLossReduction:
        """For the predict step, pass through the forward pass output."""
        return PassthroughLossReduction()


class LossLoggingCallback(pl.Callback):
    def __init__(self):
        """Log the loss at the end of each batch. For training do not reduce across the epoch but do so for validation/test."""
        self.val_losses = []
        self.test_losses = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Assuming the loss is computed internally and stored in pl_module
        if torch.distributed.get_rank() == 0 and parallel_state.is_pipeline_last_stage():
            # TODO(@jstjohn): verify when the outputs are a dictionary of "loss" and when they are just one tensor value.
            if isinstance(outputs, dict):
                outputs = outputs["loss"]
            # torch.distributed.all_reduce(outputs, op=torch.distributed.ReduceOp.AVG)
            loss = outputs
            pl_module.log("train_loss", loss, on_step=True, prog_bar=True, logger=True, rank_zero_only=True)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # TODO(@jstjohn): Add a docstring with type hints for this lightning hook
        # Assuming the loss is computed internally and stored in pl_module
        if torch.distributed.get_rank() == 0 and parallel_state.is_pipeline_last_stage():
            # TODO(@jstjohn): verify when the outputs are a dictionary of "loss" and when they are just one tensor value.
            if isinstance(outputs, dict):
                outputs = outputs["loss"]
            # TODO verify that losses are already reduced across ranks
            # torch.distributed.all_reduce(outputs, op=torch.distributed.ReduceOp.AVG)
            loss = outputs
            self.test_losses.append(loss)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # TODO(@jstjohn): Add a docstring with type hints for this lightning hook
        # Assuming the loss is computed internally and stored in pl_module
        if torch.distributed.get_rank() == 0 and parallel_state.is_pipeline_last_stage():
            # TODO(@jstjohn): verify when the outputs are a dictionary of "loss" and when they are just one tensor value.
            if isinstance(outputs, dict):
                outputs = outputs["loss"]
            # TODO verify that losses are already reduced across ranks
            # torch.distributed.all_reduce(outputs, op=torch.distributed.ReduceOp.AVG)
            # TODO verify that losses are already reduced across ranks
            # torch.distributed.all_reduce(outputs, op=torch.distributed.ReduceOp.AVG)
            loss = outputs
            self.val_losses.append(loss)

    def on_validation_epoch_end(self, trainer, pl_module):
        # TODO(@jstjohn): Add a docstring with type hints for this lightning hook
        if torch.distributed.get_rank() == 0 and parallel_state.is_pipeline_last_stage():
            if len(self.val_losses) > 0:
                avg_val_loss = torch.stack(self.val_losses).mean()
                pl_module.log("val_loss", avg_val_loss, prog_bar=True, logger=True, rank_zero_only=True)
                self.val_losses.clear()

    def on_test_epoch_end(self, trainer, pl_module):
        # TODO(@jstjohn): Add a docstring with type hints for this lightning hook
        if torch.distributed.get_rank() == 0 and parallel_state.is_pipeline_last_stage():
            if len(self.test_losses) > 0:
                avg_test_loss = torch.stack(self.test_losses).mean()
                pl_module.log("test_loss", avg_test_loss, prog_bar=True, logger=True, rank_zero_only=True)
                self.test_losses.clear()
