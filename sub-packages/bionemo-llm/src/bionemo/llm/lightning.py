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

from typing import Any, Callable, Dict, Generic, Iterable, Iterator, List, Optional, Sequence, Tuple, TypeVar, Union

import lightning.pytorch as pl
import torch.distributed
from megatron.core import parallel_state
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from nemo.lightning import io as nlio
from nemo.lightning.megatron_parallel import (
    CallbackMethods,
    DataT,
    MegatronLossReduction,
    MegatronStep,
    ReductionT,
)
from nemo.lightning.pytorch.optim import MegatronOptimizerModule
from torch import Tensor
from typing_extensions import override

from bionemo.core.model.config import BionemoTrainableModelConfig
from bionemo.llm.api import MegatronLossType, MegatronModelType
from bionemo.llm.model.loss import unreduced_token_loss_fn


__all__: Sequence[str] = (
    "get_dtype_device",
    "batch_collator",
    "PassthroughLossReduction",
    "LightningPassthroughPredictionMixin",
    "PerplexityLoggingCallback",
    "BionemoLightningModule",
    "default_megatron_optimizer",
)


T = TypeVar("T")
BatchT = TypeVar("BatchT")


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
        case Tensor(dtype=dtype, device=device):
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


# NOTE(SKH): These types are all wrong, but are close. The inner type must always be a Tensor, but the outer container should be generic.
def batch_collator(
    batches: Optional[Union[Tuple[ReductionT], List[ReductionT]]],
    batch_dim: int = 0,
    batch_dim_key_defaults: dict[str, int] = {"token_logits": 1},
) -> Optional[ReductionT]:
    """Takes a sequence of batches and collates them into a single batch.

        This is distinct from the standard pytorch default_collator since it does
        not add the batch dimension, it's assumed the batch
        dimension is already present in the input, as would be the case when
        parallelizing across minibatches.

    IMPORTANT: The underlying data primitive _must_ be a torch Tensor. The input to this function is a recurisve type,
    there can be any amount of nesting between dictionaries, tuples, and lists, as long as the inner type is a n-d Tensor.

    Examples:
        Outer container = Dict:
            [{'a': Tensor([1]), 'b': Tensor([2])}, {'a': Tensor([2]), 'b': Tensor([3])}] -> {'a': Tensor([1, 2]), 'b': Tensor([2, 3])}
        Outer container = List:
            [[Tensor([1]), Tensor([2])], [Tensor([2]), Tensor([3])]] -> [Tensor([1, 2]), Tensor([2, 3])]
        Outer container = Tuple:
            ([Tensor([1]), Tensor([2])], [Tensor([2]), Tensor([3])]) -> (Tensor([1, 2]), Tensor([2, 3]))

    Args:
        batches (Optional[Sequence[ReductionT]]): sequence of batches to collate into a single batch.
        batch_dim: If you know that the batch dim for the batch you are concatenating is not the 0th dimension (for
            example it is sequence first) then supply that dimension.
        batch_dim_key_defaults (dictionary of keys to integers): If your batch is a dictionary and you know that some
            keys have non-standard (0) batch dimensions, supply those here. By default "token_logits" has batch dim 1
            and otherwise all keys are assumed to have batch dim 0.

    Returns:
        A single batch of the same type as the elements of your input sequence.
    """
    match batches:
        # Handle base-cases for batch concatenation, either a list of None or a list of tensors
        case [None, *_]:
            return None
        case [Tensor(), *_]:
            return torch.cat(batches, dim=batch_dim)
        # Next 3 calls are the recursive calls into the sub-structures of the batch. We handle dictionaries, tuples, and lists
        case [dict(), *_]:
            return {
                key: batch_collator(
                    [batch[key] for batch in batches],
                    batch_dim=batch_dim_key_defaults.get(key, 0),
                    batch_dim_key_defaults=batch_dim_key_defaults,
                )
                for key in batches[0]
            }
        case [tuple(), *_]:
            return tuple(
                batch_collator(
                    [batch[i] for batch in batches], batch_dim=batch_dim, batch_dim_key_defaults=batch_dim_key_defaults
                )
                for i in range(len(batches[0]))
            )
        case [list(), *_]:
            return [
                batch_collator(
                    [batch[i] for batch in batches], batch_dim=batch_dim, batch_dim_key_defaults=batch_dim_key_defaults
                )
                for i in range(len(batches[0]))
            ]
        # Final cases shouldn't happen, an empty sequence (no batches), or "other".
        case []:
            raise ValueError("Cannot process an empty sequence")
        case _:
            raise ValueError("Unsupported input structure in batch_collator")


# TODO(@jstjohn): Properly use the Generic for DataT and ReductionT usage. Define our own batch/output types.
# TODO(@skothenhill): Re-think the generics here- the way that `batch_collator` is expressed, `batches` should be a recursive generic type.
class PassthroughLossReduction(MegatronLossReduction, Generic[DataT]):
    """A workaround for nemo/megatron to perform inference.

    Internally in NeMo2.0 the forward step is always expected to return a loss reduction class, and forward is
    expected to return a loss. This class hijacks that mechanism to instead pass through the forward output unperturbed
    as the loss (to enable inference in the predict step), and then the reduce method is used to collate the batch of
    forward outputs into a single batch. This supports the model forward output being a tensor, dict, tuple, or list of
    tensors. The inner type _must always be a Tensor_.
    """

    def forward(self, batch: DataT, forward_out: DataT) -> Tuple[Tensor, DataT]:
        """Passes through the `forward_out` value as the 2nd tuple element.

        Args:
            batch: The batch of data that was passed through the model to generate output. NOTE: this value is ignored.
            forward_out: The output from your model's forward pass.

        Returns:
            A tuple containing the loss tensor (dummy in this case) and the forward output (unmodified).
        """
        return torch.zeros((1, 1)), forward_out

    def reduce(self, forward_out: List[DataT]) -> DataT:
        """Collates list of model's outputs into a single output."""
        return batch_collator(forward_out)


class LightningPassthroughPredictionMixin:
    """A mixin that allows your model to do inference on the predict step by hijacking nemo's loss reduction mechanism."""

    def predict_loss_reduction(self) -> PassthroughLossReduction:
        """For the predict step, pass through the forward pass output."""
        return PassthroughLossReduction()


ForwardStep = Callable[[MegatronModelType, DataT], DataT]
"""Megatron-compatible forward pass function.
"""

DataStep = Callable[[Iterator[DataT]], DataT]
"""Batches together an iterator of individual examples.

Necessary for compatability with Megatron. This function type is similiar to the collate function of PyTorch.

A `DataStep` function takes an iterator over individual examples. Each example may be a tensor, sequence of tensors,
or a set of named tensors (provided as a `dict` mapping `str` names to each `Tensor`). Each iteration must
yield the same type.

The output of this function will mirror the same structure of each yielded example. It will be a concatenation of all
of the examples in the iterator.
"""


class BionemoLightningModule(
    Generic[MegatronModelType, MegatronLossType],
    pl.LightningModule,
    nlio.IOMixin,
    nlio.ConnectorMixin,
    LightningPassthroughPredictionMixin,
):
    """Reusable PyTorch Lightning module for Megatron models that is compatible with NeMo's conventions."""

    def __init__(
        self,
        config: BionemoTrainableModelConfig[MegatronModelType, MegatronLossType],
        forward_step: ForwardStep,
        data_step: DataStep,
        # TODO: Add transformer_layer_spec when we update mcore
        optimizer: MegatronOptimizerModule,
        model_transform: Optional[Callable[[MegatronModelType], MegatronModelType]] = None,
        **model_construct_args,
    ) -> None:
        """Constructor.

        Args:
            config: Serializable configuration object that allows one to construct a new model instance and loss
                function. Necessary for Megatron-based training as the model itself cannot be serialized and
                distributed to nodes. Instead, we serialize the procedure for making the model and distribute that.
            forward_step: Performs forward pass using the model and a batch of data.
            data_step: Custom batch-creating function for the model.
            optimizer: Megatron-compatible distributed optimizer instance. Defaults to using ADAM with a 1e-4 learning
                rate.
            model_construct_args: Optional. Any arguments necessary to construct the model in the `config`'s
                `configure_model` method.
            model_transform: Optional. The model transform function.
            **model_construct_args: Optional. Arguments necessary for the supplied model configuration's
                `configure_model` method, which will make an instance of the model.
        """
        super().__init__()
        self.config = config
        self.module_construct_args: Optional[dict[str, Any]] = model_construct_args
        # ***must** be set up in configure_model() -- megatron constraint
        # also, must be called `module`: nemo expects the actual model to be stored this way
        self.module: Optional[MegatronModelType] = None
        self.loss_reduction_class: type[MegatronLossType] = config.get_loss_reduction_class()
        self.optim = optimizer
        self.optim.connect(self)  # This will bind the `configure_optimizers` method
        self._data_step = data_step
        self._forward_step = forward_step
        self.model_transform = model_transform

    def configure_model(self) -> None:
        """Updates internal state: instantiates the model from the object's config, assigns to `model` attribute.

        NOTE: this method is idempotent; successive calls have no effect. The model is only initialized once.

        Raises:
            ValueError iff the internal config's configure_model method returns None.
        """
        if self.module is None:
            model: MegatronModelType = (
                self.config.configure_model(**self.module_construct_args)
                if self.module_construct_args is not None
                else self.config.configure_model()
            )
            self.module = model
        if self.module is None:
            raise ValueError("Invalid semantics: configure_model method **MUST** initialize the model.")

    def forward(self, *args, **kwargs) -> DataT:
        """Call the forward method of the underlying model, and return whatever it outputs."""
        # safe to do because configure_model is idempotent
        self.configure_model()
        assert self.module is not None, "ERROR: configure_model() method has been incorrectly overridden!"
        prediction = self.module(*args, **kwargs)  # for now just pass through to the underlying model
        return prediction

    def data_step(self, dataloader_iter: Iterator[DataT]) -> DataT:  # noqa: D102
        return self._data_step(dataloader_iter)

    def forward_step(self, batch) -> Tensor:
        """Megatron-required: the training forward step for the model, which is required to produce the loss.

        Normally, the forward pass of a model means its inference. Loss is computed using the predictions
        from the forward pass against labels. Megatron unfortunately conflates these two different concepts
        and instead has models "forward" method produce the loss. See the Megatron docs for details:
        https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/pipeline_parallel/schedules.py#L170

        To get actual predictions, use the :func:`forward` method instead.
        """
        # safe to do because configure_model is idempotent
        self.configure_model()
        assert self.module is not None
        return self._forward_step(self.module, batch)

    def training_step(self, batch, batch_idx: Optional[int] = None) -> Tensor:
        """In mcore the loss-function is part of the forward-pass when labels are provided."""
        return self.forward_step(batch)

    def validation_step(self, batch, batch_idx: Optional[int] = None) -> Tensor:
        """In mcore the loss-function is part of the forward-pass when labels are provided."""
        return self.forward_step(batch)

    def predict_step(self, batch, batch_idx: Optional[int] = None) -> Tensor:
        """Alias for forward_step."""
        if len(batch) == 0:
            return
        return self.forward_step(batch)

    def training_loss_reduction(self) -> MegatronLossType:
        """This is the function that takes batch['loss_mask'] and the logits output by the model and reduces the loss."""
        return self.loss_reduction_class()

    def validation_loss_reduction(self) -> MegatronLossType:  # noqa: D102
        return self.loss_reduction_class(validation_step=True)

    def test_loss_reduction(self) -> MegatronLossType:  # noqa: D102
        return self.loss_reduction_class(validation_step=True)


def default_megatron_optimizer() -> MegatronOptimizerModule:
    """Default distributed optimizer uses Adam with a 1e-4 learning rate."""
    return MegatronOptimizerModule(
        config=OptimizerConfig(lr=1e-4, optimizer="adam", use_distributed_optimizer=True),
    )


class PerplexityLoggingCallback(pl.Callback, CallbackMethods):
    """Megatron Callback to log perplexity in validation and optionally training.

    NeMo2.0 checks whether a callback is an instance of {LightningModule,LightningDataModule,Callback} but only megatron_hooks are useful.
    """

    def __init__(self, log_train: bool = False, log_val: bool = True):
        """Initialize PerplexityLoggingCallback.

        Args:
            log_train: whether to log train perplexity. Defaults to False.
            log_val: whether to log validation perplexity. Defaults to True.
        """
        super().__init__()
        self.log_train = log_train
        self.log_val = log_val

    def _pad_to_max_length(
        self,
        microbatch_outputs: List[Dict[str, Dict[str, Tensor]]],
        key1: str,
        key2: str,
        pad_value: int = 0,
        seq_dim: int = 1,
        batch_dim: int = 0,
    ) -> Tensor:
        """Pad tensors to max length in microbatch_outputs."""
        assert seq_dim != batch_dim, "Forgot to set one of seq_dim, batch_dim, they are equal!"
        max_sequence_length: int = max(output[key1][key2].shape[seq_dim] for output in microbatch_outputs)

        tensors: List[Tensor] = []
        for microbatch_output in microbatch_outputs:
            tensor = microbatch_output[key1][key2]
            assert (
                tensor.dim() >= 2
            ), f"Tensor in microbatch_outputs must have at least 2 dimensions, but got {tensor.dim()} dimensions"
            pad_size = [(0, 0)] * tensor.dim()
            pad_size[seq_dim] = (0, max_sequence_length - tensor.shape[seq_dim])
            # Flatten pad size list for F.pad
            pad_size_flat = [item for sublist in reversed(pad_size) for item in sublist]
            tensors.append(
                torch.nn.functional.pad(  # padding reverse in order
                    tensor,
                    pad_size_flat,
                    mode="constant",
                    value=pad_value,
                )
            )

        return torch.cat(tensors, dim=batch_dim)  # concat on batch dim

    @override
    def on_megatron_reduce_microbatches_end(
        self,
        step: MegatronStep,
        microbatch_outputs: List[Any],
        loss_reduction: MegatronLossReduction,
        reduced: Tensor | dict[str, Tensor],
    ) -> None:
        """Log after MegatronReductionLoss.reduce is called.

        Expected microbatch_outputs to be a list of dicts with the following keys:
            - batch: dict of tensors with the following keys:
                - labels: [b s]
                - loss_mask: [b s]; 1 means included 0 means ignored
            - forward_out: dict of tensors with the following keys:
                - token_logits: [b s vocab]
        """
        if step.trainer.sanity_checking:  # skip sanity check
            return

        if step.trainer.training and not self.log_train:
            return

        if not parallel_state.is_pipeline_last_stage():
            return

        assert step.num_microbatches is not None, "num_microbatches must be initialized to non-None"
        assert step.num_microbatches > 0, "num_microbatches must be greater than 0"
        assert (
            len(microbatch_outputs) == step.num_microbatches
        ), "microbatch_outputs length does not match num_microbatches"
        labels = self._pad_to_max_length(microbatch_outputs, "batch", "labels", pad_value=-100)
        loss_mask = self._pad_to_max_length(microbatch_outputs, "batch", "loss_mask")
        token_logits = self._pad_to_max_length(
            microbatch_outputs, "forward_out", "token_logits", seq_dim=0, batch_dim=1
        )

        unreduced_token_loss = unreduced_token_loss_fn(
            token_logits.clone(),  # [s,b] as expected unreduced_token_loss_fn has inplace operation on token_logits
            labels.clone(),  # [b,s] as expected
        )  # [b s] is the return

        cp_size = parallel_state.get_context_parallel_world_size()
        if cp_size == 1:
            ppl = torch.exp((unreduced_token_loss * loss_mask).sum() / loss_mask.sum())
        else:
            raise NotImplementedError("Context parallel perplexity logging is not supported yet")

        if self.log_val and not step.trainer.training:
            step.pl_module.log("val_ppl", ppl, prog_bar=True, on_epoch=True)
        elif self.log_train and step.trainer.training:
            step.pl_module.log("train_ppl", ppl, prog_bar=True, batch_size=1, sync_dist=False)
