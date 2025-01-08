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


import warnings
from functools import partial
from typing import Callable, Literal, Optional, Tuple, Union

import ot as pot
import torch
from jaxtyping import Bool
from torch import Tensor


class OTSampler:
    """Sampler for Exact Mini-batch Optimal Transport Plan.

    OTSampler implements sampling coordinates according to an OT plan (wrt squared Euclidean cost)
    with different implementations of the plan calculation. Code is adapted from https://github.com/atong01/conditional-flow-matching/blob/main/torchcfm/optimal_transport.py

    """

    def __init__(
        self,
        method: str = "exact",
        device: Union[str, torch.device] = "cpu",
        num_threads: int = 1,
    ) -> None:
        """Initialize the OTSampler class.

        Args:
            method (str): Choose which optimal transport solver you would like to use. Currently only support exact OT solvers (pot.emd).
            device (Union[str, torch.device], optional): The device on which to run the interpolant, either "cpu" or a CUDA device (e.g. "cuda:0"). Defaults to "cpu".
            num_threads (Union[int, str], optional): Number of threads to use for OT solver. If "max", uses the maximum number of threads. Default is 1.

        Raises:
            ValueError: If the OT solver is not documented.
            NotImplementedError: If the OT solver is not implemented.
        """
        # ot_fn should take (a, b, M) as arguments where a, b are marginals and
        # M is a cost matrix
        if method == "exact":
            self.ot_fn: Callable[..., torch.Tensor] = partial(pot.emd, numThreads=num_threads)  # type: ignore
        elif method in {"sinkhorn", "unbalanced", "partial"}:
            raise NotImplementedError("OT solver other than 'exact' is not implemented.")
        else:
            raise ValueError(f"Unknown method: {method}")
        self.device = device

    def to_device(self, device: str):
        """Moves all internal tensors to the specified device and updates the `self.device` attribute.

        Args:
            device (str): The device to move the tensors to (e.g. "cpu", "cuda:0").

        Note:
            This method is used to transfer the internal state of the OTSampler to a different device.
            It updates the `self.device` attribute to reflect the new device and moves all internal tensors to the specified device.
        """
        self.device = device
        for attr_name in dir(self):
            if attr_name.startswith("_") and isinstance(getattr(self, attr_name), torch.Tensor):
                setattr(self, attr_name, getattr(self, attr_name).to(device))
        return self

    def sample_map(self, pi: Tensor, batch_size: int, replace: Bool = False) -> Tuple[Tensor, Tensor]:
        r"""Draw source and target samples from pi $(x,z) \sim \pi$.

        Args:
            pi (Tensor): shape (bs, bs), the OT matrix between noise and data in minibatch.
            batch_size (int): The batch size of the minibatch.
            replace (bool): sampling w/ or w/o replacement from the OT plan, default to False.

        Returns:
            Tuple: tuple of 2 tensors, represents the indices of noise and data samples from pi.
        """
        if pi.shape[0] != batch_size or pi.shape[1] != batch_size:
            raise ValueError("Shape mismatch: pi.shape = {}, batch_size = {}".format(pi.shape, batch_size))
        p = pi.flatten()
        p = p / p.sum()
        choices = torch.multinomial(p, batch_size, replacement=replace)
        return torch.div(choices, pi.shape[1], rounding_mode="floor"), choices % pi.shape[1]

    def _calculate_cost_matrix(self, x0: Tensor, x1: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Compute the cost matrix between a source and a target minibatch.

        Args:
            x0 (Tensor): shape (bs, *dim), noise from source minibatch.
            x1 (Tensor): shape (bs, *dim), data from source minibatch.
            mask (Optional[Tensor], optional): mask to apply to the output, shape (batchsize, nodes), if not provided no mask is applied. Defaults to None.

        Returns:
            Tensor: shape (bs, bs), the cost matrix between noise and data in minibatch.
        """
        if mask is None:
            # Flatten the input tensors
            x0, x1 = x0.reshape(x0.shape[0], -1), x1.reshape(x1.shape[0], -1)

            # Compute the cost matrix. For exact OT, we use squared Euclidean distance.
            M = torch.cdist(x0, x1) ** 2
        else:
            # Initialize the cost matrix
            M = torch.zeros((x0.shape[0], x1.shape[0]))
            # For each x0 sample, apply its mask to all x1 samples and calculate the cost
            for i in range(x0.shape[0]):
                x0i_mask = mask[i].unsqueeze(-1)
                masked_x1 = x1 * x0i_mask
                masked_x0 = x0[i] * x0i_mask
                cost = torch.cdist(masked_x0.reshape(1, -1), masked_x1.reshape(x1.shape[0], -1)) ** 2
                M[i] = cost
        return M

    def get_ot_matrix(self, x0: Tensor, x1: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Compute the OT matrix between a source and a target minibatch.

        Args:
            x0 (Tensor): shape (bs, *dim), noise from source minibatch.
            x1 (Tensor): shape (bs, *dim), data from source minibatch.
            mask (Optional[Tensor], optional): mask to apply to the output, shape (batchsize, nodes), if not provided no mask is applied. Defaults to None.

        Returns:
            p (Tensor): shape (bs, bs), the OT matrix between noise and data in minibatch.

        """
        # Compute the cost matrix
        M = self._calculate_cost_matrix(x0, x1, mask)
        # Set uniform weights for all samples in a minibatch
        a, b = pot.unif(x0.shape[0], type_as=M), pot.unif(x1.shape[0], type_as=M)

        p = self.ot_fn(a, b, M)
        # Handle exceptions
        if not torch.all(torch.isfinite(p)):
            raise ValueError("OT plan map is not finite, cost mean, max: {}, {}".format(M.mean(), M.max()))
        if torch.abs(p.sum()) < 1e-8:
            warnings.warn("Numerical errors in OT matrix, reverting to uniform plan.")
            p = torch.ones_like(p) / p.numel()

        return p

    def apply_ot(
        self,
        x0: Tensor,
        x1: Tensor,
        mask: Optional[Tensor] = None,
        replace: Bool = False,
        sort: Optional[Literal["noise", "x0", "data", "x1"]] = "x0",
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        r"""Sample indices for noise and data in minibatch according to OT plan.

        Compute the OT plan $\pi$ (wrt squared Euclidean cost) between a source and a target
        minibatch and draw source and target samples from pi $(x,z) \sim \pi$.

        Args:
            x0 (Tensor): shape (bs, *dim), noise from source minibatch.
            x1 (Tensor): shape (bs, *dim), data from source minibatch.
            mask (Optional[Tensor], optional): mask to apply to the output, shape (batchsize, nodes), if not provided no mask is applied. Defaults to None.
            replace (bool): sampling w/ or w/o replacement from the OT plan, default to False.
            sort (str): Optional Literal string to sort either x1 or x0 based on the input.

        Returns:
            Tuple: tuple of 2 tensors or 3 tensors if mask is used, represents the noise (plus mask) and data samples following OT plan pi.
        """
        if replace and sort is not None:
            raise ValueError("Cannot sample with replacement and sort")
        # Calculate the optimal transport
        pi = self.get_ot_matrix(x0, x1, mask)

        # Sample (x0, x1) mapping indices from the OT matrix
        i, j = self.sample_map(pi, x0.shape[0], replace=replace)
        if not replace and (sort == "noise" or sort == "x0"):
            sort_idx = torch.argsort(i)
            i = i[sort_idx]
            j = j[sort_idx]

            if not (i == torch.arange(x0.shape[0], device=i.device)).all():
                raise ValueError("x0_idx should be a tensor from 0 to size - 1 when sort is 'noise' or 'x0")
            noise = x0
            data = x1[j]
        elif not replace and (sort == "data" or sort == "x1"):
            sort_idx = torch.argsort(j)
            i = i[sort_idx]
            j = j[sort_idx]

            if not (j == torch.arange(x1.shape[0], device=j.device)).all():
                raise ValueError("x1_idx should be a tensor from 0 to size - 1 when sort is 'noise' or 'x0")
            noise = x0[i]
            data = x1
        else:
            noise = x0[i]
            data = x1[j]

        # Output the permuted samples in the minibatch
        if mask is not None:
            if mask.device != x0.device:
                mask = mask.to(x0.device)
            mask = mask[i]
        return noise, data, mask
