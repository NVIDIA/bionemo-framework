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


from typing import Optional, Tuple

import torch
from torch import Tensor

from bionemo.moco.interpolants.base_interpolant import pad_like


class KabschAugmentation:
    """Point-wise Kabsch alignment."""

    def __init__(self):
        """Initialize the KabschAugmentation instance.

        Notes:
            - This implementation assumes no required initialization arguments.
            - You can add instance variables (e.g., `self.variable_name`) as needed.
        """
        pass  # No operations are performed when initializing with no args

    def kabsch_align(self, target: Tensor, noise: Tensor):
        """Find the Rotation matrix (R) such that RMSD is minimized between target @ R.T and noise.

        Args:
            target (Tensor): shape (N, *dim), data from source minibatch.
            noise (Tensor): shape (N, *dim), noise from source minibatch.

        Returns:
            R (Tensor): shape (*dim, *dim), the rotation matrix.
            Aliged Target (Tensor): target tensor rotated and shifted to reduced RMSD with noise
        """
        dimension = target.shape[-1]
        noise_translation = noise.mean(dim=0)
        noise_centered = noise - noise_translation
        target_centered = target - target.mean(dim=0)

        # Compute the covariance matrix
        covariance_matix = target_centered.T @ noise_centered

        # Compute the SVD of the covariance matrix
        U, S, Vt = torch.linalg.svd(covariance_matix)
        d = torch.sign(torch.linalg.det(Vt.T @ U.T)).item()
        d_mat = torch.tensor([1] * (dimension - 1) + [d], device=Vt.device, dtype=Vt.dtype)
        R = Vt.T @ torch.diag(d_mat) @ U.T

        target_aligned = target_centered @ R.T + noise_translation

        return R, target_aligned

    def batch_kabsch_align(self, target: Tensor, noise: Tensor):
        """Find the Rotation matrix (R) such that RMSD is minimized between target @ R.T and noise.

        Args:
            target (Tensor): shape (N, *dim), data from source minibatch.
            noise (Tensor): shape (N, *dim), noise from source minibatch.

        Returns:
            R (Tensor): shape (*dim, *dim), the rotation matrix.
            Aliged Target (Tensor): target tensor rotated and shifted to reduced RMSD with noise
        """
        # Corrected Batched Kabsch Alignment
        batch_size, _, dimension = target.shape

        # Center the target and noise tensors along the middle dimension (N) for each batch item
        noise_translation = noise.mean(dim=1, keepdim=True)
        noise_centered = noise - noise_translation
        target_centered = target - target.mean(dim=1, keepdim=True)

        # Compute the covariance matrix for each batch item
        covariance_matrix = torch.matmul(target_centered.transpose(1, 2), noise_centered)

        # Compute the SVD of the covariance matrix for each batch item
        U, S, Vt = torch.linalg.svd(covariance_matrix)

        # Adjust for proper rotation (determinant=1) for each batch item
        d = torch.sign(torch.linalg.det(Vt @ U.transpose(-1, -2)))  # Keep as tensor for batch operations
        d_mat = torch.diag_embed(
            torch.cat(
                [torch.ones(batch_size, dimension - 1, device=Vt.device, dtype=Vt.dtype), d.unsqueeze(-1)], dim=-1
            )
        )

        R_batch = torch.matmul(torch.matmul(Vt.transpose(-1, -2), d_mat), U.transpose(-1, -2))

        target_aligned = target_centered @ R_batch.transpose(-1, -2) + noise_translation
        return R_batch, target_aligned

    def apply_ot(
        self,
        x0: Tensor,
        x1: Tensor,
        mask: Optional[Tensor] = None,
        align_noise_to_data=True,
    ) -> Tuple[Tensor, Tensor]:
        r"""Sample indices for noise and data in minibatch according to OT plan.

        Compute the OT plan $\pi$ (wrt squared Euclidean cost after Kabsch alignment) between a source and a target
        minibatch and draw source and target samples from pi $(x,z) \sim \pi$.

        Args:
            x0 (Tensor): shape (bs, *dim), noise from source minibatch.
            x1 (Tensor): shape (bs, *dim), data from source minibatch.
            mask (Optional[Tensor], optional): mask to apply to the output, shape (batchsize, nodes), if not provided no mask is applied. Defaults to None.
            replace (bool): sampling w/ or w/o replacement from the OT plan, default to False.
            align_noise_to_data (bool): Direction of alignment default is True meaning it augments Noise to reduce error to Data.

        Returns:
            Tuple: tuple of 2 tensors, represents the noise and data samples following OT plan pi.
        """
        if x1.ndim > 2:
            align_func = self.batch_kabsch_align
        else:
            align_func = self.kabsch_align
        if mask is not None:
            mask = pad_like(mask, x1)
            x1 = x1 * mask
            x0 = x0 * mask
        if align_noise_to_data:
            # Compute the rotation matrix R that aligns x0 to x1
            R, aligned_x0 = align_func(x0, x1)
            noise = aligned_x0
            data = x1
        else:
            # Compute the rotation matrix R that aligns x1 to x0
            R, aligned_x1 = align_func(x1, x0)
            noise = x0
            data = aligned_x1
        if mask is not None:
            noise = noise * mask
            data = data * mask
        # Output the permuted samples in the minibatch
        return noise, data
