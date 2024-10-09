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


from typing import Optional

# Disable logging of invalid SMILES moloecules
from rdkit import RDLogger

from bionemo.data.molecule import MoleculeEnumeration
from bionemo.data.molecule.megamolbart_utils import megamolbart_build_train_valid_test_datasets
from bionemo.model.molecule.mol_enc_dec_model_base import MolEncDecModelBase


lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

__all__ = ["MegaMolBARTModel"]


class MegaMolBARTModel(MolEncDecModelBase):
    """
    MegaMolBART pretraining
    """

    def _setup_collate(self, pad_size_divisible_by_8: bool):
        """
        Sets up collate fn that is required by dataloader used to pretrain MegaMolBART
        Args:
            pad_size_divisible_by_8: should torch.Tensors be padded to sizes divisible by 8?
        Returns:
            callable collate fn
        """
        return MoleculeEnumeration(
            tokenizer=self.tokenizer,
            seq_length=self._cfg.seq_length,
            pad_size_divisible_by_8=pad_size_divisible_by_8,
            **self._cfg.data,
        ).collate_fn

    def _load_train_valid_test_datasets(self, train_n_samples: Optional[int] = None):
        """
        Helper method that sets instance variables corresponding to train, val and test datasets
        Args:
            train_n_samples: number of samples to limit training set to, if defined. Validation/Test sets should be controlled with trainer.limit_val_batches and trainer.limit_test_batches
        """
        self._train_ds, self._validation_ds, self._test_ds = megamolbart_build_train_valid_test_datasets(
            self._cfg.data, train_n_samples=train_n_samples
        )
