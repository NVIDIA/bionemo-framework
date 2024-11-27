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

from rdkit import Chem
from rdkit.Chem.rdchem import BondStereo, BondType

from bionemo.geometric.base_featurizer import BaseFeaturizer, one_hot_enc


ALL_BOND_FEATURIZERS = ["RingFeaturizer"]
N_BOND_TYPES = 4  # currently only single, aromatic, double, and triple
N_BOND_STEREO_TYPES = len(BondStereo.values)


class BondTypeFeaturizer(BaseFeaturizer):
    """Class for featurizing bond its bond type."""

    def __init__(self) -> None:
        """Initializes BondTypeFeaturizer class."""
        pass

    @property
    def n_dim(self) -> int:
        """Returns dimensionality of the computed features."""
        return N_BOND_TYPES

    def get_features(self, bond: Chem.Bond) -> List[bool]:
        """Returns features of the bond."""
        bond_type = bond.GetBondType()
        return [
            bond_type == BondType.SINGLE,
            bond_type == BondType.AROMATIC,
            bond_type == BondType.DOUBLE,
            bond_type == BondType.TRIPLE,
        ]


class BondConjugationFeaturizer(BaseFeaturizer):
    """Class for featurizing bond based on its conjugation."""

    def __init__(self) -> None:
        """Initializes BondConjugationFeaturizer class."""
        pass

    @property
    def n_dim(self) -> int:
        """Returns dimensionality of the computed features."""
        return 1

    def get_features(self, bond: Chem.Bond) -> List[bool]:
        """Returns features of the bond."""
        return [bond.GetIsConjugated()]


class BondStereochemistryFeaturizer(BaseFeaturizer):
    """Class for featurizing bond based on its stereochemistry ex. cis/trans."""

    def __init__(self) -> None:
        """Initializes BondStereochemistryFeaturizer class."""
        pass

    @property
    def n_dim(self) -> int:
        """Returns dimensionality of the computed features."""
        return N_BOND_STEREO_TYPES

    def get_features(self, bond: Chem.Bond) -> List[bool]:
        """Returns features of the bond."""
        return one_hot_enc(int(bond.GetStereo()), self.n_dim)


class RingFeaturizer(BaseFeaturizer):
    """Class for featurizing bond its ring membership."""

    def __init__(self, n_ring_sizes=7) -> None:
        """Initializes RingFeaturizer class."""
        self.n_ring_sizes = n_ring_sizes  # ring size 3 - 8 and UNK

    @property
    def n_dim(self) -> int:
        """Returns dimensionality of the computed features."""
        return self.n_ring_sizes

    def compute_features(self, mol) -> Chem.Mol:
        """Precomputes ring membership features for all bonds in the molecule."""
        ri = mol.GetRingInfo()

        for bond in mol.GetBonds():
            bidx = bond.GetIdx()
            bond_ring_sizes = ",".join(map(str, ri.BondRingSizes(bidx)))
            bond.SetProp("bond_ring_sizes", bond_ring_sizes)
        return mol

    def get_features(self, bond: Chem.Bond) -> List[bool]:
        """Returns features of the bond."""
        ring_sizes = bond.GetProp("bond_ring_sizes").split(",")
        ring_sizes = [int(r) for r in ring_sizes if r != ""]

        feats = [False] * self.n_ring_sizes
        for r in ring_sizes:
            if r > 8:
                feats[-1] = True
            else:
                feats[r - 3] = True  # ring of size 3 is at first position
        return feats
