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


from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Atom, Mol, rdMolDescriptors
from rdkit.Chem.rdchem import ChiralType, HybridizationType
from rdkit.Chem.Scaffolds import MurckoScaffold

from typing import Optional, Iterable, List
from bionemo.geometric.base_featurizer import (
    BaseAtomFeaturizer,
    get_boolean_atomic_prop,
    get_double_atomic_prop,
    one_hot_enc,
)


ALL_ATOM_FEATURIZERS = [
    "PeriodicTableFeaturizer",
    "ElectronicPropertyFeaturizer",
    "ScaffoldFeaturizer",
    "SmartsFeaturizer",
    "CrippenFeaturizer",
]


class AtomicNumberFeaturizer(BaseAtomFeaturizer):
    """Class for featurizing atom by its atomic number."""

    def __init__(self, max_atomic_num: Optional[int] = None) -> None:
        """Initializes AtomicNumberFeaturizer class."""
        MAX_ATOMIC_NUM = 118
        self.max_atomic_num = max_atomic_num if max_atomic_num else MAX_ATOMIC_NUM

    def n_dim(self) -> int:
        """Returns dimensionality of the computed features."""
        return self.max_atomic_num

    def get_atom_features(self, mol: Mol, atom_indices: Optional[Iterable] = None) -> List[int]:
        """Computes features of atoms of all of select atoms."""
        _atom_indices = atom_indices if atom_indices else range(mol.GetNumAtoms())
        return [mol.GetAtomWithIdx(a).GetAtomicNum() for a in _atom_indices]

    
class DegreeFeaturizer(BaseAtomFeaturizer):
    """Class for featurizing atom by its degree (excluding hydrogens) of connectivity."""

    @property
    def n_dim(self) -> int:
        """Returns dimensionality of the computed features."""
        return 6

    def get_atom_features(self, mol: Mol, atom_indices: Optional[Iterable] = None) -> List[int]:
        """Computes features of atoms of all of select atoms."""
        _atom_indices = atom_indices if atom_indices else range(mol.GetNumAtoms())
        return [mol.GetAtomWithIdx(a).GetDegree() for a in _atom_indices]


class TotalDegreeFeaturizer(BaseAtomFeaturizer):
    """Class for featurizing atom by its total degree (including hydrogens) of connectivity."""

    @property
    def n_dim(self) -> int:
        """Returns dimensionality of the computed features."""
        return 6

    def get_atom_features(self, mol: Mol, atom_indices: Optional[Iterable] = None) -> List[int]:
        """Computes features of atoms of all of select atoms."""
        _atom_indices = atom_indices if atom_indices else range(mol.GetNumAtoms())
        return [mol.GetAtomWithIdx(a).GetTotalDegree() for a in _atom_indices]


class ChiralTypeFeaturizer(BaseAtomFeaturizer):
    """Class for featurizing atom by its chirality type."""

    def __init__(self) -> None:
        """Initializes ChiralTypeFeaturizer class."""
        self.max_chiral_types = len(ChiralType.values)

    @property
    def n_dim(self) -> int:
        """Returns dimensionality of the computed features."""
        return self.max_chiral_types

    def get_atom_features(self, mol: Mol, atom_indices: Optional[Iterable] = None) -> List[int]:
        """Computes features of atoms of all of select atoms."""
        _atom_indices = atom_indices if atom_indices else range(mol.GetNumAtoms())
        return [int(mol.GetAtomWithIdx(a).GetChiralTag()) for a in _atom_indices]


class TotalNumHFeaturizer(BaseAtomFeaturizer):
    """Class for featurizing atom by total number of hydrogens."""

    def __init__(self) -> None:
        """Initializes TotalNumHFeaturizer class."""
        self.max_num_hs = 5  # 4 + 1 (no hydrogens)

    @property
    def n_dim(self) -> int:
        """Returns dimensionality of the computed features."""
        return self.max_num_hs

    def get_atom_features(self, mol: Mol, atom_indices: Optional[Iterable] = None) -> List[int]:
        """Computes features of atoms of all of select atoms."""
        _atom_indices = atom_indices if atom_indices else range(mol.GetNumAtoms())
        return [mol.GetAtomWithIdx(a).GetTotalNumHs() for a in _atom_indices]


class HybridizationFeaturizer(BaseAtomFeaturizer):
    """Class for featurizing atom by its hybridization type."""

    def __init__(self) -> None:
        """Initializes HybridizationFeaturizer class."""
        self.max_hybridization_types = len(HybridizationType.values)

    @property
    def n_dim(self) -> int:
        """Returns dimensionality of the computed features."""
        return self.max_hybridization_types

    def get_atom_features(self, mol: Mol, atom_indices: Optional[Iterable] = None) -> List[int]:
        """Computes features of atoms of all of select atoms."""
        _atom_indices = atom_indices if atom_indices else range(mol.GetNumAtoms())
        return [int(mol.GetAtomWithIdx(a).GetHybridization()) for a in _atom_indices]


class AromaticityFeaturizer(BaseAtomFeaturizer):
    """Class for featurizing atom based on its aromaticity."""

    @property
    def n_dim(self) -> int:
        """Returns dimensionality of the computed features."""
        return 1

    def get_atom_features(self, mol: Mol, atom_indices: Optional[Iterable] = None) -> List[int]:
        """Computes features of atoms of all of select atoms."""
        _atom_indices = atom_indices if atom_indices else range(mol.GetNumAtoms())
        return [int(mol.GetAtomWithIdx(a).GetIsAromatic()) for a in _atom_indices]


class PeriodicTableFeaturizer(BaseAtomFeaturizer):
    """Class for featurizing atom by its position (period and group) in the periodic table."""

    def __init__(self) -> None:
        """Initializes PeriodicTableFeaturizer class."""
        self.pt = Chem.GetPeriodicTable()

    @property
    def n_dim(self) -> int:
        """Returns dimensionality of the computed features."""
        return 25

    def get_period(self, atom: Chem.Atom) -> list[bool]:
        """Returns one-hot encoded period of atom."""
        atomic_number = atom.GetAtomicNum()
        # The number of elements per period in the periodic table
        period_limits = [2, 10, 18, 36, 54, 86, 118]

        # Determine the period based on atomic number
        for period, limit in enumerate(period_limits, start=1):
            if atomic_number <= limit:
                return period
        return None

    def get_group(self, atom: Chem.Atom) -> list[bool]:
        """Returns one-hot encoded group of atom."""
        group = self.pt.GetNOuterElecs(atom.GetAtomicNum())
        return group

    def get_atom_features(self, mol: Mol, atom_indices: Optional[Iterable] = None) -> list[int]:
        """Computes features of atoms of all of select atoms."""
        _atom_indices = atom_indices if atom_indices else range(mol.GetNumAtoms())
        return [[self.get_period(mol.GetAtomWithIdx(a)), self.get_group(mol.GetAtomWithIdx(a))] for a in _atom_indices]


class AtomicRadiusFeaturizer(BaseAtomFeaturizer):

    def __init__(self) -> None:
        """Initializes AtomicRadiusFeaturizer class."""
        self.min_val = [
            0.0, # Bond radius
            0.28, # Covalent radius
            1.2, # van der Waals radius
            ]
        self.max_val = [
            2.4, # Bond radius
            2.6, # Covalent radius
            3.0, # van der Waals radius
        ]

    @property
    def n_dim(self) -> int:
        """Returns dimensionality of the computed features."""
        return 3

    @property
    def min_val(self) -> np.array:
        return self.min_val

    @property
    def max_val(self) -> np.array:
        return self.max_val

    def get_atom_features(self, mol: Mol, atom_indices: Optional[Iterable] = None) -> np.array:
        """Computes bond radius, covalent radius, and van der Waals radius without normalization."""

        _atom_indices = atom_indices if atom_indices else range(mol.GetNumAtoms())

        feats = []
        for aidx in _atom_indices:
            atomic_num = mol.GetAtomWithIdx(aidx).GetAtomicNum()
            feats.append([pt.GetRb0(atomic_num), pt.GetRcovalent(atomic_num), pt.GetRvdw(atomic_num)])

        return np.array(feats)


class ElectronicPropertyFeaturizer(BaseAtomFeaturizer):
    """Class for featurizing atom by its electronic properties.

    This class computes electronic properties like electronegativity, ionization energy, and electron affinity.
    """

    def __init__(self, data_file=None) -> None:
        """Initializes PeriodicTableFeaturizer class.

        Args:
            data_file: Path to the data file.
        """
        if data_file is None:
            # Use default
            root_path = Path(__file__).resolve().parent
            data_file = root_path / "data" / "electronic_data.csv"
        self.data_df = pd.read_csv(data_file).set_index("AtomicNumber")

        self.pauling_en_dict = self.data_df["Electronegativity"].to_dict()
        self.ie_dict = self.data_df["IonizationEnergy"].to_dict()
        self.ea_dict = self.data_df["ElectronAffinity"].to_dict()

        self._min_val = np.array([
            self.data_df["Electronegativity"].min(),
            self.data_df["IonizationEnergy"].min(),
            self.data_df["ElectronAffinity"].min(),
        ])

        self._max_val = np.array([
            self.data_df["Electronegativity"].max(),
            self.data_df["IonizationEnergy"].max(),
            self.data_df["ElectronAffinity"].max(),
        ])

    @property
    def n_dim(self) -> int:
        """Returns dimensionality of the computed features."""
        return 3

    @property
    def min_val(self) -> np.array:
        return self._min_val

    @property
    def max_val(self) -> np.array:
        return self._max_val

    def get_atom_features(self, mol: Mol, atom_indices: Optional[Iterable] = None) -> np.array:
        """Returns features of the atom."""

        _atom_indices = atom_indices if atom_indices else range(mol.GetNumAtoms())

        feats = []
        for aidx in _atom_indices:
            atomic_num = mol.GetAtomWithIdx(aidx).GetAtomicNum()
            feats.append([
                self.pauling_en_dict[atomic_num],
                self.ie_dict[atomic_num],
                self.ea_dict[atomic_num]
                ])
        return np.array(feats)


class ScaffoldFeaturizer(BaseAtomFeaturizer):
    """Class for featurizing atom based on whether it is present in Bemis-Murcko scaffold."""

    @property
    def n_dim(self) -> int:
        """Returns dimensionality of the computed features."""
        return 1

    def get_atom_features(self, mol: Mol, atom_indices: Optional[Iterable] = None) -> list[bool]:
        """Returns features of the atom."""

        _atom_indices = atom_indices if atom_indices else range(mol.GetNumAtoms())

        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold_atom_idx = set(mol.GetSubstructMatch(scaffold))

        feats = [aidx in scaffold_atom_idx for aidx in _atom_indices]
        return feats


class SmartsFeaturizer(BaseAtomFeaturizer):
    """Class for featurizing atom by hydrogen donor/acceptor and acidity/basicity."""

    def __init__(self):
        """Initializes SmartsFeaturizer class."""
        self.hydrogen_donor = Chem.MolFromSmarts("[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]")
        self.hydrogen_acceptor = Chem.MolFromSmarts(
            "[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=[O,N,P,S])]),"
            "n&H0&+0,$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]"
        )
        self.acidic = Chem.MolFromSmarts("[$([C,S](=[O,S,P])-[O;H1,-1])]")
        self.basic = Chem.MolFromSmarts(
            "[#7;+,$([N;H2&+0][$([C,a]);!$([C,a](=O))]),$([N;H1&+0]([$([C,a]);!$([C,a](=O))])[$([C,a]);"
            "!$([C,a](=O))]),$([N;H0&+0]([C;!$(C(=O))])([C;!$(C(=O))])[C;!$(C(=O))])]"
        )

    @property
    def n_dim(self) -> int:
        """Returns dimensionality of the computed features."""
        return 4

    def get_atom_features(self, mol: Mol, atom_indices: Optional[Iterable] = None) -> list[list[bool]]:
        """Computes matches by prefixed SMARTS patterns."""
        hydrogen_donor_match = sum(mol.GetSubstructMatches(self.hydrogen_donor), ())
        hydrogen_acceptor_match = sum(mol.GetSubstructMatches(self.hydrogen_acceptor), ())
        acidic_match = sum(mol.GetSubstructMatches(self.acidic), ())
        basic_match = sum(mol.GetSubstructMatches(self.basic), ())

        _atom_indices = atom_indices if atom_indices else range(mol.GetNumAtoms())
        feats = [
            [aidx in hydrogen_donor_match,
            aidx in hydrogen_acceptor_match,
            aidx in acidic_match,
            aidx in basic_match,
            ] for aidx in _atom_indices]

        return feats


class CrippenFeaturizer(BaseAtomFeaturizer):
    """Class for featurizing atom by Crippen logP and molar refractivity."""

    def __init__(self):
        """Initializes CrippenFeaturizer class."""
        self.min_val = np.array([
            -2.996, # logP
            0.0, # MR
        ])

        self.max_val = np.array([
            0.8857, # logP
            6.0, # MR
        ])

    @property
    def n_dim(self) -> int:
        """Returns dimensionality of the computed features."""
        return 2

    def get_atom_features(self, mol: Mol, atom_indices: Optional[Iterable] = None) -> np.array:
        """Compute atomic contributions to Crippen logP and molar refractivity."""
        logp_mr_list = np.array(rdMolDescriptors._CalcCrippenContribs(mol))
        logp_mr_list[:, 0] = np.clip(logp_mr_list[:, 0], a_min=MIN_LOGP, a_max=MAX_LOGP)
        logp_mr_list[:, 1] = np.clip(logp_mr_list[:, 1], a_min=MIN_MR, a_max=MAX_MR)

        _atom_indices = atom_indices if atom_indices else range(mol.GetNumAtoms())
        return logp_mr_list[_atom_indices, :]

# # TODO Implement more features
# ## - Size of ring atom is present in
# ## - 2D partial charges like Gasteiger charges
