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
from typing import List

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Atom, rdMolDescriptors
from rdkit.Chem.rdchem import ChiralType, HybridizationType
from rdkit.Chem.Scaffolds import MurckoScaffold

from bionemo.geometric.base_featurizer import (
    BaseFeaturizer,
    get_boolean_atomic_prop,
    get_double_atomic_prop,
    one_hot_enc,
)


# Extremum for molar refractivity
MIN_MR = 0.0
MAX_MR = 6.0

# Extremum for logP
MIN_LOGP = -2.996
MAX_LOGP = 0.8857

ALL_ATOM_FEATURIZERS = [
    "PeriodicTableFeaturizer",
    "ElectronicPropertyFeaturizer",
    "ScaffoldFeaturizer",
    "SmartsFeaturizer",
    "CrippenFeaturizer",
]

MAX_ATOMIC_NUM = 100
MAX_CHIRAL_TYPES = len(ChiralType.values)
MAX_HYBRIDIZATION_TYPES = len(HybridizationType.values)
MAX_NUM_HS = 5  # 4 + 1 (no hydrogens)


class AtomicNumberFeaturizer(BaseFeaturizer):
    """Class for featurizing atom by its atomic number."""

    def __init__(self) -> None:
        """Initializes AtomicNumberFeaturizer class."""
        pass

    @property
    def n_dim(self) -> int:
        """Returns dimensionality of the computed features."""
        return MAX_ATOMIC_NUM

    def get_features(self, atom: Atom) -> List[bool]:
        """Returns features of the atom."""
        return one_hot_enc(atom.GetAtomicNum() - 1, MAX_ATOMIC_NUM)


class DegreeFeaturizer(BaseFeaturizer):
    """Class for featurizing atom by its degree (excluding hydrogens) of connectivity."""

    def __init__(self) -> None:
        """Initializes DegreeFeaturizer class."""
        pass

    @property
    def n_dim(self) -> int:
        """Returns dimensionality of the computed features."""
        return 6

    def get_features(self, atom: Atom) -> List[bool]:
        """Returns features of the atom."""
        return one_hot_enc(atom.GetDegree(), self.n_dim)


class TotalDegreeFeaturizer(BaseFeaturizer):
    """Class for featurizing atom by its total degree (including hydrogens) of connectivity."""

    def __init__(self) -> None:
        """Initializes TotalDegreeFeaturizer class."""
        pass

    @property
    def n_dim(self) -> int:
        """Returns dimensionality of the computed features."""
        return 6

    def get_features(self, atom: Atom) -> List[bool]:
        # """Returns features of the atom."""
        return one_hot_enc(atom.GetTotalDegree(), self.n_dim)


class ChiralTypeFeaturizer(BaseFeaturizer):
    """Class for featurizing atom by its chirality type."""

    def __init__(self) -> None:
        """Initializes ChiralTypeFeaturizer class."""
        pass

    @property
    def n_dim(self) -> int:
        """Returns dimensionality of the computed features."""
        return MAX_CHIRAL_TYPES

    def get_features(self, atom: Atom) -> List[bool]:
        """Returns features of the atom."""
        return one_hot_enc(int(atom.GetChiralTag()), MAX_CHIRAL_TYPES)


class TotalNumHFeaturizer(BaseFeaturizer):
    """Class for featurizing atom by total number of hydrogens."""

    def __init__(self) -> None:
        """Initializes TotalNumHFeaturizer class."""
        pass

    @property
    def n_dim(self) -> int:
        """Returns dimensionality of the computed features."""
        return MAX_NUM_HS

    def get_features(self, atom: Atom) -> List[bool]:
        """Returns features of the atom."""
        return one_hot_enc(atom.GetTotalNumHs(), self.n_dim)


class HybridizationFeaturizer(BaseFeaturizer):
    """Class for featurizing atom by its hybridization type."""

    def __init__(self) -> None:
        """Initializes HybridizationFeaturizer class."""
        pass

    @property
    def n_dim(self) -> int:
        """Returns dimensionality of the computed features."""
        return MAX_HYBRIDIZATION_TYPES

    def get_features(self, atom: Atom) -> List[bool]:
        """Returns features of the atom."""
        return one_hot_enc(atom.GetHybridization(), self.n_dim)


class AromaticityFeaturizer(BaseFeaturizer):
    """Class for featurizing atom based on its aromaticity."""

    def __init__(self) -> None:
        """Initializes AromaticityFeaturizer class."""
        pass

    @property
    def n_dim(self) -> int:
        """Returns dimensionality of the computed features."""
        return 1

    def get_features(self, atom: Atom) -> List[bool]:
        """Returns features of the atom."""
        return [atom.GetIsAromatic()]


class PeriodicTableFeaturizer(BaseFeaturizer):
    """Class for featurizing atom by its position (period and group) in the periodic table."""

    def __init__(self) -> None:
        """Initializes PeriodicTableFeaturizer class."""
        self.pt = Chem.GetPeriodicTable()

    @property
    def n_dim(self) -> int:
        """Returns dimensionality of the computed features."""
        return 25

    def get_period(self, atom: Chem.Atom) -> List[bool]:
        """Returns one-hot encoded period of atom."""
        atomic_number = atom.GetAtomicNum()
        # The number of elements per period in the periodic table
        period_limits = [2, 10, 18, 36, 54, 86, 118]

        # Determine the period based on atomic number
        for period, limit in enumerate(period_limits, start=1):
            if atomic_number <= limit:
                return one_hot_enc(period - 1, 7)
        return None

    def get_group(self, atom: Chem.Atom) -> List[bool]:
        """Returns one-hot encoded group of atom."""
        group = self.pt.GetNOuterElecs(atom.GetAtomicNum())
        return one_hot_enc(group - 1, 15)

    def get_atomic_radii(self, atom: Chem.Atom) -> List[float]:
        """Computes bond radius, covalent radius, and van der Waals radius and normalizes it."""
        atomic_num = atom.GetAtomicNum()
        Rb0 = self.pt.GetRb0(atomic_num)
        Rb0 = Rb0 / 2.4  # Standard scaler

        Rcovalent = self.pt.GetRcovalent(atomic_num)
        Rcovalent = (Rcovalent - 0.28) / (2.6 - 0.28)  # Standard scaler

        Rvdw = self.pt.GetRvdw(atomic_num)
        Rvdw = (Rvdw - 1.2) / (3.0 - 1.2)

        return [Rb0, Rcovalent, Rvdw]

    def get_features(self, atom: Chem.Atom) -> List[float]:
        """Returns features of the atom."""
        return self.get_period(atom) + self.get_group(atom) + self.get_atomic_radii(atom)


class ElectronicPropertyFeaturizer(BaseFeaturizer):
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
            print(f"{data_file}")
        self.data_df = pd.read_csv(data_file).set_index("AtomicNumber")

        self.pauling_en_dict = self.data_df["Electronegativity"].to_dict()
        self.ie_dict = self.data_df["IonizationEnergy"].to_dict()
        self.ea_dict = self.data_df["ElectronAffinity"].to_dict()

    @property
    def n_dim(self) -> int:
        """Returns dimensionality of the computed features."""
        return 3

    def get_pauling_electronegativity(self, atom: Chem.Atom) -> List[float]:
        """Returns electronegativity on Pauling scale."""
        atomic_num = atom.GetAtomicNum()
        en = self.pauling_en_dict[atomic_num]
        en = (en - 0.7) / (3.98 - 0.7)
        return [en]

    def get_ie(self, atom: Chem.Atom) -> List[float]:
        """Returns ionization energy."""
        atomic_num = atom.GetAtomicNum()
        ie = self.ie_dict[atomic_num]
        ie = (ie - 3.894) / (24.587 - 3.894)
        return [ie]

    def get_ea(self, atom: Chem.Atom) -> List[float]:
        """Retuns electron affinity."""
        atomic_num = atom.GetAtomicNum()
        ea = self.ea_dict[atomic_num]
        ea = (ea - 0.079) / (3.617 - 0.079)
        return [ea]

    def get_features(self, atom: Chem.Atom) -> List[float]:
        """Returns features of the atom."""
        return self.get_pauling_electronegativity(atom) + self.get_ie(atom) + self.get_ea(atom)


class ScaffoldFeaturizer(BaseFeaturizer):
    """Class for featurizing atom based on whether it is present in Bemis-Murcko scaffold."""

    def __init__(self):
        """Initializes ScaffoldFeaturizer class."""
        pass

    @property
    def n_dim(self) -> int:
        """Returns dimensionality of the computed features."""
        return 1

    def compute_features(self, mol):
        """Computes if atom is part of scaffold."""
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold_atom_idx = set(mol.GetSubstructMatch(scaffold))

        # Set everything to False first
        for atom in mol.GetAtoms():
            atom.SetBoolProp("InScaffold", False)

        # Set only scaffold atoms to True
        for idx in scaffold_atom_idx:
            iatom = mol.GetAtomWithIdx(idx)
            iatom.SetBoolProp("InScaffold", True)

        return mol

    def get_features(self, atom) -> List[bool]:
        """Returns features of the atom."""
        return get_boolean_atomic_prop(atom, prop_list=["InScaffold"])


class SmartsFeaturizer(BaseFeaturizer):
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

    def compute_features(self, mol):
        """Computes matches by prefixed SMARTS patterns."""
        hydrogen_donor_match = sum(mol.GetSubstructMatches(self.hydrogen_donor), ())
        hydrogen_acceptor_match = sum(mol.GetSubstructMatches(self.hydrogen_acceptor), ())
        acidic_match = sum(mol.GetSubstructMatches(self.acidic), ())
        basic_match = sum(mol.GetSubstructMatches(self.basic), ())

        # Set everything to False by default
        for atom in mol.GetAtoms():
            atom.SetBoolProp("hba", False)
            atom.SetBoolProp("hbd", False)
            atom.SetBoolProp("acidic", False)
            atom.SetBoolProp("basic", False)

        # Set matching substructures to True
        for idx in hydrogen_acceptor_match:
            mol.GetAtomWithIdx(idx).SetBoolProp("hba", True)

        for idx in hydrogen_donor_match:
            mol.GetAtomWithIdx(idx).SetBoolProp("hbd", True)

        for idx in acidic_match:
            mol.GetAtomWithIdx(idx).SetBoolProp("acidic", True)

        for idx in basic_match:
            mol.GetAtomWithIdx(idx).SetBoolProp("basic", True)

        return mol

    def get_features(self, atom):
        """Returns features of the atom."""
        return get_boolean_atomic_prop(atom, prop_list=["hba", "hbd", "acidic", "basic"])


class CrippenFeaturizer(BaseFeaturizer):
    """Class for featurizing atom by Crippen logP and molar refractivity."""

    def __init__(self):
        """Initializes CrippenFeaturizer class."""
        pass

    @property
    def n_dim(self) -> int:
        """Returns dimensionality of the computed features."""
        return 2

    def compute_features(self, mol):
        """Compute atomic contributions to Crippen logP and molar refractivity."""
        logp_mr_list = np.array(rdMolDescriptors._CalcCrippenContribs(mol))
        logp_mr_list[:, 0] = np.clip(logp_mr_list[:, 0], a_min=MIN_LOGP, a_max=MAX_LOGP)
        logp_mr_list[:, 1] = np.clip(logp_mr_list[:, 1], a_min=MIN_MR, a_max=MAX_MR)

        logp_mr_list[:, 0] = (logp_mr_list[:, 0] - MIN_LOGP) / (MAX_LOGP - MIN_LOGP)
        logp_mr_list[:, 1] = (logp_mr_list[:, 1] - MIN_MR) / (MAX_MR - MIN_MR)

        for iatom, atom in enumerate(mol.GetAtoms()):
            atom.SetDoubleProp("crippen_logp", logp_mr_list[iatom][0])
            atom.SetDoubleProp("crippen_mr", logp_mr_list[iatom][1])

        return mol

    def get_features(self, atom):
        """Returns features of the atom."""
        return get_double_atomic_prop(atom, prop_list=["crippen_logp", "crippen_mr"])


# TODO Implement more features
## - Size of ring atom is present in
## - 2D partial charges like Gasteiger charges
