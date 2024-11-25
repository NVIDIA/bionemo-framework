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


import pytest
from rdkit import Chem
from rdkit.Chem.rdchem import ChiralType, HybridizationType
import numpy as np
from bionemo.geometric.atom_featurizers import (
    AromaticityFeaturizer,
    AtomicNumberFeaturizer,
    ChiralTypeFeaturizer,
    DegreeFeaturizer,
    ElectronicPropertyFeaturizer,
    HybridizationFeaturizer,
    PeriodicTableFeaturizer,
    ScaffoldFeaturizer,
    SmartsFeaturizer,
    TotalDegreeFeaturizer,
    TotalNumHFeaturizer,
)
from bionemo.geometric.base_featurizer import one_hot_enc


@pytest.fixture(scope="module")
def test_mol():
    return Chem.MolFromSmiles("NC(=O)c1cn(-c2ccc(S(N)(=O)=O)cc2)nc1-c1ccc(Cl)cc1")  # CHEMBL3126825


@pytest.fixture(scope="module")
def acetic_acid():
    return Chem.MolFromSmiles("CC(=O)O")


@pytest.fixture(scope="module")
def methylamine():
    return Chem.MolFromSmiles("CN")


@pytest.fixture(scope="module")
def chiral_mol():
    return Chem.MolFromSmiles("Cn1cc(C(=O)N2CC[C@@](O)(c3ccccc3)[C@H]3CCCC[C@@H]32)ccc1=O")

def test_atomic_num_featurizer(test_mol):
    anf = AtomicNumberFeaturizer()
    anf_feats = anf(test_mol)
    anf_feats_ref = [7, 6, 8, 6, 6, 7, 6, 6, 6, 6, 16, 7, 8, 8, 6, 6, 7, 6, 6, 6, 6, 6, 17, 6, 6]
    assert anf_feats == anf_feats_ref

def test_degree_featurizer(test_mol):
    df = DegreeFeaturizer()
    df_feats = df(test_mol)

    df_feats_ref = [1, 3, 1, 3, 2, 3, 3, 2, 2, 3, 4, 1, 1, 1, 2, 2, 2, 3, 3, 2, 2, 3, 1, 2, 2]

    assert df_feats == df_feats_ref


def test_total_degree_featurizer(test_mol):
    tdf = TotalDegreeFeaturizer()

    tdf_feats = tdf(test_mol)
    tdf_feats_ref = [3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 4, 3, 1, 1, 3, 3, 2, 3, 3, 3, 3, 3, 1, 3, 3]
    assert tdf_feats == tdf_feats_ref

def test_chiral_type_featurizer(chiral_mol):
    cf = ChiralTypeFeaturizer()

    cf_feats = cf(chiral_mol)
    cf_feats_ref = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0]
    assert cf_feats == cf_feats_ref


def test_total_numh_featurizer(test_mol):
    num_hf = TotalNumHFeaturizer()

    h2_feats = num_hf(test_mol)
    h2_feats_ref = [2, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 2, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1]
    assert h2_feats == h2_feats_ref


def test_hybridization_featurizer(test_mol, chiral_mol):
    hf = HybridizationFeaturizer()

    hf_feats = hf(test_mol)
    hf_feats_ref = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 3, 3]
    assert hf_feats == hf_feats_ref

def test_aromaticity_featurizer(test_mol):
    af = AromaticityFeaturizer()
    af_feats = af(test_mol)
    af_feats_ref = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]
    assert af_feats == af_feats_ref

def test_periodic_table_featurizer(test_mol):
    pt = PeriodicTableFeaturizer()

    pt_feats = pt(test_mol)
    pt_feats_ref = [[2, 5], [2, 4], [2, 6], [2, 4], [2, 4], [2, 5], [2, 4], [2, 4], [2, 4], [2, 4], [3, 6], [2, 5], [2, 6], [2, 6], [2, 4], [2, 4], [2, 5], [2, 4], [2, 4], [2, 4], [2, 4], [2, 4], [3, 7], [2, 4], [2, 4]]

    assert pt_feats == pt_feats_ref

def test_electronic_property_featurizer(test_mol):
    ep = ElectronicPropertyFeaturizer()

    ep_feats = ep(test_mol)
    ep_feats_ref = np.array(
        [[3.04, 14.534, 1.0721403509],
        [ 2.55, 11.26,  1.263       ],
        [ 3.44, 13.618, 1.461       ],
        [ 2.55, 11.26,  1.263       ],
        [ 2.55, 11.26,  1.263       ],
        [ 3.04, 14.534, 1.0721403509],
        [ 2.55, 11.26,  1.263       ],
        [ 2.55, 11.26,  1.263       ],
        [ 2.55, 11.26,  1.263       ],
        [ 2.55, 11.26,  1.263       ],
        [ 2.58, 10.36,  2.077       ],
        [ 3.04, 14.534, 1.0721403509],
        [ 3.44, 13.618, 1.461       ],
        [ 3.44, 13.618, 1.461       ],
        [ 2.55, 11.26,  1.263       ],
        [ 2.55, 11.26,  1.263       ],
        [ 3.04, 14.534, 1.0721403509],
        [ 2.55, 11.26,  1.263       ],
        [ 2.55, 11.26,  1.263       ],
        [ 2.55, 11.26,  1.263       ],
        [ 2.55, 11.26,  1.263       ],
        [ 2.55, 11.26,  1.263       ],
        [ 3.16, 12.968, 3.617       ],
        [ 2.55, 11.26,  1.263       ],
        [ 2.55, 11.26,  1.263       ],])

    assert np.all(np.isclose(ep_feats, ep_feats_ref))

def test_scaffold_featurizer(test_mol):
    sf = ScaffoldFeaturizer()
    sf_feats = sf(test_mol)
    sf_feats_ref = [False, False, False, True, True, True, True, True, True, True, False, False, False, False, True, True, True, True, True, True, True, True, False, True, True]
    assert sf_feats == sf_feats_ref

def test_smarts_featurizer(test_mol, acetic_acid, methylamine):
    sf = SmartsFeaturizer()
    sf_feats = sf(test_mol)
    sf_feats_ref = [[True, False, False, False], [False, False, False, False], [False, True, False, False], [False, False, False, False], [False, False, False, False], [False, True, False, False], [False, False, False, False], [False, False, False, False], [False, False, False, False], [False, False, False, False], [False, False, False, False], [True, False, False, False], [False, True, False, False], [False, True, False, False], [False, False, False, False], [False, False, False, False], [False, True, False, False], [False, False, False, False], [False, False, False, False], [False, False, False, False], [False, False, False, False], [False, False, False, False], [False, False, False, False], [False, False, False, False], [False, False, False, False]]
    assert sf_feats == sf_feats_ref

    sf_feats = sf(acetic_acid)
    sf_feats_ref = [[False, False, False, False], [False, False, True, False], [False, True, False, False], [True, False, False, False]]
    assert sf_feats == sf_feats_ref

    sf_feats = sf(methylamine)
    sf_feats_ref = [[False, False, False, False], [True, True, False, True]]
    assert sf_feats == sf_feats_ref
