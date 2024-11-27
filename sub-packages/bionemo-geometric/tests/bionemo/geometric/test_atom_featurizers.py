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

from bionemo.geometric.atom_featurizers import (
    MAX_ATOMIC_NUM,
    MAX_HYBRIDIZATION_TYPES,
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
    assert anf.get_features(test_mol.GetAtomWithIdx(0)) == one_hot_enc(7 - 1, MAX_ATOMIC_NUM)  # N
    assert anf.get_features(test_mol.GetAtomWithIdx(2)) == one_hot_enc(8 - 1, MAX_ATOMIC_NUM)  # O
    assert anf.get_features(test_mol.GetAtomWithIdx(10)) == one_hot_enc(16 - 1, MAX_ATOMIC_NUM)  # S


def test_degree_featurizer(test_mol):
    df = DegreeFeaturizer()
    deg1 = df.get_features(test_mol.GetAtomWithIdx(0))
    deg1_ref = [False, True, False, False, False, False]
    assert deg1 == deg1_ref

    deg3 = df.get_features(test_mol.GetAtomWithIdx(1))
    deg3_ref = [False, False, False, True, False, False]
    assert deg3 == deg3_ref

    deg5 = df.get_features(test_mol.GetAtomWithIdx(10))
    deg5_ref = [False, False, False, False, True, False]
    assert deg5 == deg5_ref


def test_total_degree_featurizer(test_mol):
    tdf = TotalDegreeFeaturizer()

    totdeg3 = tdf.get_features(test_mol.GetAtomWithIdx(0))
    totdeg3_ref = [False, False, False, True, False, False]
    assert totdeg3 == totdeg3_ref

    totdeg1 = tdf.get_features(test_mol.GetAtomWithIdx(2))
    totdeg1_ref = [False, True, False, False, False, False]
    assert totdeg1 == totdeg1_ref

    totdeg2 = tdf.get_features(test_mol.GetAtomWithIdx(16))
    totdeg2_ref = [False, False, True, False, False, False]
    assert totdeg2 == totdeg2_ref


def test_chiral_type_featurizer(chiral_mol):
    cf = ChiralTypeFeaturizer()
    unspec_feats = cf.get_features(chiral_mol.GetAtomWithIdx(0))
    unspec_feats_ref = one_hot_enc(int(ChiralType.CHI_UNSPECIFIED), cf.n_dim)
    assert unspec_feats == unspec_feats_ref

    cw_feats = cf.get_features(chiral_mol.GetAtomWithIdx(9))
    cw_feats_ref = one_hot_enc(int(ChiralType.CHI_TETRAHEDRAL_CW), cf.n_dim)
    assert cw_feats == cw_feats_ref

    ccw_feats = cf.get_features(chiral_mol.GetAtomWithIdx(22))
    ccw_feats_ref = one_hot_enc(int(ChiralType.CHI_TETRAHEDRAL_CCW), cf.n_dim)
    assert ccw_feats == ccw_feats_ref


def test_total_numh_featurizer(test_mol):
    num_hf = TotalNumHFeaturizer()

    h2_feats = num_hf.get_features(test_mol.GetAtomWithIdx(0))
    h2_feats_ref = one_hot_enc(2, 5)
    assert h2_feats == h2_feats_ref

    h1_feats = num_hf.get_features(test_mol.GetAtomWithIdx(7))
    h1_feats_ref = one_hot_enc(1, 5)
    assert h1_feats == h1_feats_ref

    h0_feats = num_hf.get_features(test_mol.GetAtomWithIdx(1))
    h0_feats_ref = one_hot_enc(0, 5)
    assert h0_feats == h0_feats_ref


def test_hybridization_featurizer(test_mol):
    hf = HybridizationFeaturizer()

    sp2_feats = hf.get_features(test_mol.GetAtomWithIdx(1))
    sp2_feats_ref = one_hot_enc(HybridizationType.SP2, MAX_HYBRIDIZATION_TYPES)
    assert sp2_feats == sp2_feats_ref

    sp3_feats = hf.get_features(test_mol.GetAtomWithIdx(11))
    sp3_feats_ref = one_hot_enc(HybridizationType.SP3, MAX_HYBRIDIZATION_TYPES)
    assert sp3_feats == sp3_feats_ref


def test_aromaticity_featurizer(test_mol):
    af = AromaticityFeaturizer()

    non_aro_feats = af.get_features(test_mol.GetAtomWithIdx(1))
    assert non_aro_feats == [False]

    aro_feats = af.get_features(test_mol.GetAtomWithIdx(4))
    assert aro_feats == [True]

    non_aro_feats = af.get_features(test_mol.GetAtomWithIdx(22))
    assert non_aro_feats == [False]


def test_periodic_table_featurizer(test_mol):
    pt = PeriodicTableFeaturizer()

    n_feats = pt.get_features(test_mol.GetAtomWithIdx(0))  # N
    n_feats_ref = [
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        0.2916666666666667,
        0.18534482758620685,
        0.2222222222222223,
    ]
    assert (
        len(n_feats) == pt.n_dim
    ), f"Dimension of generated features ({len(n_feats)}) not equal to expected dimension ({pt.n_dim})"
    assert n_feats == n_feats_ref

    c_feats = pt.get_features(test_mol.GetAtomWithIdx(1))  # C
    c_feats_ref = [
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        0.32083333333333336,
        0.2068965517241379,
        0.2777777777777778,
    ]
    assert c_feats == c_feats_ref

    o_feats = pt.get_features(test_mol.GetAtomWithIdx(2))  # O
    o_feats_ref = [
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        0.275,
        0.16379310344827586,
        0.19444444444444448,
    ]
    assert o_feats == o_feats_ref

    s_feats = pt.get_features(test_mol.GetAtomWithIdx(10))  # S
    s_feats_ref = [
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        0.43333333333333335,
        0.3318965517241379,
        0.33333333333333337,
    ]
    assert s_feats == s_feats_ref

    cl_feats = pt.get_features(test_mol.GetAtomWithIdx(22))  # Cl
    cl_feats_ref = [
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        0.41541666666666666,
        0.3189655172413793,
        0.33333333333333337,
    ]
    assert cl_feats == cl_feats_ref


def test_electronic_property_featurizer(test_mol):
    ep = ElectronicPropertyFeaturizer()

    n_feats = ep.get_features(test_mol.GetAtomWithIdx(0))  # N
    n_feats_ref = [0.7134146341463413, 0.5141835403276471, 0.28070671308004325]
    assert n_feats == n_feats_ref

    c_feats = ep.get_features(test_mol.GetAtomWithIdx(1))  # C
    c_feats_ref = [0.5640243902439024, 0.3559657855313391, 0.33465234595816845]
    assert c_feats == c_feats_ref

    o_feats = ep.get_features(test_mol.GetAtomWithIdx(2))  # O
    o_feats_ref = [0.8353658536585366, 0.4699173633595903, 0.39061616732617305]
    assert o_feats == o_feats_ref

    s_feats = ep.get_features(test_mol.GetAtomWithIdx(10))  # S
    s_feats_ref = [0.573170731707317, 0.31247281689460205, 0.5647258338044093]
    assert s_feats == s_feats_ref

    cl_feats = ep.get_features(test_mol.GetAtomWithIdx(22))  # Cl
    cl_feats_ref = [0.7499999999999999, 0.4385057748997246, 1.0]
    assert cl_feats == cl_feats_ref


def test_scaffold_featurizer(test_mol):
    sf = ScaffoldFeaturizer()
    test_mol_featurized = sf.compute_features(test_mol)

    prop_list = []
    for atom in test_mol_featurized.GetAtoms():
        prop_list.append(sf.get_features(atom))

    prop_list_ref = [
        [False],
        [False],
        [False],
        [True],
        [True],
        [True],
        [True],
        [True],
        [True],
        [True],
        [False],
        [False],
        [False],
        [False],
        [True],
        [True],
        [True],
        [True],
        [True],
        [True],
        [True],
        [True],
        [False],
        [True],
        [True],
    ]
    assert prop_list == prop_list_ref


def test_smarts_featurizer(test_mol, acetic_acid, methylamine):
    sf = SmartsFeaturizer()

    test_mol_featurized = sf.compute_features(test_mol)
    test_mol_props = []
    for atom in test_mol_featurized.GetAtoms():
        test_mol_props.append(sf.get_features(atom))
    test_mol_props_ref = [
        [False, True, False, False],
        [False, False, False, False],
        [True, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
        [True, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
        [False, True, False, False],
        [True, False, False, False],
        [True, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
        [True, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
    ]

    assert test_mol_props == test_mol_props_ref

    # acetic acid
    aa_featurized = sf.compute_features(acetic_acid)
    aa_props = []
    for atom in aa_featurized.GetAtoms():
        aa_props.append(sf.get_features(atom))
    aa_props_ref = [
        [False, False, False, False],
        [False, False, True, False],
        [True, False, False, False],
        [False, True, False, False],
    ]
    assert aa_props == aa_props_ref

    # methylamine
    ma_featurized = sf.compute_features(methylamine)
    ma_props = []
    for atom in ma_featurized.GetAtoms():
        ma_props.append(sf.get_features(atom))

    ma_props_ref = [[False, False, False, False], [True, True, False, True]]

    assert ma_props == ma_props_ref
