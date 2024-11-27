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

from bionemo.geometric.base_featurizer import one_hot_enc
from bionemo.geometric.bond_featurizers import (
    BondConjugationFeaturizer,
    BondStereochemistryFeaturizer,
    BondTypeFeaturizer,
    RingFeaturizer,
)


@pytest.fixture(scope="module")
def sample_mol2():
    return Chem.MolFromSmiles("C[C@H]1CN(c2ncnc3[nH]cc(-c4cccc(F)c4)c23)CCO1")  # CHEMBL3927167


@pytest.fixture(scope="module")
def sample_mol3():
    return Chem.MolFromSmiles("N#C[C@H]1CN(c2ncnc3[nH]cc(-c4cccc(F)c4)c23)CCO1")


@pytest.fixture(scope="module")
def stereoe_mol():
    return Chem.MolFromSmiles(r"F/C=C/F")


@pytest.fixture(scope="module")
def stereoz_mol():
    return Chem.MolFromSmiles(r"F/C=C\F")


def test_bond_type_featurizer(sample_mol3):
    btf = BondTypeFeaturizer()
    triple_feats = btf.get_features(sample_mol3.GetBondWithIdx(0))
    triple_feats_ref = one_hot_enc(3, 4)
    assert triple_feats == triple_feats_ref

    single_feats = btf.get_features(sample_mol3.GetBondWithIdx(1))
    single_feats_ref = one_hot_enc(0, 4)
    assert single_feats == single_feats_ref

    aromatic_feats = btf.get_features(sample_mol3.GetBondWithIdx(5))
    aromatic_feats_ref = one_hot_enc(1, 4)
    assert aromatic_feats == aromatic_feats_ref


def test_bond_conjugation_featurizer(sample_mol2):
    bcf = BondConjugationFeaturizer()

    assert bcf.get_features(sample_mol2.GetBondWithIdx(0)) == [False]
    assert bcf.get_features(sample_mol2.GetBondWithIdx(3)) == [True]


def test_bond_stereochemistry_featurizer(stereoe_mol, stereoz_mol):
    bsf = BondStereochemistryFeaturizer()
    stereoe_feats = bsf.get_features(stereoe_mol.GetBondWithIdx(1))
    stereoe_feats_ref = one_hot_enc(3, 6)
    assert stereoe_feats == stereoe_feats_ref
    stereoz_feats = bsf.get_features(stereoz_mol.GetBondWithIdx(1))
    stereoz_feats_ref = one_hot_enc(2, 6)
    assert stereoz_feats == stereoz_feats_ref


def test_ring_featurizer(sample_mol2):
    rf = RingFeaturizer()
    feat_mol = rf.compute_features(sample_mol2)

    bidx0_feats = rf.get_features(feat_mol.GetBondWithIdx(0))
    bidx0_feats_ref = [False, False, False, False, False, False, False]
    assert bidx0_feats == bidx0_feats_ref

    bidx1_feats = rf.get_features(feat_mol.GetBondWithIdx(1))
    bidx1_feats_ref = [False, False, False, True, False, False, False]
    assert bidx1_feats == bidx1_feats_ref

    bidx24_feats = rf.get_features(feat_mol.GetBondWithIdx(24))
    bidx24_feats_ref = [False, False, True, True, False, False, False]
    assert bidx24_feats == bidx24_feats_ref
