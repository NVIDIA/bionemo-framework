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
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
from bionemo.geometric.molecule_featurizers import RDkit2DDescriptorFeaturizer
import re

@pytest.fixture(scope="module")
def sample_mol():
    return Chem.MolFromSmiles("NC(=O)c1cn(-c2ccc(S(N)(=O)=O)cc2)nc1-c1ccc(Cl)cc1")  # CHEMBL3126825


@pytest.fixture(scope="module")
def sample_mol2():
    return Chem.MolFromSmiles("C[C@H]1CN(c2ncnc3[nH]cc(-c4cccc(F)c4)c23)CCO1")  # CHEMBL3927167


def test_rdkit2d_descriptor_featurizer(sample_mol, sample_mol2):
    rdf = RDkit2DDescriptorFeaturizer()
    mol_feats = rdf(sample_mol)

    # separate out int and float descriptors
    int_desc_idx = [idx for idx, (name, _) in enumerate(Descriptors.descList) if re.search(r"(num|fr_|count)", name, re.IGNORECASE)]
    float_desc_idx = list(set(range(len(Descriptors.descList))) - set(int_desc_idx))

    # 2D RDkit descriptors listed in rdkit.Chem.Descriptors.descList
    mol_feats_ref = torch.Tensor(
        [
            11.739234088578126,
            11.739234088578126,
            0.017170781893004028,
            -3.781515243079616,
            0.7220230649240628,
            11.44,
            376.82500000000005,
            363.7210000000001,
            376.039688956,
            128,
            0,
            0.25206014374186,
            -0.36548056472166857,
            0.36548056472166857,
            0.25206014374186,
            1.04,
            1.64,
            2.16,
            35.49569200759094,
            10.087164052685537,
            2.1660855035461717,
            -2.027864781099429,
            2.2489599335785333,
            -2.116891175408362,
            7.8876058111792995,
            0.10006104487344632,
            3.002802610441619,
            2.0768149956684177,
            1041.9888121550669,
            18.189869965382485,
            12.756237786950482,
            14.328663313896664,
            11.753038761063381,
            7.041763285225907,
            8.966047236156697,
            5.246005089352769,
            7.243863927292341,
            3.5271517413640963,
            4.708359290115149,
            2.2960288277468415,
            2.937456811390436,
            -2.679999999999999,
            443165.6204594159,
            17.153053800207758,
            6.320498570343795,
            3.553955393718707,
            148.42311271634844,
            5.733667477162185,
            5.693927994848461,
            0.0,
            10.023291153407584,
            5.907179729351506,
            0.0,
            4.794537184071822,
            18.238573657082064,
            5.098681808301038,
            0.0,
            23.733674027155736,
            36.39820241076966,
            16.782928377051398,
            16.146321241898335,
            13.212334168400758,
            27.531410772991606,
            0.0,
            9.780484743446223,
            10.872641214770127,
            4.895483475517775,
            0.0,
            65.31386492474428,
            0.0,
            16.944765761229018,
            10.872641214770127,
            0.0,
            0.0,
            11.600939890232516,
            24.105461457126665,
            10.023291153407584,
            0.0,
            10.357988675768818,
            59.623263594823726,
            5.022633313741326,
            16.944765761229018,
            0.0,
            121.07000000000001,
            15.930470882759089,
            13.212334168400758,
            0.0,
            10.45893496721477,
            21.967399074970345,
            0.0,
            35.1441147806047,
            24.26546827384644,
            0.0,
            5.098681808301038,
            22.473581105002644,
            24.089632667996874,
            5.877622127433703,
            11.722063306685122,
            10.022775328080856,
            7.306696335665342,
            -0.627448034769462,
            12.600266576990965,
            1.4843513794406649,
            0.0,
            -3.781515243079616,
            0.0,
            25,
            4,
            7,
            0,
            0,
            0,
            2,
            1,
            3,
            5,
            2,
            9,
            4,
            0,
            0,
            0,
            3,
            1.9389999999999998,
            93.90109999999999,
            0,
            0,
            0,
            0,
            0,
            2,
            0,
            0,
            0,
            0,
            1,
            1,
            0,
            0,
            0,
            2,
            0,
            2,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            2,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
    )

    assert torch.allclose(mol_feats[float_desc_idx], mol_feats_ref[float_desc_idx])
    assert torch.all(mol_feats[int_desc_idx] == mol_feats_ref[int_desc_idx])

    mol_feats = rdf(sample_mol2)
    mol_feats_ref = torch.Tensor(
        [
            13.59718565265102,
            13.59718565265102,
            0.1577100655076844,
            -0.2544035021415971,
            0.790191047029685,
            18.52173913043478,
            312.3480000000001,
            295.212,
            312.138639384,
            118,
            0,
            0.14300203502449213,
            -0.3748311129862186,
            0.3748311129862186,
            0.14300203502449213,
            1.3478260869565217,
            2.260869565217391,
            3.0869565217391304,
            19.142144572357893,
            10.056111934300976,
            2.2213633877518912,
            -2.319743090015102,
            2.338456916736118,
            -2.4119088332112777,
            6.006799652275313,
            0.05302438950277102,
            2.9869932884921298,
            1.8682589828156948,
            847.4805792241272,
            15.81119030894213,
            12.79062577785999,
            12.79062577785999,
            11.220346690612276,
            7.667349370161911,
            7.667349370161911,
            5.7610037335759205,
            5.7610037335759205,
            4.1260534814726375,
            4.1260534814726375,
            3.0806881394164973,
            3.0806881394164973,
            -2.4699999999999993,
            351363.0075186177,
            14.143261835899917,
            5.624189240497252,
            2.5482703016171846,
            132.67641196129384,
            14.620751205597735,
            23.609580914413193,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            14.358372089569237,
            0.0,
            0.0,
            12.13273413692322,
            24.619922828310838,
            24.849807875135223,
            18.097072566726016,
            9.127278001474869,
            16.85126421306755,
            0.0,
            14.951935562841626,
            0.0,
            13.027703587438927,
            24.59630450718855,
            42.606852761269955,
            0.0,
            11.126902983393991,
            4.899909730850478,
            10.208277825509848,
            0.0,
            0.0,
            40.75229672692799,
            4.736862953800049,
            5.817220841045895,
            6.923737199690624,
            36.78963192022406,
            0.0,
            22.160304418626517,
            0.0,
            54.040000000000006,
            0.0,
            4.39041504767482,
            0.0,
            11.921187228794198,
            6.606881964512918,
            41.067680008286686,
            12.13273413692322,
            12.393687143226153,
            12.263210640074686,
            26.775582493382725,
            4.736862953800049,
            19.208401570721648,
            0.0,
            14.165834278155707,
            0.9233333333333335,
            2.482647156084657,
            0.6119494834971028,
            6.578309193078312,
            3.5794837333081375,
            4.283374585154437,
            0.0,
            0.29411764705882354,
            23,
            1,
            5,
            0,
            1,
            1,
            1,
            2,
            3,
            4,
            1,
            6,
            2,
            0,
            1,
            1,
            4,
            2.9891000000000014,
            86.90970000000003,
            0,
            0,
            0,
            0,
            0,
            3,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            3,
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
    )

    assert torch.allclose(mol_feats[float_desc_idx], mol_feats_ref[float_desc_idx])
    assert torch.all(mol_feats[int_desc_idx] == mol_feats_ref[int_desc_idx])
