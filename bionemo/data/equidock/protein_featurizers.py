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


#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from dgllife.utils import one_hot_encoding
from nemo.utils import logging


RESNAME_3_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "HIP": "H",
    "HIE": "H",
    "TPO": "T",
    "HID": "H",
    "LEV": "L",
    "MEU": "M",
    "PTR": "Y",
    "GLV": "E",
    "CYT": "C",
    "SEP": "S",
    "HIZ": "H",
    "CYM": "C",
    "GLM": "E",
    "ASQ": "D",
    "TYS": "Y",
    "CYX": "C",
    "GLZ": "G",
}


def residue_type_one_hot_dips(residue):
    dit = RESNAME_3_1
    allowable_set = [
        "Y",
        "R",
        "F",
        "G",
        "I",
        "V",
        "A",
        "W",
        "E",
        "H",
        "C",
        "N",
        "M",
        "D",
        "T",
        "S",
        "K",
        "L",
        "Q",
        "P",
    ]
    res_name = residue
    if res_name not in dit.keys():
        res_name = None
    else:
        res_name = dit[res_name]
    return one_hot_encoding(res_name, allowable_set, encode_unknown=True)


def residue_type_one_hot_dips_not_one_hot(residue):
    dit = RESNAME_3_1

    rare_residues = {
        "HIP": "H",
        "HIE": "H",
        "TPO": "T",
        "HID": "H",
        "LEV": "L",
        "MEU": "M",
        "PTR": "Y",
        "GLV": "E",
        "CYT": "C",
        "SEP": "S",
        "HIZ": "H",
        "CYM": "C",
        "GLM": "E",
        "ASQ": "D",
        "TYS": "Y",
        "CYX": "C",
        "GLZ": "G",
    }

    if residue in rare_residues.keys():
        logging.info("Encountered some rare residue: ", residue)

    indicator = {
        "Y": 0,
        "R": 1,
        "F": 2,
        "G": 3,
        "I": 4,
        "V": 5,
        "A": 6,
        "W": 7,
        "E": 8,
        "H": 9,
        "C": 10,
        "N": 11,
        "M": 12,
        "D": 13,
        "T": 14,
        "S": 15,
        "K": 16,
        "L": 17,
        "Q": 18,
        "P": 19,
    }
    res_name = residue
    if res_name not in dit.keys():
        return 20
    else:
        res_name = dit[res_name]
        return indicator[res_name]
