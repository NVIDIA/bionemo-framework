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


import pandas as pd

from bionemo.dnadl.tools.genome_editing import create_personal_sequence
from bionemo.dnadl.tools.genome_interval import GenomeInterval


def test_mutate():
    sequence = "AAAAAAAAAAAAAAAAAAAA"
    interval = GenomeInterval("chr1", 1, 10)
    variants_df = pd.DataFrame([{"POS": 5, "REF": "A", "ALT": "G"}, {"POS": 2, "REF": "AA", "ALT": "TC"}])

    assert create_personal_sequence(sequence, interval, variants_df) == "TCAGAAAAAAAAAAAAAAAA"
