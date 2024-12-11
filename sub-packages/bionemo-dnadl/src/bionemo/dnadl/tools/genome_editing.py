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

from bionemo.dnadl.tools.genome_interval import GenomeInterval


def create_personal_sequence(ref_sequence: str, interval: GenomeInterval, variants: pd.DataFrame) -> str:
    """Create a personal sequence by introducing mutations in the reference sequence."""
    # TO:DO Implement in a more efficient way as to not bottleneck dataloaders
    personal_sequence = ref_sequence
    for row in variants.itertuples():
        personal_sequence = _mutate(personal_sequence, interval, row.ALT, row.REF, row.POS)

    return personal_sequence


def _mutate(
    sequence: str,
    interval: GenomeInterval,
    alternate: str,
    reference: str,
    position: int,
) -> str:
    """Introduce a mutation (substitution) in one or more bases of the sequence."""
    if len(alternate) != len(reference):
        # TO:DO Insertion or deletion to be supported later
        return sequence

    relative_position = position - interval.start - 1

    if sequence[relative_position : relative_position + len(reference)].upper() != reference:
        raise ValueError(
            f"Reference sequence does not match the sequence at the specified position {relative_position}"
        )

    return sequence[:relative_position] + alternate + sequence[relative_position + len(reference) :]
