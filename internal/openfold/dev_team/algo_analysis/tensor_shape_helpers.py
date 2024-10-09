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


from typing import Tuple


def num_elements(input_shape: Tuple[int]):
    out = 1
    for i in range(len(input_shape)):
        out *= input_shape[i]
    return out


def shape_excluding(input_shape: Tuple[int], excluded_axes=None):
    number_of_axes = len(input_shape)
    excluded_axes_modded = [i % number_of_axes for i in excluded_axes] if excluded_axes is not None else []
    shape_other_components = [input_shape[i] for i in range(len(input_shape)) if i not in excluded_axes_modded]
    return shape_other_components


def num_elements_excluding(input_shape: Tuple[int], excluded_axes=None):
    shape_other_components = shape_excluding(input_shape, excluded_axes)
    print(shape_other_components)
    return num_elements(shape_other_components)
