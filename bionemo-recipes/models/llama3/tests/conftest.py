# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import sys
from pathlib import Path

import pytest
from transformer_engine.common import recipe as recipe_module
from transformer_engine.pytorch import fp8


sys.path.append(Path(__file__).parent.parent.as_posix())
sys.path.append(Path(__file__).parent.as_posix())


@pytest.fixture(scope="session")
def recipe_path() -> Path:
    """Return the root directory of the recipe."""
    return Path(__file__).parent.parent


ALL_RECIPES = [
    recipe_module.DelayedScaling(),
    recipe_module.Float8CurrentScaling(),
    recipe_module.Float8BlockScaling(),
    recipe_module.MXFP8BlockScaling(),
    # recipe_module.NVFP4BlockScaling(disable_rht=True, disable_stochastic_rounding=True),
]


def _check_recipe_support(recipe: recipe_module.Recipe):
    """Check if a recipe is supported and return (supported, reason)."""
    if isinstance(recipe, recipe_module.DelayedScaling):
        recipe_supported, reason = fp8.check_fp8_support()
    elif isinstance(recipe, recipe_module.Float8CurrentScaling):
        recipe_supported, reason = fp8.check_fp8_support()
    elif isinstance(recipe, recipe_module.Float8BlockScaling):
        recipe_supported, reason = fp8.check_fp8_block_scaling_support()
    elif isinstance(recipe, recipe_module.MXFP8BlockScaling):
        recipe_supported, reason = fp8.check_mxfp8_support()
    elif isinstance(recipe, recipe_module.NVFP4BlockScaling):
        recipe_supported, reason = fp8.check_nvfp4_support()
    else:
        recipe_supported = False
        reason = "Unsupported recipe"
    return recipe_supported, reason


def parametrize_recipes_with_support(recipes):
    """Generate pytest.param objects with skip marks for unsupported recipes."""
    parametrized_recipes = []
    for recipe in recipes:
        recipe_supported, reason = _check_recipe_support(recipe)
        parametrized_recipes.append(
            pytest.param(
                recipe,
                id=recipe.__class__.__name__,
                marks=pytest.mark.skipif(
                    not recipe_supported,
                    reason=reason,
                ),
            )
        )
    return parametrized_recipes


@pytest.fixture(params=parametrize_recipes_with_support(ALL_RECIPES))
def fp8_recipe(request):
    return request.param
