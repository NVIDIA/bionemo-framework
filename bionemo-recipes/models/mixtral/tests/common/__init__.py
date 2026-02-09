# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Common test utilities for BioNeMo models.

This package provides reusable test infrastructure following HuggingFace
transformers patterns, including:

- BioNeMoModelTester: Abstract base class for model-specific test configuration
- BioNeMoModelTest: Base test class with all common test methods
- TestTolerances: Dataclass for model-specific numerical tolerances
- Distributed testing utilities for multi-GPU tests
- Shared fixtures for common test requirements

Example usage:

    ```python
    from tests.common import BioNeMoModelTester, BioNeMoModelTest, TestTolerances

    class ESM2ModelTester(BioNeMoModelTester):
        def get_model_class(self):
            return NVEsmForMaskedLM
        # ... implement other abstract methods
    ```
"""

from .test_modeling_common import BaseModelTest, TestTolerances


__all__ = [
    "BaseModelTest",
    "TestTolerances",
]
