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


# conftest.py
import gc

import pytest
import torch

from bionemo.testing.test_tools import get_device_and_memory_usage

def pytest_sessionstart(session):
    """Called at the start of the test session."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        print(
            f"""
            sub-packages/bionemo-evo2/tests/bionemoe/evo2: Starting test session
            {get_device_and_memory_usage()}
            """
        )
        


def pytest_sessionfinish(session, exitstatus):
    """Called at the end of the test session."""
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated()
        final_memory = torch.cuda.memory_allocated()
        print(
            f"""
            sub-packages/bionemo-evo2/tests/bionemoe/evo2: Test session complete
            current_device={torch.cuda.current_device()}
            current device name: {torch.cuda.get_device_name(torch.cuda.current_device())}
            gb_available={torch.cuda.mem_get_info()[0] / 1024**3}
            Peak GPU memory: {peak_memory / 1024**3:.3f} GB
            Final GPU memory: {final_memory / 1024**3:.3f} GB
            """
        )
 
 

     
 
        

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Clean up GPU memory after each test."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
