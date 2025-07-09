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

# Hold a certain amount of memory until the script is killed
import time


def hold_memory(gb):
    print(f"Allocating {gb} GB of memory...")
    blocks = []
    block_size = 1024 * 1024 * 100  # 100 MB
    num_blocks = int(gb * 1024 / 100)
    for _ in range(num_blocks):
        blocks.append(bytearray(block_size))
    print("Holding memory... Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Releasing memory...")


hold_memory(100)  # change 4 to however many GB you want to simulate
