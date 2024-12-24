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


import re


with open("documentation.md", "r") as file:
    lines = file.readlines()

# Delete lines that start with "  * " and "    * "
lines = [line for line in lines if not line.startswith("  * ") and not line.startswith("    * ")]

# Join the lines back into a string
markdown = "".join(lines)

# Replace dots with no space in anchor ids
markdown = re.sub(r'<a id="([a-zA-Z0-9_\.]+)">', lambda match: f'<a id="{match.group(1).replace(".", "")}">', markdown)

# Replace dots with no space in links
markdown = re.sub(
    r"\[([^\]]+)\]\(#([a-zA-Z0-9_\.]+)\)",
    lambda match: f'[{match.group(1)}](#{match.group(2).replace(".", "")})',
    markdown,
)

# Replace 'moco.' with 'bionemo.moco.'
markdown = re.sub(r"moco\.", "bionemo.moco.", markdown)

with open("documentation.md", "w") as file:
    file.write(markdown)
