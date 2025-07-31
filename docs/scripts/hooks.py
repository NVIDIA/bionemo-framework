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


import os
import shutil


def copy_interactives(config, **kwargs):
    """Copy over interactive webpage content."""
    site_dir = config["site_dir"]
    src = "interactives/static"
    dst = os.path.join(site_dir, "interactives")

    if os.path.exists(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)

        # Fix absolute paths in HTML files
        for root, dirs, files in os.walk(dst):
            for file in files:
                if file.endswith(".html"):
                    filepath = os.path.join(root, file)
                    with open(filepath, "r") as f:
                        content = f.read()

                    # Replace absolute paths with relative ones
                    content = content.replace('href="/assets/', 'href="./assets/')
                    content = content.replace('src="/assets/', 'src="./assets/')

                    with open(filepath, "w") as f:
                        f.write(content)
