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

import re
import subprocess
import textwrap
from pathlib import Path

from omegaconf import OmegaConf


def render_launcher_string(script: str, all_config_json: str, template: str = "convergence_tests") -> str:
    """Renders the wrapper shell script template with the provided script and config JSON.

    Args:
        script (str): The shell script to be embedded into the template.
        all_config_json (str): The full configuration in JSON format to be injected into the template.
        template (str): Template style - "convergence_tests" or "scdl_performance".

    Returns:
        str: The rendered shell script with the script and config JSON substituted in place.
    """
    # Map template names to their launcher directories
    template_paths = {
        "convergence_tests": Path(__file__).parent.parent / "model_convergence" / "launchers" / "convergence_tests.sh",
        "scdl_performance": Path(__file__).parent.parent / "scdl_performance" / "launchers" / "scdl_performance.sh",
    }

    template_path = template_paths.get(template)

    if not template_path or not template_path.exists():
        raise ValueError(f"Template not found: {template}. Valid options: 'convergence_tests', 'scdl_performance'")

    tpl = template_path.read_text(encoding="utf-8")
    script_indented = textwrap.indent(script.rstrip("\n"), "  ")
    return tpl.replace("__SCRIPT__", script_indented).replace("__ALL_CONFIG_JSON__", all_config_json)


def register_resolvers():
    """Registers custom OmegaConf resolvers for use in configuration files.

    This function currently registers the "sanitize" resolver, which replaces
    any character in a string that is not alphanumeric, an underscore, or a dash
    with a dash ("-"). This is useful for generating safe strings for use in
    resource names, file paths, etc.

    Example:
        ${sanitize:some/unsafe value!} -> "some-unsafe-value-"
    """

    def sanitize(value: str) -> str:
        # Replace all forbidden characters `/ \ # ? % :` with '-'
        return re.sub(r"[\/\\#\?\%:_]", "-", value).lower()

    def gitsha():
        try:
            args = ["git", "rev-parse", "--short", "HEAD"]
            return subprocess.check_output(args, text=True).strip()
        except Exception:
            return "unknown"

    def multiply(a, b):
        return int(a) * int(b)

    OmegaConf.register_new_resolver("sanitize", sanitize)
    OmegaConf.register_new_resolver("gitsha", gitsha)
    OmegaConf.register_new_resolver("multiply", multiply)
