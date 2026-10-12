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

"""Release packaging and recipe integration for the Stitch example skill."""

import json
import re
import shutil
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).parents[2]
SKILL_ROOT = REPO_ROOT / "skills" / "nemotron-stitch-extend-example"


def test_discovery():
    """The root alias exposes the only canonical Stitch skill package."""
    discovered = REPO_ROOT / ".agents" / "skills" / SKILL_ROOT.name
    assert discovered.resolve() == SKILL_ROOT.resolve()
    assert (discovered / "SKILL.md").is_file()
    assert not (REPO_ROOT / "recipes" / "nemotron-stitch" / "skills").exists()


def test_checkout_references():
    """All required recipe entrypoints named by the locator exist."""
    instructions = (SKILL_ROOT / "SKILL.md").read_text()
    references = re.findall(r"`(recipes/nemotron-stitch/[^`]+)`", instructions)
    assert references
    for reference in references:
        path = (REPO_ROOT / reference).resolve()
        assert path.is_relative_to(REPO_ROOT / "recipes" / "nemotron-stitch")
        assert path.is_file(), reference


def test_installed_package(tmp_path):
    """Copying the portable skill preserves metadata, evals, and card links."""
    installed = tmp_path / SKILL_ROOT.name
    shutil.copytree(SKILL_ROOT, installed)
    metadata = yaml.safe_load((installed / "SKILL.md").read_text().split("---", 2)[1])
    cases = json.loads((installed / "evals" / "evals.json").read_text())
    assert metadata["name"] == installed.name == cases["skill_name"]
    assert metadata["description"]
    ids = [case["id"] for case in cases["evals"]]
    assert ids and len(ids) == len(set(ids))
    for case in cases["evals"]:
        assert case["expected_skill"] == metadata["name"]
        assert case["prompt"] and case["expected_output"]
        assert isinstance(case["assertions"], list) and case["assertions"]
        assert all(isinstance(assertion, str) and assertion.strip() for assertion in case["assertions"])
        if case["expected_script"] is not None:
            assert (installed / case["expected_script"]).is_file()
    links = re.findall(r"\[[^\]]+\]\(([^)]+)\)", (installed / "skill-card.md").read_text())
    local_links = [link for link in links if "://" not in link]
    assert local_links
    for link in local_links:
        target = (installed / link).resolve()
        assert target.is_relative_to(installed)
        assert target.is_file(), link
