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
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import toml


def find_pyproject_files(base_dir):
    """Find all pyproject.toml files in subdirectories."""
    pyproject_files = []
    for root, _, files in os.walk(base_dir):
        if "pyproject.toml" in files:
            pyproject_files.append(os.path.join(root, "pyproject.toml"))
    return pyproject_files


def parse_dependencies(pyproject_path):
    """Parse dependencies from a pyproject.toml file."""
    with open(pyproject_path, "r") as f:
        pyproject_data = toml.load(f)

    dependencies = []
    package_name = None

    # Extract package name
    try:
        package_name = pyproject_data["project"]["name"]
    except KeyError:
        print(f"Warning: Could not find package name in {pyproject_path}")

    # Extract dependencies
    try:
        deps = pyproject_data["project"]["dependencies"]
        if isinstance(deps, dict):  # If dependencies are a dictionary
            for dep, _ in deps.items():
                if dep.startswith("bionemo-"):
                    dependencies.append((dep))
        elif isinstance(deps, list):  # If dependencies are a list
            for dep in deps:
                if dep.startswith("bionemo-"):
                    dependencies.append((dep))
    except KeyError:
        print(f"Warning: Could not find dependencies in {pyproject_path}")

    return package_name, dependencies


def build_dependency_graph(base_dir):
    """Build a dependency graph for all sub-packages."""
    pyproject_files = find_pyproject_files(base_dir)
    dependency_graph = defaultdict(list)

    for pyproject_file in pyproject_files:
        package_name, dependencies = parse_dependencies(pyproject_file)
        if package_name:
            for dep in dependencies:
                dependency_graph[f"{dep}"].append(package_name)

    return dependency_graph


def visualize_dependency_graph(dependency_graph, filename):
    """Visualize the dependency graph using NetworkX."""
    G = nx.DiGraph()

    for package, dependents in dependency_graph.items():
        for dep in dependents:
            G.add_edge(package, dep)

    plt.figure(figsize=(14, 10))
    pos = nx.shell_layout(G)  # Use shell layout for better visualization of hierarchical dependencies
    # pos = nx.spiral_layout(G)

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=3000,
        node_color="lightblue",
        font_size=10,
        font_weight="bold",
        arrowsize=20,
        edge_color="gray",
    )
    plt.title("Dependency Graph", fontsize=16)
    plt.savefig(filename)


if __name__ == "__main__":
    script_path = Path(__file__).resolve()

    # Get the parent directory
    parent_directory = script_path.parent.parent

    base_dir = parent_directory / "sub-packages"
    dependency_graph = build_dependency_graph(base_dir)
    visualize_dependency_graph(dependency_graph, "dependency_graph.png")
