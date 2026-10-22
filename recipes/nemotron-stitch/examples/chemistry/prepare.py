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

"""Encode the bundled SMILES examples through a KERMT perception endpoint."""

from __future__ import annotations

import argparse
import json
import os
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

from nemotron_stitch.features.manifest import load_manifest
from nemotron_stitch.features.preparation import Partition, prepare_features
from nemotron_stitch.provenance import sha256_json

LAYERS = ("molecule_fingerprint", "atom_concat", "bond_from_atom", "bond_from_bond")


def prepare(
    client: Any, source: str | Path = "data/sanity.jsonl", output: str | Path = "artifacts/data"
) -> dict[str, Any]:
    """Publish a padded graph cache and three training-stage manifests."""
    from rdkit import Chem

    rows = load_manifest(source)

    @cache
    def encode(smiles: str) -> tuple[np.ndarray, dict[str, Any]]:
        molecule_graph = Chem.MolFromSmiles(smiles)
        if molecule_graph is None:
            raise ValueError(f"invalid SMILES: {smiles!r}")
        canonical = Chem.MolToSmiles(molecule_graph, canonical=True, isomericSmiles=True)
        molecule_graph = Chem.MolFromSmiles(canonical)
        # KERMT sorts low/high atom pairs and emits each forward/reverse pair together.
        endpoints = sorted(
            tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))) for bond in molecule_graph.GetBonds()
        )
        reply = client.forward(smiles=canonical, outputs=[{"layer": layer} for layer in LAYERS])
        molecule, atoms, bond_atom, bond_bond = [reply.tensor(f"{layer}.output").numpy_float32() for layer in LAYERS]
        if molecule.shape != (1, 3200) or atoms.ndim != 2 or atoms.shape[1] != 1600 or len(atoms) < 1:
            raise ValueError("KERMT must return molecule [1,3200] and atoms [N,1600]")
        if bond_atom.ndim != 2 or bond_atom.shape[1] != 800 or bond_bond.shape != bond_atom.shape or len(bond_atom) % 2:
            raise ValueError("KERMT must return two matching [2E,800] directed-bond branches")
        for descriptor, layer in zip(reply.tensor_metadata, LAYERS, strict=True):
            if descriptor["resolved_layer"] != layer or descriptor["pooling"] != {"kind": "none"}:
                raise ValueError("KERMT response does not match the requested unpooled layers")
        if len(atoms) != molecule_graph.GetNumAtoms() or len(bond_atom) != 2 * len(endpoints):
            raise ValueError("KERMT features disagree with the canonical graph geometry")
        packed = np.zeros((1 + len(atoms) + len(bond_atom), 3200), dtype=np.float32)
        packed[:1] = molecule
        packed[1 : 1 + len(atoms), :1600] = atoms
        packed[1 + len(atoms) :, :1600] = np.concatenate((bond_atom, bond_bond), axis=1)
        if not np.isfinite(packed).all():
            raise ValueError("KERMT features must be finite")
        return packed, dict(
            encoder_smiles=canonical,
            atom_count=len(atoms),
            directed_bond_count=len(bond_atom),
            bond_endpoints=endpoints,
        )

    def encode_inputs(inputs: tuple[str, ...]) -> np.ndarray:
        if not inputs:
            return np.empty((0, 3200), dtype=np.float32)
        graphs = [encode(smiles) for smiles in inputs]
        return np.concatenate(
            [packed[:1] for packed, _ in graphs]
            + [packed[1 : 1 + graph["atom_count"]] for packed, graph in graphs]
            + [packed[1 + graph["atom_count"] :] for packed, graph in graphs]
        )

    def normalize(
        row: dict[str, Any], *, configuration: str, source_index: int
    ) -> tuple[dict[str, Any], tuple[str, ...]]:
        inputs = []
        messages = []
        for message in row["messages"]:
            content = message["content"]
            if isinstance(content, list):
                content = [dict(block) for block in content]
                for block in content:
                    if block.get("type") == "molecule":
                        inputs.append(block["smiles"])
                        block.update(encode(block["smiles"])[1])
            messages.append({**message, "content": content})
        return {
            "sample_id": f"{configuration}-{row['sample_id']}",
            "prompt": "",
            "messages": messages,
            "target": row.get("target", "\\boxed{" + str(row["answer"]) + "}"),
            "reasoning_content": row.get("reasoning_content", ""),
            "question": "",
            "ground_truth": str(row["answer"]),
            "chat_template_kwargs": row.get(
                "chat_template_kwargs", {"enable_thinking": bool(row.get("reasoning_content"))}
            ),
            "molecule_count": len(inputs),
            "atom_count": sum(encode(smiles)[1]["atom_count"] for smiles in inputs),
            "directed_bond_count": sum(encode(smiles)[1]["directed_bond_count"] for smiles in inputs),
            "bond_representation": "undirected_concat",
        }, tuple(inputs)

    return prepare_features(
        {stage: rows for stage in ("alignment", "sft", "grpo")},
        encode_inputs,
        normalize,
        partitions=tuple(Partition(stage, "train", stage, 0, len(rows)) for stage in ("alignment", "sft", "grpo")),
        encoder={"request_layers": LAYERS},
        dataset={"revision": sha256_json(rows)},
        cache_root=Path(output) / "feature-cache",
        manifest_path=Path(output) / "manifest.jsonl",
        max_scan_rows=len(rows),
    )


def main() -> None:
    from bionemo_perception import KERMTClient

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/sanity.jsonl")
    parser.add_argument("--output", default="artifacts/data")
    args = parser.parse_args()
    client = KERMTClient(os.environ["KERMT_URL"], token=os.environ.get("KERMT_TOKEN") or None, timeout=900)
    print(json.dumps(prepare(client, args.input, args.output), indent=2))


if __name__ == "__main__":
    main()
