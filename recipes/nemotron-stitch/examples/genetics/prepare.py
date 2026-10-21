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

"""Encode the bundled DNA examples through an Evo2 perception endpoint."""

from __future__ import annotations

import argparse
import json
import os
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
from genetics_example import POOLING_WINDOW

from nemotron_stitch.features.manifest import load_manifest
from nemotron_stitch.features.preparation import Partition, prepare_features
from nemotron_stitch.provenance import sha256_json

OUTPUT = {"layer": "decoder.layers.12", "name": "features", "pooling": {"kind": "mean", "window": POOLING_WINDOW}}


def prepare(
    client: Any,
    source: str | Path = "data/sanity.jsonl",
    output: str | Path = "artifacts/data",
    *,
    encoder_width: int = 1920,
    pooling_window: int = POOLING_WINDOW,
) -> dict[str, Any]:
    """Publish cached features and alignment/SFT/GRPO views of the tiny dataset."""
    if pooling_window < 1:
        raise ValueError("pooling_window must be positive")
    output_spec = {**OUTPUT, "pooling": {**OUTPUT["pooling"], "window": pooling_window}}
    rows = load_manifest(source)

    @cache
    def encode(sequence: str) -> np.ndarray:
        if not sequence or set(sequence) - set("ACGTN"):
            raise ValueError("sequence must contain only uppercase A, C, G, T, N")
        reply = client.forward(sequence=sequence, outputs=[output_spec])
        feature = reply.tensor("features.output").numpy_float32()
        expected = ((len(sequence) + pooling_window - 1) // pooling_window, encoder_width)
        if feature.shape != expected or not np.isfinite(feature).all():
            raise ValueError(f"Evo2 features must be finite {expected}, got {feature.shape}")
        (descriptor,) = reply.tensor_metadata
        if descriptor["resolved_layer"] != output_spec["layer"] or descriptor["pooling"] != dict(
            kind="windowed", window=pooling_window
        ):
            raise ValueError("Evo2 response does not match the requested layer and pooling")
        return feature

    def normalize(
        row: dict[str, Any], *, configuration: str, source_index: int
    ) -> tuple[dict[str, Any], tuple[str, ...]]:
        inputs = tuple(
            block["sequence"]
            for message in row["messages"]
            for block in (message["content"] if isinstance(message["content"], list) else [])
            if block.get("type") == "dna"
        )
        return {
            "sample_id": f"{configuration}-{row['sample_id']}",
            "prompt": "",
            "messages": row["messages"],
            "target": row.get("target", "\\boxed{" + str(row["answer"]) + "}"),
            "reasoning_content": row.get("reasoning_content", ""),
            "question": "",
            "ground_truth": str(row["answer"]),
            "chat_template_kwargs": row.get(
                "chat_template_kwargs", {"enable_thinking": bool(row.get("reasoning_content"))}
            ),
        }, inputs

    return prepare_features(
        {stage: rows for stage in ("alignment", "sft", "grpo")},
        lambda inputs: (
            np.concatenate([encode(sequence) for sequence in inputs])
            if inputs
            else np.empty((0, encoder_width), dtype=np.float32)
        ),
        normalize,
        partitions=tuple(Partition(stage, "train", stage, 0, len(rows)) for stage in ("alignment", "sft", "grpo")),
        encoder={"layer": 12, "request": output_spec, "hidden_size": encoder_width},
        dataset={"revision": sha256_json(rows)},
        cache_root=Path(output) / "feature-cache",
        manifest_path=Path(output) / "manifest.jsonl",
        max_scan_rows=len(rows),
    )


def main() -> None:
    from bionemo_perception import Evo2Client

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/sanity.jsonl")
    parser.add_argument("--output", default="artifacts/data")
    parser.add_argument("--encoder-width", type=int, default=int(os.environ.get("STITCH_ENCODER_WIDTH", "1920")))
    parser.add_argument(
        "--pooling-window", type=int, default=int(os.environ.get("STITCH_POOLING_WINDOW", POOLING_WINDOW))
    )
    args = parser.parse_args()
    client = Evo2Client(os.environ["EVO2_URL"], token=os.environ.get("EVO2_TOKEN") or None, timeout=900)
    result = prepare(
        client, args.input, args.output, encoder_width=args.encoder_width, pooling_window=args.pooling_window
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
