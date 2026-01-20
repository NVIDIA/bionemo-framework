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

import csv
from collections import defaultdict
from pathlib import Path

import torch
from transformers.trainer_pt_utils import get_parameter_names


SS3_ID2LABEL = {0: "H", 1: "E", 2: "C"}

SS3_LABEL2ID = {
    "H": 0,
    "I": 0,
    "G": 0,
    "E": 1,
    "B": 1,
    "S": 2,
    "T": 2,
    "~": 2,
    "C": 2,
    "L": 2,
}  # '~' denotes coil / unstructured

SS8_ID2LABEL = {0: "H", 1: "I", 2: "G", 3: "E", 4: "B", 5: "S", 6: "T", 7: "C"}

SS8_LABEL2ID = {
    "H": 0,
    "I": 1,
    "G": 2,
    "E": 3,
    "B": 4,
    "S": 5,
    "T": 6,
    "~": 7,
    "C": 7,
    "L": 7,
}  # '~' denotes coil / unstructured


def compute_accuracy(preds, labels, ignore_index=-100) -> tuple[int, int]:
    """Calculate the accuracy."""
    preds_labels = torch.argmax(preds, dim=-1)
    mask = labels != ignore_index
    correct = (preds_labels == labels) & mask

    return correct.sum().item(), mask.sum().item()


def get_parameter_names_with_lora(model):
    """Get layers with non-zero weight decay.

    This function reuses the Transformers' library function
    to list all the layers that should have weight decay.
    """
    forbidden_name_patterns = [
        r"bias",
        r"layernorm",
        r"rmsnorm",
        r"(?:^|\.)norm(?:$|\.)",
        r"_norm(?:$|\.)",
        r"\.lora_[AB]\.",
    ]

    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm], forbidden_name_patterns)

    return decay_parameters


def load_fasta(path: Path) -> list[dict]:
    """Read FASTA file and return input sequences."""
    records = []
    seq, pdb_id = [], None

    with open(path) as f:
        for raw_line in f:
            line = raw_line.strip()
            if line.startswith(">"):
                if seq:
                    records.append({"pdb_id": pdb_id, "sequence": "".join(seq)})
                pdb_id = line[1:] or None
                seq = []
            else:
                seq.append(line)

        if seq:
            records.append({"pdb_id": pdb_id, "sequence": "".join(seq)})

    return records


def load_csv(path: Path) -> list[dict]:
    """Read input CSV file for inference.

    It is assumed that the input CSV file contains:
    - Optional column named 'pdb_id' of the sequence.
    - Aminoacid sequence.
    """
    with open(path) as f:
        reader = csv.DictReader(f)
        has_pdb_id = "pdb_id" in reader.fieldnames

        return [
            {
                "pdb_id": row["pdb_id"] if has_pdb_id else None,
                "sequence": row["sequence"],
            }
            for row in reader
        ]


def load_input(path: Path) -> list[dict]:
    """Read the input sequences from FASTA or CSV file."""
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return load_csv(path)
    elif suffix in {".fa", ".fasta", ".faa"}:
        return load_fasta(path)
    else:
        raise ValueError(f"Unsupported input format: {suffix}")


def format_output_rows(records, predictions, sequences_to_sample_mapping):
    """Format the output into CSV-type lines.

    Returns:
      header: list[str]
      rows: list[tuple[str, str]]
    """
    has_pdb_id = any(r.get("pdb_id") for r in records)
    header = ["pdb_id", "prediction"] if has_pdb_id else ["id", "prediction"]

    counts = defaultdict(int)
    rows = []

    for pred, orig_idx in zip(predictions, sequences_to_sample_mapping):
        counts[orig_idx] += 1
        suffix = counts[orig_idx]

        base = records[orig_idx]["pdb_id"] if has_pdb_id else str(orig_idx)

        out_id = base if suffix == 1 else f"{base}_{suffix}"
        rows.append((out_id, pred))

    return header, rows


def write_output(records, predictions, sequences_to_sample_mapping: list[int], output_path: Path):
    """Write the predictions to an output file."""
    header, rows = format_output_rows(records, predictions, sequences_to_sample_mapping)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
