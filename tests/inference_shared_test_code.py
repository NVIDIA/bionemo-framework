# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging
import os
from pathlib import Path
from typing import Callable, List, Type

import torch
import torch.utils

from bionemo.model.core.infer import BaseEncoderDecoderInference
from bionemo.model.molecule.megamolbart import MegaMolBARTInference
from bionemo.model.molecule.molmim.infer import MolMIMInference
from bionemo.model.protein.esm1nv import ESM1nvInference
from bionemo.model.protein.prott5nv import ProtT5nvInference


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

UPDATE_GOLDEN_VALUES = os.environ.get("UPDATE_GOLDEN_VALUES", "0") == "1"


def get_inference_class(model_name: str) -> Type[BaseEncoderDecoderInference]:
    return {
        "megamolbart": MegaMolBARTInference,
        "molmim": MolMIMInference,
        "prott5nv": ProtT5nvInference,
        "esm1nv": ESM1nvInference,
    }[model_name]


def get_config_dir(bionemo_home: Path, model_name: str) -> str:
    modality = {
        "megamolbart": "molecule",
        "molmim": "molecule",
        "dsmbind": "molecule",
        "prott5nv": "protein",
        "esm1nv": "protein",
        "geneformer": "singlecell",
        "dnabert": "dna",
    }
    return str(bionemo_home / "examples" / modality[model_name] / model_name / "conf")


def get_expected_vals_file(bionemo_home: Path, model_name: str) -> Path:
    return (
        bionemo_home
        / "examples"
        / "tests"
        / "test_data"
        / "expected_outputs"
        / "inference_golden_values"
        / model_name
        / "inference_test_golden_values.pt"
    )


def run_seqs_to_hiddens_with_goldens(
    inferer: BaseEncoderDecoderInference,
    seqs: List[str],
    expected_vals_path: Path,
    hidden_size: int,
    encoder_arch: str,
    tokenize_fn: Callable,
):
    if UPDATE_GOLDEN_VALUES:
        expected_vals_path.parent.mkdir(exist_ok=True, parents=True)
    else:
        assert (
            expected_vals_path.exists()
        ), f"Expected values file not found at {expected_vals_path}. Rerun with UPDATE_GOLDEN_VALUES=1 to create it."
    assert inferer.training is False
    hidden_state, pad_masks = inferer.seq_to_hiddens(seqs)
    hidden_state2, pad_masks2 = inferer.seq_to_hiddens(seqs)
    hidden_state3, pad_masks3 = inferer.seq_to_hiddens(seqs)
    assert hidden_state is not None
    assert hidden_state2 is not None
    assert hidden_state3 is not None
    # Shape should be batch, position (max of input batch here), hidden_size
    assert len(hidden_state.shape) == 3
    assert hidden_state.shape[0] == len(seqs)
    if encoder_arch == "perceiver" or isinstance(inferer, MolMIMInference):
        # Perceiver uses a fixed length for the position state. MolMIM is defined with this arch
        # so assert it
        assert inferer.model.cfg.encoder.arch == "perceiver"
        assert hidden_state.shape[1] == inferer.model.cfg.encoder.hidden_steps
    else:
        # Number of hidden states should be equal to token length
        expected_max_padded_len: int = max([len(tokenize_fn([s])[0]) for s in seqs])
        assert hidden_state.shape[1] == expected_max_padded_len
    assert hidden_state.shape[2] == hidden_size
    assert pad_masks is not None
    assert pad_masks2 is not None
    assert pad_masks3 is not None
    assert pad_masks.shape == hidden_state.shape[:2]

    # Make sure that sequential runs of infer give the same result.
    torch.testing.assert_close(pad_masks3, pad_masks2, rtol=None, atol=None, equal_nan=True)
    torch.testing.assert_close(pad_masks, pad_masks2, rtol=None, atol=None, equal_nan=True)
    torch.testing.assert_close(hidden_state3, hidden_state2, rtol=None, atol=None, equal_nan=True)
    torch.testing.assert_close(hidden_state, hidden_state2, rtol=None, atol=None, equal_nan=True)

    if UPDATE_GOLDEN_VALUES:
        torch.save(
            {
                "expected_hidden_state": hidden_state,
                "expected_pad_masks": pad_masks,
            },
            expected_vals_path,
        )
        assert False, f"Updated expected values at {expected_vals_path}, rerun with UPDATE_GOLDEN_VALUES=0"
    else:
        expected_vals = {k: v.to(pad_masks.device) for k, v in torch.load(expected_vals_path).items()}
        # Equation for relative diff
        # relative_diff = torch.abs(h - e) / torch.max(atol, rtol * torch.max(torch.abs(h), torch.abs(e)))
        torch.testing.assert_close(
            hidden_state, expected_vals["expected_hidden_state"], rtol=None, atol=None, equal_nan=True
        )
        assert torch.all(pad_masks == expected_vals["expected_pad_masks"])


def run_seqs_to_embedding(inferer: BaseEncoderDecoderInference, seqs: List[str], expected_hidden_size_encoder: int):
    embedding = inferer.seq_to_embeddings(seqs)
    assert embedding is not None
    # Shape should be batch, hidden_size (Embeddings pool out the position axis of hiddens by some means)
    assert embedding.shape[0] == len(seqs)
    assert embedding.shape[1] == expected_hidden_size_encoder
    assert len(embedding.shape) == 2
