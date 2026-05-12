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

"""Mac-runnable unit tests for the Evo2 FastAPI serve module.

These tests intentionally live outside ``tests/bionemo/evo2/`` because the
conftest there imports ``bionemo.core`` and ``torch.cuda``-only utilities,
which are not installable on a CPU-only Mac. By keeping these tests in a
sibling directory, pytest collects them independently and they can run on
any machine with just::

    pip install fastapi 'uvicorn[standard]' 'pydantic>=2' pytest httpx torch numpy

To run from the recipe root::

    PYTHONPATH=src pytest tests/unit_no_gpu/test_serve.py -v

Integration tests that actually load the model live under
``tests/bionemo/evo2/run/`` and require CUDA + the full bionemo install.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import ClassVar, List
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Load serve.py directly to bypass package-path shadowing
# ---------------------------------------------------------------------------
#
# pytest's test-discovery treats ``tests/`` as a rootdir and resolves
# ``bionemo.evo2.run`` to ``tests/bionemo/evo2/run/__init__.py`` (which exists
# for the CUDA test suite), shadowing the real source at
# ``src/bionemo/evo2/run/``. We side-step that by loading ``serve.py`` from
# its absolute path under a unique module name.

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SERVE_PATH = _REPO_ROOT / "src" / "bionemo" / "evo2" / "run" / "serve.py"


def _load_serve_module():
    if "_evo2_serve_under_test" in sys.modules:
        return sys.modules["_evo2_serve_under_test"]
    spec = importlib.util.spec_from_file_location("_evo2_serve_under_test", _SERVE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_evo2_serve_under_test"] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stub_score_result(num_seqs: int, collapse: str) -> dict:
    """Build a fake ``_score_sequences`` return value with the right shapes.

    Uses ``torch`` because the production code returns a tensor with
    ``.tolist()``. ``torch`` installs on Mac (CPU build).

    Args:
        num_seqs: Number of sequences scored.
        collapse: One of ``"sum"``, ``"mean"``, ``"per_token"``.

    Returns:
        Dict matching ``_compute_log_probs`` output shape.
    """
    import torch

    if collapse == "per_token":
        return {
            "log_probs_seqs": torch.full((num_seqs, 5), -1.5),
            "seq_idx": torch.arange(num_seqs),
            "loss_mask": torch.ones((num_seqs, 5), dtype=torch.bool),
        }
    return {
        "log_probs_seqs": torch.full((num_seqs,), -42.0),
        "seq_idx": torch.arange(num_seqs),
    }


@pytest.fixture
def configured_client(monkeypatch):
    """Yield a TestClient with model loading short-circuited and stubs installed.

    The lifespan handler sees no ``app.state.ckpt_dir`` and skips the real
    model load. We then set ``state.components`` to a sentinel ``MagicMock``
    so the endpoints don't 503, and patch ``_score_sequences`` plus the
    ``bionemo.evo2.run.infer`` module to avoid touching CUDA.
    """
    import sys

    serve = _load_serve_module()

    for attr in ("ckpt_dir", "tensor_parallel_size", "mixed_precision_recipe"):
        if hasattr(serve.app.state, attr):
            delattr(serve.app.state, attr)

    def fake_score(components, sequences: List[str], collapse: str) -> dict:
        return _stub_score_result(len(sequences), collapse)

    monkeypatch.setattr(serve, "_score_sequences", fake_score)

    fake_infer = MagicMock()
    fake_infer.generate = MagicMock(
        return_value=[
            MagicMock(
                generated_text="ACGTACGT",
                generated_length=4,
                generated_log_probs=None,
            )
        ]
    )
    fake_infer._unwrap_result = lambda r: r
    monkeypatch.setitem(sys.modules, "bionemo.evo2.run.infer", fake_infer)

    with TestClient(serve.app) as client:
        serve.app.state.serve_state.components = MagicMock()
        serve.app.state.serve_state.model_id = "evo2-test-stub"
        serve.app.state.serve_state.ckpt_dir = None
        serve.app.state.serve_state.requests_served = 0
        yield client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_health_endpoint_returns_model_metadata(configured_client):
    """GET /health returns 200 with model_id + uptime + counters."""
    resp = configured_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_id"] == "evo2-test-stub"
    assert body["tensor_parallel_size"] == 1
    assert body["uptime_s"] >= 0
    assert body["requests_served"] == 0


def test_score_endpoint_accepts_sequence_list(configured_client):
    """POST /score with `sequences` returns one log-prob per sequence."""
    resp = configured_client.post(
        "/score",
        json={"sequences": ["ACGTACGT", "TTTTAAAA"], "collapse": "sum"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["collapse"] == "sum"
    assert len(body["log_probs"]) == 2
    assert all(v == pytest.approx(-42.0) for v in body["log_probs"])
    assert body["seq_ids"] == ["seq_0", "seq_1"]


def test_score_endpoint_accepts_fasta_string(configured_client):
    """POST /score with `fasta` parses headers as seq_ids."""
    fasta = ">my_first_seq description goes here\nACGT\n>second\nGGTT\n"
    resp = configured_client.post("/score", json={"fasta": fasta, "collapse": "mean"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["seq_ids"] == ["my_first_seq", "second"]
    assert body["collapse"] == "mean"
    assert len(body["log_probs"]) == 2


def test_score_collapse_modes_return_correct_shape(configured_client):
    """`per_token` returns nested lists; sum/mean return scalars."""
    payload = {"sequences": ["ACGT"]}
    for mode in ("sum", "mean"):
        resp = configured_client.post("/score", json={**payload, "collapse": mode})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert isinstance(body["log_probs"][0], float)

    resp = configured_client.post("/score", json={**payload, "collapse": "per_token"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert isinstance(body["log_probs"][0], list)
    assert len(body["log_probs"][0]) == 5


def test_score_rejects_invalid_alphabet(configured_client):
    """Non-nucleotide characters trigger Pydantic validation (422)."""
    resp = configured_client.post(
        "/score",
        json={"sequences": ["ACGT", "ACXZ"], "collapse": "sum"},
    )
    assert resp.status_code == 422, resp.text


def test_score_rejects_oversized_input(configured_client):
    """Sequences over MAX_SEQUENCE_LENGTH are rejected (422)."""
    serve = _load_serve_module()

    oversized = "A" * (serve.MAX_SEQUENCE_LENGTH + 1)
    resp = configured_client.post(
        "/score",
        json={"sequences": [oversized], "collapse": "sum"},
    )
    assert resp.status_code == 422, resp.text


def test_score_requires_some_input(configured_client):
    """No sequences and no fasta => 400."""
    resp = configured_client.post("/score", json={"collapse": "sum"})
    assert resp.status_code == 400, resp.text


def test_generate_endpoint_basic_request(configured_client):
    """POST /generate returns the stubbed completion text."""
    resp = configured_client.post(
        "/generate",
        json={"prompts": ["ACGT"], "max_new_tokens": 16, "top_k": 1},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["completions"] == ["ACGTACGT"]
    assert body["finish_reasons"][0] in ("length", "stop")
    assert body["generated_log_probs"] is None


def test_generate_endpoint_returns_log_probs_when_requested(configured_client):
    """When `return_log_probs=True`, the response includes per-token logprobs."""
    import sys

    fake_infer = sys.modules["bionemo.evo2.run.infer"]

    class _Result:
        generated_text: ClassVar[str] = "ACGT"
        generated_length: ClassVar[int] = 4
        generated_log_probs: ClassVar[list] = [-0.1, -0.2, -0.3, -0.4]

    fake_infer.generate = MagicMock(return_value=[_Result()])
    fake_infer._unwrap_result = lambda r: r

    resp = configured_client.post(
        "/generate",
        json={
            "prompts": ["ACGT"],
            "max_new_tokens": 16,
            "top_k": 1,
            "return_log_probs": True,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["generated_log_probs"] == [[-0.1, -0.2, -0.3, -0.4]]


def test_fasta_parser_handles_multiline_records():
    """The in-module FASTA parser concatenates wrapped lines."""
    _parse_fasta = _load_serve_module()._parse_fasta

    fasta = ">one\nACGT\nACGT\n>two\nGGGG\n"
    ids, seqs = _parse_fasta(fasta)
    assert ids == ["one", "two"]
    assert seqs == ["ACGTACGT", "GGGG"]


def test_fasta_parser_rejects_empty_input():
    """Empty FASTA raises ValueError."""
    _parse_fasta = _load_serve_module()._parse_fasta

    with pytest.raises(ValueError):
        _parse_fasta("")
