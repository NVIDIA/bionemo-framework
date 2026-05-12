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

"""FastAPI service exposing Evo2 inference endpoints (/score, /generate, /health).

This is a thin wrapper around the existing batch CLIs (``predict_evo2``,
``infer_evo2``). It loads the model ONCE at startup and reuses the same
inference engine across requests, eliminating the per-batch cold-start cost
that the torchrun-based CLIs incur.

Endpoints:
    POST /score    -> per-sequence aggregated log-probabilities (forward-pass scoring)
    POST /generate -> autoregressive completion + optional per-token log-probs
    GET  /health   -> liveness, model metadata, uptime

Usage:
    evo2_serve --ckpt-dir /path/to/mbridge/checkpoint --port 8000 --host 0.0.0.0

Designed for single-process, single-GPU, single-flight serving. Concurrent
requests are serialized; multi-GPU tensor parallelism is not supported in v0.
Intended for use behind an SSH tunnel — no authentication is implemented.

The heavy ``bionemo`` / ``torch`` / ``megatron`` imports are deferred until
startup (via ``_load_components``) and per-request (via ``_score_sequences``),
so this module can be imported on machines without CUDA for unit testing.
"""

import argparse
import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, List, Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator


logger = logging.getLogger(__name__)


VALID_NUCLEOTIDES = set("ACGTNacgtn")
MAX_SEQUENCE_LENGTH = 1_048_576  # match Evo2 training context window


# =============================================================================
# Request / Response models
# =============================================================================


class ScoreRequest(BaseModel):
    """Request body for ``POST /score``."""

    sequences: Optional[List[str]] = Field(
        default=None,
        description="List of nucleotide strings (alphabet: A/C/G/T/N).",
    )
    fasta: Optional[str] = Field(
        default=None,
        description="Raw FASTA text. Overrides ``sequences`` if both are provided.",
    )
    collapse: Literal["sum", "mean", "per_token"] = Field(
        default="sum",
        description="How to aggregate per-token log-probabilities into a per-sequence score.",
    )

    @field_validator("sequences")
    @classmethod
    def _validate_alphabet(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is None:
            return v
        for i, seq in enumerate(v):
            bad = set(seq) - VALID_NUCLEOTIDES
            if bad:
                raise ValueError(f"sequence[{i}] contains non-nucleotide characters: {sorted(bad)}")
            if len(seq) > MAX_SEQUENCE_LENGTH:
                raise ValueError(f"sequence[{i}] exceeds MAX_SEQUENCE_LENGTH={MAX_SEQUENCE_LENGTH}")
        return v


class ScoreResponse(BaseModel):
    """Response body for ``POST /score``."""

    log_probs: List[Any]  # List[float] for sum/mean; List[List[float]] for per_token
    seq_ids: List[str]
    collapse: str


class GenerateRequest(BaseModel):
    """Request body for ``POST /generate``."""

    prompts: List[str]
    max_new_tokens: int = Field(default=100, gt=0, le=10000)
    top_k: int = Field(default=0, ge=0)
    top_p: float = Field(default=0.0, ge=0.0, le=1.0)
    temperature: float = Field(default=1.0, gt=0.0)
    return_log_probs: bool = False


class GenerateResponse(BaseModel):
    """Response body for ``POST /generate``."""

    completions: List[str]
    finish_reasons: List[str]
    generated_log_probs: Optional[List[List[float]]] = None


class HealthResponse(BaseModel):
    """Response body for ``GET /health``."""

    status: str
    model_id: str
    ckpt_dir: str
    tensor_parallel_size: int
    mixed_precision_recipe: str
    uptime_s: float
    requests_served: int


# =============================================================================
# App state (single-process; lives on ``app.state``)
# =============================================================================


class _AppState:
    """Mutable state attached to the FastAPI app instance.

    All model-related objects are loaded lazily during the lifespan startup
    phase and reused across requests.
    """

    def __init__(self) -> None:
        self.components: Optional[Any] = None  # Evo2InferenceComponents at runtime
        self.ckpt_dir: Optional[Path] = None
        self.model_id: str = "unknown"
        self.tensor_parallel_size: int = 1
        self.mixed_precision_recipe: str = "bf16_mixed"
        self.started_at: float = time.time()
        self.requests_served: int = 0
        self.serve_lock: Optional[asyncio.Lock] = None


# =============================================================================
# Lazy model loading + scoring
# =============================================================================


def _load_components(
    ckpt_dir: Path,
    *,
    max_seq_length: int = 8192,
    max_batch_size: int = 1,
    tensor_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    context_parallel_size: int = 1,
    mixed_precision_recipe: str = "bf16_mixed",
    vortex_style_fp8: bool = False,
    use_subquadratic_ops: bool = False,
) -> Any:
    """Import bionemo modules and call ``setup_inference_engine``.

    Imports are deferred to this function so that ``serve`` can be imported
    on machines without CUDA (e.g. for unit tests using ``TestClient`` with
    a monkey-patched loader).

    Args:
        ckpt_dir: Path to the MBridge checkpoint directory.
        max_seq_length: Maximum sequence length supported by the engine.
        max_batch_size: Maximum batch size for inference.
        tensor_parallel_size: Tensor parallelism degree (v0 requires 1).
        pipeline_model_parallel_size: Pipeline parallelism degree (v0 requires 1).
        context_parallel_size: Context parallelism degree (v0 requires 1).
        mixed_precision_recipe: Mixed precision recipe name.
        vortex_style_fp8: Apply FP8 only to projection layers.
        use_subquadratic_ops: Use fused subquadratic-ops kernels.

    Returns:
        ``Evo2InferenceComponents`` instance.
    """
    from bionemo.evo2.run.infer import setup_inference_engine

    return setup_inference_engine(
        ckpt_dir=ckpt_dir,
        max_seq_length=max_seq_length,
        max_batch_size=max_batch_size,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        context_parallel_size=context_parallel_size,
        mixed_precision_recipe=mixed_precision_recipe,
        vortex_style_fp8=vortex_style_fp8,
        use_subquadratic_ops=use_subquadratic_ops,
    )


def _score_sequences(
    components: Any,
    sequences: List[str],
    collapse: Literal["sum", "mean", "per_token"],
) -> dict:
    """Run a forward pass over ``sequences`` and aggregate per-token log-probs.

    Mirrors the forward-pass logic in ``predict.py`` (single-GPU, TP=CP=1
    fast path). Imports torch + bionemo lazily so the module is Mac-importable.

    Args:
        components: ``Evo2InferenceComponents`` from ``_load_components``.
        sequences: List of nucleotide strings.
        collapse: Aggregation: ``"sum"``, ``"mean"``, or ``"per_token"``.

    Returns:
        Dict with key ``"log_probs_seqs"`` (and ``"seq_idx"``, plus
        ``"loss_mask"`` if ``collapse == "per_token"``).
    """
    import torch

    from bionemo.evo2.run.predict import _compute_log_probs

    tokenizer = components.tokenizer
    model = components.model

    encoded = [torch.tensor(tokenizer.text_to_ids(seq), dtype=torch.long) for seq in sequences]
    max_len = max(t.shape[0] for t in encoded)
    pad_id = getattr(tokenizer, "pad_token_id", 0) or 0

    tokens = torch.full((len(encoded), max_len), pad_id, dtype=torch.long)
    loss_mask = torch.zeros((len(encoded), max_len), dtype=torch.float)
    for i, t in enumerate(encoded):
        tokens[i, : t.shape[0]] = t
        loss_mask[i, : t.shape[0]] = 1.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokens = tokens.to(device)
    loss_mask = loss_mask.to(device)
    position_ids = torch.arange(max_len, dtype=torch.long, device=device).unsqueeze(0).expand(len(encoded), -1)
    seq_idx = torch.arange(len(encoded), dtype=torch.long, device=device)

    # The model returned by ``setup_inference_engine`` has ``flash_decode=True``,
    # which requires an ``inference_context`` on every forward call. Pass the
    # wrapper's context (which has ``materialize_only_last_token_logits=False``
    # at setup time, so all-position logits are returned).
    components.inference_context.reset()
    with torch.no_grad():
        logits = model(
            input_ids=tokens,
            position_ids=position_ids,
            attention_mask=None,
            inference_context=components.inference_context,
            runtime_gather_output=True,
        )

    return _compute_log_probs(
        logits=logits,
        tokens=tokens,
        loss_mask=loss_mask,
        seq_idx=seq_idx,
        collapse_option=collapse,
        context_parallel_size=1,
    )


def _parse_fasta(fasta_text: str) -> tuple:
    """Minimal in-memory FASTA parser.

    Args:
        fasta_text: Raw FASTA text (one or more records).

    Returns:
        Tuple ``(ids, sequences)`` of equal length.
    """
    ids: List[str] = []
    seqs: List[str] = []
    current_id = ""
    current_parts: List[str] = []
    for raw in fasta_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_id:
                ids.append(current_id)
                seqs.append("".join(current_parts))
            header = line[1:].strip()
            current_id = header.split()[0] if header else f"seq_{len(ids)}"
            current_parts = []
        else:
            current_parts.append(line)
    if current_id:
        ids.append(current_id)
        seqs.append("".join(current_parts))
    if not ids:
        raise ValueError("FASTA contained no records.")
    return ids, seqs


# =============================================================================
# FastAPI app
# =============================================================================


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model at startup; nothing to release at shutdown.

    The lifespan handler reads configuration from ``app.state`` (populated
    by ``main`` from CLI args). In unit tests, ``app.state.ckpt_dir`` is
    left unset and the model load is skipped — tests then patch
    ``app.state.serve_state.components`` directly.
    """
    state: _AppState = app.state.serve_state
    state.serve_lock = asyncio.Lock()
    ckpt_dir = getattr(app.state, "ckpt_dir", None)
    if ckpt_dir is None:
        logger.warning("No ckpt_dir set on app.state; skipping model load (test mode).")
        yield
        return
    state.ckpt_dir = Path(ckpt_dir)
    state.tensor_parallel_size = getattr(app.state, "tensor_parallel_size", 1)
    state.mixed_precision_recipe = getattr(app.state, "mixed_precision_recipe", "bf16_mixed")
    logger.info("Loading model from %s ...", ckpt_dir)
    state.components = _load_components(
        ckpt_dir=state.ckpt_dir,
        max_seq_length=getattr(app.state, "max_seq_length", 8192),
        max_batch_size=getattr(app.state, "max_batch_size", 1),
        tensor_parallel_size=state.tensor_parallel_size,
        pipeline_model_parallel_size=getattr(app.state, "pipeline_model_parallel_size", 1),
        context_parallel_size=getattr(app.state, "context_parallel_size", 1),
        mixed_precision_recipe=state.mixed_precision_recipe,
        vortex_style_fp8=getattr(app.state, "vortex_style_fp8", False),
        use_subquadratic_ops=getattr(app.state, "use_subquadratic_ops", False),
    )
    state.model_id = state.ckpt_dir.name
    logger.info("Model loaded.")
    yield


def create_app() -> FastAPI:
    """Build the FastAPI app and register the three endpoints.

    Returns:
        Configured ``FastAPI`` instance. Model loading happens during the
        lifespan startup phase, not at app construction.
    """
    app = FastAPI(title="Evo2 Serve", version="0.1.0", lifespan=_lifespan)
    app.state.serve_state = _AppState()

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        state: _AppState = app.state.serve_state
        return HealthResponse(
            status="ok" if state.components is not None else "loading",
            model_id=state.model_id,
            ckpt_dir=str(state.ckpt_dir) if state.ckpt_dir else "",
            tensor_parallel_size=state.tensor_parallel_size,
            mixed_precision_recipe=state.mixed_precision_recipe,
            uptime_s=time.time() - state.started_at,
            requests_served=state.requests_served,
        )

    @app.post("/score", response_model=ScoreResponse)
    async def score(req: ScoreRequest) -> ScoreResponse:
        state: _AppState = app.state.serve_state
        if state.components is None:
            raise HTTPException(status_code=503, detail="Model not loaded.")

        if req.fasta:
            seq_ids, sequences = _parse_fasta(req.fasta)
        elif req.sequences is not None:
            sequences = req.sequences
            seq_ids = [f"seq_{i}" for i in range(len(sequences))]
        else:
            raise HTTPException(status_code=400, detail="Provide `sequences` or `fasta`.")
        if not sequences:
            raise HTTPException(status_code=400, detail="No sequences supplied.")

        assert state.serve_lock is not None
        async with state.serve_lock:
            out = _score_sequences(state.components, sequences, req.collapse)
            state.requests_served += 1

        log_probs_tensor = out["log_probs_seqs"]
        if req.collapse == "per_token":
            log_probs = [row.tolist() for row in log_probs_tensor]
        else:
            log_probs = log_probs_tensor.tolist()

        return ScoreResponse(log_probs=log_probs, seq_ids=seq_ids, collapse=req.collapse)

    @app.post("/generate", response_model=GenerateResponse)
    async def generate_endpoint(req: GenerateRequest) -> GenerateResponse:
        state: _AppState = app.state.serve_state
        if state.components is None:
            raise HTTPException(status_code=503, detail="Model not loaded.")

        # Lazy import: pulls in megatron + torch.
        from bionemo.evo2.run.infer import _unwrap_result, generate

        assert state.serve_lock is not None
        async with state.serve_lock:
            results = generate(
                state.components,
                prompts=req.prompts,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
                return_log_probs=req.return_log_probs,
            )
            state.requests_served += 1

        completions: List[str] = []
        finish_reasons: List[str] = []
        log_probs: List[List[float]] = []
        for raw in results:
            r = _unwrap_result(raw)
            completions.append(r.generated_text or "")
            finish_reasons.append("length" if (r.generated_length or 0) >= req.max_new_tokens else "stop")
            if req.return_log_probs and getattr(r, "generated_log_probs", None) is not None:
                lp = r.generated_log_probs
                if hasattr(lp, "tolist"):
                    lp = lp.tolist()
                log_probs.append(lp)

        return GenerateResponse(
            completions=completions,
            finish_reasons=finish_reasons,
            generated_log_probs=log_probs if req.return_log_probs else None,
        )

    return app


# Module-level app for uvicorn and TestClient. Lifespan runs only when the app
# is actually started (uvicorn, or TestClient context manager).
app = create_app()


# =============================================================================
# CLI
# =============================================================================


def _parse_args() -> argparse.Namespace:
    """Parse CLI args. Mirrors ``predict_evo2`` flag names where applicable.

    Returns:
        Parsed ``argparse.Namespace``.
    """
    ap = argparse.ArgumentParser(
        description="Serve Evo2 inference endpoints over HTTP (FastAPI).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--ckpt-dir", type=Path, required=True, help="Path to MBridge checkpoint directory.")
    ap.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind.")
    ap.add_argument("--port", type=int, default=8000, help="Port to bind.")
    ap.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallelism (v0: must be 1).")
    ap.add_argument("--pipeline-model-parallel-size", type=int, default=1)
    ap.add_argument("--context-parallel-size", type=int, default=1)
    ap.add_argument("--mixed-precision-recipe", type=str, default="bf16_mixed")
    ap.add_argument("--vortex-style-fp8", action="store_true")
    ap.add_argument("--use-subquadratic-ops", action="store_true")
    ap.add_argument("--max-seq-length", type=int, default=8192)
    ap.add_argument("--max-batch-size", type=int, default=1)
    ap.add_argument("--log-level", type=str, default="info")
    return ap.parse_args()


def main() -> None:
    """CLI entry point: parse args, stash on ``app.state``, run uvicorn.

    Raises:
        SystemExit: If ``tensor_parallel_size != 1`` (deferred to v1).
    """
    import uvicorn

    args = _parse_args()
    if args.tensor_parallel_size != 1:
        raise SystemExit("v0 supports only --tensor-parallel-size=1. Use `predict_evo2` for multi-GPU batch scoring.")

    app.state.ckpt_dir = args.ckpt_dir
    app.state.tensor_parallel_size = args.tensor_parallel_size
    app.state.pipeline_model_parallel_size = args.pipeline_model_parallel_size
    app.state.context_parallel_size = args.context_parallel_size
    app.state.mixed_precision_recipe = args.mixed_precision_recipe
    app.state.vortex_style_fp8 = args.vortex_style_fp8
    app.state.use_subquadratic_ops = args.use_subquadratic_ops
    app.state.max_seq_length = args.max_seq_length
    app.state.max_batch_size = args.max_batch_size

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
