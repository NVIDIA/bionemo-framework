"""
FastAPI server for SAE feature steering with SSE streaming.

Provides three endpoints:
- GET  /features — feature metadata from parquet (for the picker UI)
- POST /chat     — steered text generation with SSE streaming
- GET  /health   — model info

Launch via launch_steering_server() or use create_app() for custom setups.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from .steering import Intervention, InterventionMode, SteeredModel


def load_feature_metadata(parquet_dir: str | Path) -> List[Dict[str, Any]]:
    """Read feature_metadata.parquet into a list of dicts."""
    import pyarrow.parquet as pq

    parquet_dir = Path(parquet_dir)
    meta_path = parquet_dir / "feature_metadata.parquet"
    if not meta_path.exists():
        return []
    table = pq.read_table(meta_path)
    rows = table.to_pydict()
    n = len(rows.get("feature_id", []))
    features = []
    for i in range(n):
        feat = {}
        for col in rows:
            feat[col] = rows[col][i]
        features.append(feat)
    return features


def create_app(
    steered_model: SteeredModel,
    tokenizer: Any,
    parquet_dir: str | Path,
    max_new_tokens: int = 256,
) -> Any:
    """Create a FastAPI application for steering.

    Args:
        steered_model: SteeredModel wrapping the LM + SAE.
        tokenizer: HuggingFace tokenizer for the model.
        parquet_dir: Directory containing feature_metadata.parquet.
        max_new_tokens: Default max tokens for generation.

    Returns:
        FastAPI app instance.
    """
    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse

    app = FastAPI(title="SAE Steering Server")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Load feature metadata at startup
    feature_metadata = load_feature_metadata(parquet_dir)

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model": getattr(steered_model.model, "name_or_path", str(type(steered_model.model).__name__)),
            "sae_hidden_dim": steered_model.sae.hidden_dim,
            "sae_input_dim": steered_model.sae.input_dim,
            "layer": steered_model.layer,
            "n_features": len(feature_metadata),
        }

    @app.get("/features")
    def get_features(search: Optional[str] = None, limit: int = 200):
        results = feature_metadata
        if search:
            q = search.lower()
            results = [
                f for f in results
                if q in str(f.get("description", "")).lower()
                or q in str(f.get("feature_id", ""))
            ]
        return results[:limit]

    @app.post("/chat")
    async def chat(request: Request):
        body = await request.json()
        messages = body.get("messages", [])
        raw_interventions = body.get("interventions", [])
        compare = body.get("compare", False)
        max_tokens = body.get("max_tokens", max_new_tokens)

        # Build prompt from messages
        prompt = _build_prompt(messages)

        # Parse interventions
        interventions = [
            Intervention(
                feature_id=iv["feature_id"],
                weight=iv["weight"],
                mode=InterventionMode(iv.get("mode", "additive_code")),
            )
            for iv in raw_interventions
        ]

        async def event_stream():
            # Generate steered response
            if interventions:
                steered_model.set_interventions(interventions)
            else:
                steered_model.clear_interventions()

            steered_tokens = _generate_tokens(
                steered_model, tokenizer, prompt, max_tokens
            )
            for token_text in steered_tokens:
                event = json.dumps({"source": "steered", "token": token_text})
                yield f"event: token\ndata: {event}\n\n"

            # Generate baseline response if compare mode
            if compare:
                steered_model.clear_interventions()
                baseline_tokens = _generate_tokens(
                    steered_model, tokenizer, prompt, max_tokens
                )
                for token_text in baseline_tokens:
                    event = json.dumps({"source": "baseline", "token": token_text})
                    yield f"event: token\ndata: {event}\n\n"

            # Clean up: restore interventions if they were set
            if interventions and not compare:
                pass  # Already set
            elif interventions:
                steered_model.set_interventions(interventions)

            yield "event: done\ndata: {}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return app


def _build_prompt(messages: List[Dict[str, str]]) -> str:
    """Convert chat messages into a single prompt string for the LM."""
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
        elif role == "system":
            parts.append(content)
    parts.append("Assistant:")
    return "\n".join(parts)


def _generate_tokens(
    steered_model: SteeredModel,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
) -> List[str]:
    """Generate tokens one at a time, returning a list of token strings.

    Uses model.generate() with the steering hook active, then splits
    the output into individual tokens for streaming.
    """
    device = steered_model.device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    prompt_len = input_ids.shape[1]

    with torch.no_grad():
        output_ids = steered_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Extract only the new tokens
    new_token_ids = output_ids[0, prompt_len:]

    # Decode each token individually for streaming
    tokens = []
    for tid in new_token_ids:
        if tid.item() == tokenizer.eos_token_id:
            break
        tokens.append(tokenizer.decode([tid.item()]))

    return tokens


def launch_steering_server(
    steered_model: SteeredModel,
    tokenizer: Any,
    parquet_dir: str | Path,
    port: int = 8000,
    host: str = "127.0.0.1",
) -> None:
    """Start the steering API server.

    Args:
        steered_model: SteeredModel wrapping the LM + SAE.
        tokenizer: HuggingFace tokenizer.
        parquet_dir: Directory with feature_metadata.parquet.
        port: Server port (default: 8000).
        host: Server host (default: 127.0.0.1).
    """
    import uvicorn

    app = create_app(steered_model, tokenizer, parquet_dir)
    print(f"Starting steering server at http://{host}:{port}")
    print(f"  Features loaded from: {parquet_dir}")
    uvicorn.run(app, host=host, port=port)
