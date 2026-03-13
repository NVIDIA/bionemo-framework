"""
Feature steering via SAE interventions at inference time.

Intercepts a model's residual stream at a target layer, modifies specific
SAE feature activations (amplify, suppress, or clamp), and re-injects the
modified activations. Supports three intervention modes:

- additive_code: encode → codes[f] += weight → decode → replace
- multiplicative_code: encode → codes[f] *= weight → decode → replace
- direct: activations += weight * W_dec[f]  (no encode/decode)
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .architectures.base import SparseAutoencoder


class InterventionMode(str, Enum):
    ADDITIVE_CODE = "additive_code"
    MULTIPLICATIVE_CODE = "multiplicative_code"
    DIRECT = "direct"


@dataclass
class Intervention:
    """A single feature intervention."""
    feature_id: int
    weight: float
    mode: InterventionMode = InterventionMode.ADDITIVE_CODE


class SteeredModel:
    """Wraps a language model + SAE to apply feature steering at inference time.

    Registers a forward hook on the target transformer layer that intercepts
    the residual stream, applies interventions, and re-injects modified
    activations.

    Args:
        model: A HuggingFace-style transformer model (has .transformer.h or
            .model.layers attribute).
        sae: A trained SparseAutoencoder instance.
        layer: The layer index where the SAE was trained.
        device: Device for computations. If None, uses model's device.

    Example:
        >>> steered = SteeredModel(gpt2_model, sae, layer=6)
        >>> steered.set_interventions([
        ...     Intervention(feature_id=42, weight=3.0, mode=InterventionMode.ADDITIVE_CODE),
        ... ])
        >>> output = model.generate(input_ids)
    """

    def __init__(
        self,
        model: nn.Module,
        sae: SparseAutoencoder,
        layer: int,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.sae = sae
        self.layer = layer
        self.device = device or next(model.parameters()).device
        self._interventions: List[Intervention] = []
        self._hook_handle = None

        # Move SAE to same device as model, in eval mode
        self.sae = self.sae.to(self.device).eval()

        # Resolve the target layer module
        self._target_module = self._resolve_layer(model, layer)

    @staticmethod
    def _resolve_layer(model: nn.Module, layer: int) -> nn.Module:
        """Find the transformer block at the given layer index."""
        # GPT-2 style: model.transformer.h[layer]
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return model.transformer.h[layer]
        # LLaMA / Mistral style: model.model.layers[layer]
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers[layer]
        # Generic: try common attribute names
        for attr in ("layers", "blocks", "encoder.layer", "decoder.layer"):
            parts = attr.split(".")
            obj = model
            try:
                for p in parts:
                    obj = getattr(obj, p)
                return obj[layer]
            except (AttributeError, IndexError, TypeError):
                continue
        raise ValueError(
            f"Cannot find transformer layer {layer}. "
            "Supported layouts: model.transformer.h[], model.model.layers[], "
            "model.layers[], model.blocks[]"
        )

    def set_interventions(self, interventions: List[Intervention]) -> None:
        """Set active interventions and register/update the forward hook."""
        self._interventions = list(interventions)
        self._unregister_hook()
        if self._interventions:
            self._register_hook()

    def clear_interventions(self) -> None:
        """Remove all interventions and unregister the hook."""
        self._interventions = []
        self._unregister_hook()

    @contextmanager
    def intervene(self, interventions: List[Intervention]):
        """Context manager for temporary steering."""
        prev = self._interventions[:]
        self.set_interventions(interventions)
        try:
            yield self
        finally:
            self.set_interventions(prev)

    def _register_hook(self) -> None:
        self._hook_handle = self._target_module.register_forward_hook(
            self._hook_fn
        )

    def _unregister_hook(self) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    @torch.no_grad()
    def _hook_fn(self, module, input, output):
        """Forward hook that applies interventions to the residual stream."""
        # HuggingFace transformer blocks return (hidden_states, ...) tuples
        if isinstance(output, tuple):
            hidden_states = output[0]
            rest = output[1:]
        else:
            hidden_states = output
            rest = None

        # Separate interventions by type
        direct_interventions = [
            iv for iv in self._interventions if iv.mode == InterventionMode.DIRECT
        ]
        code_interventions = [
            iv for iv in self._interventions if iv.mode != InterventionMode.DIRECT
        ]

        # Apply direct interventions: activations += weight * W_dec[feature_id]
        if direct_interventions:
            # Get decoder weight matrix: shape [input_dim, hidden_dim]
            W_dec = self.sae.decoder.weight.data
            for iv in direct_interventions:
                # W_dec[:, feature_id] is the decoder direction for this feature
                direction = W_dec[:, iv.feature_id]  # [input_dim]
                hidden_states = hidden_states + iv.weight * direction

        # Apply code-space interventions: encode → modify → decode → replace
        if code_interventions:
            original_shape = hidden_states.shape  # [batch, seq_len, hidden_dim]
            flat = hidden_states.reshape(-1, original_shape[-1])  # [B*T, D]

            # Encode
            codes = self.sae.encode(flat)  # [B*T, n_features]

            # Apply modifications
            for iv in code_interventions:
                if iv.mode == InterventionMode.ADDITIVE_CODE:
                    codes[:, iv.feature_id] = codes[:, iv.feature_id] + iv.weight
                elif iv.mode == InterventionMode.MULTIPLICATIVE_CODE:
                    codes[:, iv.feature_id] = codes[:, iv.feature_id] * iv.weight

            # Decode back — use base decode (without normalization info)
            reconstructed = self.sae.decode(codes)  # [B*T, D]

            # Compute the SAE residual on the unmodified input to preserve
            # information not captured by the SAE
            flat_original = hidden_states.reshape(-1, original_shape[-1])
            codes_original = self.sae.encode(flat_original)
            recon_original = self.sae.decode(codes_original)
            residual = flat_original - recon_original  # what SAE can't represent

            # Final output: steered reconstruction + original residual
            hidden_states = (reconstructed + residual).reshape(original_shape)

        if rest is not None:
            return (hidden_states,) + rest
        return hidden_states

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **generate_kwargs,
    ) -> torch.Tensor:
        """Run model.generate() with current interventions active.

        This is a convenience wrapper — the hook fires automatically on each
        forward pass during autoregressive generation.
        """
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

    def __del__(self):
        self._unregister_hook()
