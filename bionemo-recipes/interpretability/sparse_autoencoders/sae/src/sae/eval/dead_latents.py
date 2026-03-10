"""Simplified dead latent tracking for SAE training."""

import torch
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DeadLatentStats:
    """Dead latent statistics."""
    n_dead: int
    n_total: int
    dead_pct: float
    dead_indices: List[int]
    
    def __repr__(self) -> str:
        return f"DeadLatentStats(dead={self.n_dead}/{self.n_total} ({self.dead_pct:.1f}%))"


class DeadLatentTracker:
    """
    Tracks which latents have activated during training.
    
    Usage:
        tracker = DeadLatentTracker(hidden_dim=4096, device='cuda')
        
        for batch in dataloader:
            codes = sae.encode(batch)
            tracker.update(codes)
            
            # Log periodically
            if step % 1000 == 0:
                stats = tracker.get_stats()
                print(f"Dead latents: {stats.dead_pct:.1f}%")
        
        # Reset after resampling dead latents
        tracker.reset()
    """
    
    def __init__(
        self,
        hidden_dim: int,
        device: str = "cpu",
        eps: float = 1e-3,
    ):
        self.hidden_dim = hidden_dim
        self.device = device
        self.eps = eps

        # Tracks activity since last reset
        self.recently_active = torch.zeros(
            hidden_dim, dtype=torch.bool, device=device
        )
    
    @torch.no_grad()
    def update(self, codes: torch.Tensor) -> None:
        """
        codes: [batch, hidden_dim]
        """
        codes = codes.to(self.device)

        # Per-latent activity (since last reset)
        batch_active = (codes > self.eps).any(dim=0)
        self.recently_active |= batch_active

        # Store batch-local density metric
        self._last_avg_nonzero = (
            (codes > self.eps).sum(dim=-1).float().mean().item()
        )
    
    def get_stats(self) -> dict:
        dead_mask = ~self.recently_active
        n_dead = dead_mask.sum().item()

        return {
            "dead_pct": 100.0 * n_dead / self.hidden_dim,
            "n_dead": n_dead,
            "n_total": self.hidden_dim,
            "avg_nonzero": self._last_avg_nonzero,
        }
    
    def reset(self) -> None:
        """Call every few thousand steps."""
        self.recently_active.zero_()