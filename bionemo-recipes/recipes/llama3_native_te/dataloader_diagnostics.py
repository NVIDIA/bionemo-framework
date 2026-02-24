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

"""Diagnostic logging for comparing Eden and HF streaming dataloaders.

Instruments both dataloaders to log:
  - Per-batch: token composition, sequence lengths, unique sequences seen
  - Per-epoch: shard coverage, window repetition, sequence diversity
  - Rolling: effective shuffle radius, batch similarity over time

Usage:
    diag = DataloaderDiagnostics(rank=0, log_dir="/path/to/logs", tag="eden_thd")
    for step, batch in enumerate(dataloader):
        diag.log_batch(step, batch)
        if step % 500 == 0:
            diag.log_summary(step)
    diag.close()
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
import torch


logger = logging.getLogger(__name__)


class DataloaderDiagnostics:
    """Unified diagnostic logger for comparing different dataloaders.

    Tracks batch composition, sequence diversity, and effective shuffling quality
    to identify why one dataloader overfits while another doesn't.
    """

    # Lookup table: ASCII token ID → base index (0=A, 1=C, 2=G, 3=T, -1=other)
    _TOKEN_TO_BASE = np.full(256, -1, dtype=np.int8)
    _TOKEN_TO_BASE[65] = 0  # A
    _TOKEN_TO_BASE[97] = 0  # a
    _TOKEN_TO_BASE[67] = 1  # C
    _TOKEN_TO_BASE[99] = 1  # c
    _TOKEN_TO_BASE[71] = 2  # G
    _TOKEN_TO_BASE[103] = 2  # g
    _TOKEN_TO_BASE[84] = 3  # T
    _TOKEN_TO_BASE[116] = 3  # t

    def __init__(  # noqa: D107
        self,
        rank: int,
        world_size: int = 1,
        log_dir: str | None = None,
        tag: str = "default",
        log_every_n_steps: int = 100,
        rolling_window: int = 1000,
        grad_acc_steps: int = 8,
        enabled: bool = True,
        resume: bool = False,
    ) -> None:
        self.rank = rank
        self.world_size = world_size
        self.tag = tag
        self.log_every_n_steps = log_every_n_steps
        self.grad_acc_steps = grad_acc_steps
        self.enabled = enabled and (rank == 0)  # Only log on rank 0

        if not self.enabled:
            return

        self.log_dir = Path(log_dir) if log_dir else Path(os.getcwd()) / "dataloader_diagnostics"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # --- Per-batch CSV log ---
        # On resume, append to existing CSV to preserve pre-resume data
        batch_csv_path = self.log_dir / f"batch_stats_{tag}_rank{rank}.csv"
        file_exists = batch_csv_path.exists() and batch_csv_path.stat().st_size > 0
        mode = "a" if (resume and file_exists) else "w"
        self._batch_file = open(batch_csv_path, mode, newline="")
        self._batch_writer = csv.writer(self._batch_file)
        if mode == "w":
            self._batch_writer.writerow(
                [
                    "step",
                    "timestamp",
                    "num_sequences",
                    "total_tokens",
                    "valid_tokens",
                    "masked_tokens",
                    "pad_tokens",
                    "mean_seq_len",
                    "min_seq_len",
                    "max_seq_len",
                    "std_seq_len",
                    "batch_hash",
                    "gc_content",
                    "n_content",
                    "kmer4_entropy",
                    "kmer4_cosine_prev",
                    "grad_acc_pairwise_sim",
                ]
            )

        # --- Rolling window tracking ---
        self._rolling_window = rolling_window
        self._recent_batch_hashes: deque[str] = deque(maxlen=rolling_window)

        # --- 4-mer frequency profile (256 dimensions for ACGT alphabet) ---
        # The 4-mer profile captures species-level signature, GC content, repeat
        # structure, and biological similarity in a single 256-dim vector.
        # Cosine similarity of these vectors between consecutive batches is the
        # primary measure of batch-to-batch autocorrelation.
        self._prev_kmer_profile: np.ndarray | None = None
        self._kmer_cosine_history: deque[float] = deque(maxlen=rolling_window)
        self._latest_kmer_cosine: float = 0.0

        # --- Grad accumulation window tracking ---
        # Stores k-mer profiles for the last grad_acc_steps microbatches.
        # Mean pairwise cosine within this window = effective sample size proxy.
        # If the 8 samples in a grad_acc window are highly correlated, gradient
        # accumulation is amplifying redundancy rather than reducing variance.
        self._grad_acc_kmer_profiles: list[np.ndarray] = []
        self._latest_grad_acc_sim: float = 0.0
        self._grad_acc_sim_history: deque[float] = deque(maxlen=rolling_window)

        # --- k-mer entropy (species diversity proxy) ---
        # Shannon entropy of the 4-mer distribution. Higher = more diverse content.
        # Different species have different k-mer profiles, so low entropy = samples
        # from one organism, high entropy = samples from many organisms.
        self._kmer_entropy_history: deque[float] = deque(maxlen=rolling_window)
        self._latest_kmer_entropy: float = 0.0

        # --- GC content tracking ---
        self._gc_content_history: deque[float] = deque(maxlen=rolling_window)
        self._n_content_history: deque[float] = deque(maxlen=rolling_window)
        self._latest_gc_content: float = 0.0
        self._latest_n_content: float = 0.0

        # --- Per-sequence fingerprinting (legacy) ---
        self._prev_batch_fingerprints: set[str] = set()
        self._fingerprint_overlaps: deque[float] = deque(maxlen=rolling_window)

        # --- Cumulative tracking ---
        self._total_batches = 0
        self._total_tokens_seen = 0
        self._all_batch_hashes: Counter = Counter()
        self._token_histogram: Counter = Counter()  # token_id -> count
        self._seq_len_histogram: list[int] = []

        # --- Summary JSON log ---
        self._summary_path = self.log_dir / f"summary_{tag}_rank{rank}.jsonl"
        summary_mode = "a" if resume else "w"
        self._summary_file = open(self._summary_path, summary_mode)

        logger.info(f"[DIAGNOSTICS] Initialized: tag={tag}, log_dir={self.log_dir}, rolling_window={rolling_window}")

    def _compute_kmer_profile(self, flat_tokens: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Compute 4-mer frequency vector and GC/N content from flat token array.

        Uses vectorized numpy operations for efficiency (~0.5ms for 8192 tokens).

        Args:
            flat_tokens: 1D numpy array of token IDs (ASCII values).

        Returns:
            Tuple of (kmer_counts[256], gc_content, n_content).
        """
        # Map ASCII token IDs to base indices (0=A, 1=C, 2=G, 3=T, -1=other)
        # Clip to 0-255 range to handle any out-of-range token IDs
        clipped = np.clip(flat_tokens, 0, 255).astype(np.int32)
        bases = self._TOKEN_TO_BASE[clipped]

        # GC content from base counts
        base_counts = np.bincount(bases[bases >= 0], minlength=4)
        a_count, c_count, g_count, t_count = (
            int(base_counts[0]),
            int(base_counts[1]),
            int(base_counts[2]),
            int(base_counts[3]),
        )
        n_count = int(np.sum(clipped == 78)) + int(np.sum(clipped == 110))  # N, n
        dna_total = a_count + c_count + g_count + t_count + n_count
        gc_content = (c_count + g_count) / max(dna_total, 1)
        n_content = n_count / max(dna_total, 1)

        # Compute 4-mer indices using vectorized operations
        # kmer_idx = b[i]*64 + b[i+1]*16 + b[i+2]*4 + b[i+3]
        # Cast to int32 first: bases is int8, and 3*64=192 overflows int8 range (-128..127)
        kmer_counts = np.zeros(256, dtype=np.float64)
        if len(bases) >= 4:
            b0 = bases[:-3].astype(np.int32)
            b1 = bases[1:-2].astype(np.int32)
            b2 = bases[2:-1].astype(np.int32)
            b3 = bases[3:].astype(np.int32)
            all_valid = (b0 >= 0) & (b1 >= 0) & (b2 >= 0) & (b3 >= 0)
            kmer_idx = b0 * 64 + b1 * 16 + b2 * 4 + b3
            valid_kmers = kmer_idx[all_valid]
            if len(valid_kmers) > 0:
                kmer_counts = np.bincount(valid_kmers, minlength=256).astype(np.float64)

        return kmer_counts, gc_content, n_content

    @staticmethod
    def _shannon_entropy(counts: np.ndarray) -> float:
        """Compute Shannon entropy of a count distribution.

        Args:
            counts: Array of counts (not necessarily normalized).

        Returns:
            Entropy in bits. Max for 256 bins = log2(256) = 8.0.
        """
        total = counts.sum()
        if total == 0:
            return 0.0
        probs = counts / total
        probs = probs[probs > 0]  # Avoid log(0)
        return float(-np.sum(probs * np.log2(probs)))

    @staticmethod
    def _mean_pairwise_cosine(profiles: list[np.ndarray]) -> float:
        """Compute mean pairwise cosine similarity among a list of unit vectors.

        Args:
            profiles: List of L2-normalized vectors.

        Returns:
            Mean cosine similarity across all pairs. Range [0, 1] for non-negative vectors.
        """
        n = len(profiles)
        if n < 2:
            return 0.0
        # Stack into matrix and compute cosine via dot products
        mat = np.stack(profiles)  # (n, 256)
        gram = mat @ mat.T  # (n, n) cosine similarity matrix (vectors are normalized)
        # Mean of upper triangle (excluding diagonal)
        total = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += gram[i, j]
                count += 1
        return total / max(count, 1)

    def log_batch(self, step: int, batch: dict[str, Any]) -> None:
        """Log diagnostics for a single batch."""
        if not self.enabled:
            return

        self._total_batches += 1

        input_ids = batch.get("input_ids")
        labels = batch.get("labels")
        attention_mask = batch.get("attention_mask")

        if input_ids is None:
            return

        if isinstance(input_ids, torch.Tensor):
            input_ids_np = input_ids.detach().cpu().numpy()
        else:
            input_ids_np = np.array(input_ids)

        # --- Basic stats ---
        total_tokens = int(input_ids_np.size)
        self._total_tokens_seen += total_tokens

        # Count valid/masked/pad tokens from labels
        valid_tokens = total_tokens
        masked_tokens = 0
        pad_tokens = 0
        if labels is not None:
            labels_np = labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor) else np.array(labels)
            masked_tokens = int((labels_np == -100).sum())
            valid_tokens = total_tokens - masked_tokens
        if attention_mask is not None:
            attn_np = (
                attention_mask.detach().cpu().numpy()
                if isinstance(attention_mask, torch.Tensor)
                else np.array(attention_mask)
            )
            pad_tokens = int((attn_np == 0).sum())

        # --- Sequence length stats ---
        # For THD (packed) batches, use cu_seq_lens to get individual sequence lengths
        cu_seq_lens = batch.get("cu_seq_lens_q")
        if cu_seq_lens is not None:
            if isinstance(cu_seq_lens, torch.Tensor):
                cu = cu_seq_lens.detach().cpu().numpy()
            else:
                cu = np.array(cu_seq_lens)
            seq_lens = np.diff(cu).tolist()
        elif input_ids_np.ndim == 2:
            # BSHD: each row is a sequence, get actual lengths from attention mask
            if attention_mask is not None:
                seq_lens = attn_np.sum(axis=1).tolist()
            else:
                seq_lens = [input_ids_np.shape[1]] * input_ids_np.shape[0]
        else:
            seq_lens = [total_tokens]

        num_sequences = len(seq_lens)
        seq_lens_arr = np.array(seq_lens, dtype=np.float64)
        mean_seq_len = float(seq_lens_arr.mean()) if len(seq_lens_arr) > 0 else 0.0
        min_seq_len = int(seq_lens_arr.min()) if len(seq_lens_arr) > 0 else 0
        max_seq_len = int(seq_lens_arr.max()) if len(seq_lens_arr) > 0 else 0
        std_seq_len = float(seq_lens_arr.std()) if len(seq_lens_arr) > 0 else 0.0
        self._seq_len_histogram.extend(seq_lens)

        # --- Batch hash (detect exact duplicates) ---
        flat = input_ids_np.flatten()
        batch_hash = hashlib.md5(flat.tobytes()).hexdigest()[:12]
        self._all_batch_hashes[batch_hash] += 1
        self._recent_batch_hashes.append(batch_hash)

        # --- Token histogram (sample to avoid memory blowup) ---
        if self._total_batches % 10 == 0:
            for tok in flat[:1000]:
                self._token_histogram[int(tok)] += 1

        # --- 4-mer frequency profile + genomic composition ---
        # Compute the 256-dim 4-mer frequency vector from nucleotide tokens.
        # This is the core representation for all diversity metrics.
        kmer_profile, gc_content, n_content = self._compute_kmer_profile(flat)

        # Store GC/N content
        self._latest_gc_content = gc_content
        self._latest_n_content = n_content
        self._gc_content_history.append(gc_content)
        self._n_content_history.append(n_content)

        # --- k-mer entropy (species diversity proxy) ---
        # Shannon entropy of the 4-mer distribution.
        # High entropy = diverse sequence content (many species/regions).
        # Low entropy = repetitive or single-species content.
        kmer_entropy = self._shannon_entropy(kmer_profile)
        self._latest_kmer_entropy = kmer_entropy
        self._kmer_entropy_history.append(kmer_entropy)

        # --- Batch-to-batch 4-mer cosine similarity (autocorrelation) ---
        # Cosine of 256-dim k-mer vectors between consecutive microbatches.
        # This is the KEY metric: high cosine = model seeing same genomic region.
        kmer_cosine = 0.0
        kmer_norm = np.linalg.norm(kmer_profile)
        kmer_normed = kmer_profile / max(kmer_norm, 1e-12)
        if self._prev_kmer_profile is not None:
            kmer_cosine = float(np.dot(kmer_normed, self._prev_kmer_profile))
        self._prev_kmer_profile = kmer_normed
        self._latest_kmer_cosine = kmer_cosine
        self._kmer_cosine_history.append(kmer_cosine)

        # --- Grad accumulation window similarity (ESS proxy) ---
        # Store k-mer profiles for the current grad_acc window.
        # Every grad_acc_steps microbatches, compute mean pairwise cosine.
        # If the 8 samples in a grad_acc window are highly similar,
        # gradient accumulation is amplifying redundancy.
        self._grad_acc_kmer_profiles.append(kmer_normed)
        if len(self._grad_acc_kmer_profiles) >= self.grad_acc_steps:
            self._latest_grad_acc_sim = self._mean_pairwise_cosine(self._grad_acc_kmer_profiles)
            self._grad_acc_sim_history.append(self._latest_grad_acc_sim)
            self._grad_acc_kmer_profiles = []

        # --- Per-sequence fingerprinting ---
        # Hash the first 64 tokens of each sequence to detect same-source windows.
        current_fingerprints: set[str] = set()
        if input_ids_np.ndim == 2:
            for row in input_ids_np:
                fp = hashlib.md5(row[:64].tobytes()).hexdigest()[:8]
                current_fingerprints.add(fp)
        elif cu_seq_lens is not None:
            # THD: extract individual sequences from packed tensor
            cu = cu_seq_lens.detach().cpu().numpy() if isinstance(cu_seq_lens, torch.Tensor) else np.array(cu_seq_lens)
            for i in range(len(cu) - 1):
                start, end = int(cu[i]), int(cu[i + 1])
                seq_tokens = flat[start : min(start + 64, end)]
                fp = hashlib.md5(seq_tokens.tobytes()).hexdigest()[:8]
                current_fingerprints.add(fp)

        fingerprint_overlap = 0.0
        if self._prev_batch_fingerprints and current_fingerprints:
            shared = len(current_fingerprints & self._prev_batch_fingerprints)
            fingerprint_overlap = shared / max(len(current_fingerprints), 1)
        self._prev_batch_fingerprints = current_fingerprints
        self._fingerprint_overlaps.append(fingerprint_overlap)

        # --- Write batch CSV ---
        if step % self.log_every_n_steps == 0 or self._total_batches <= 20:
            self._batch_writer.writerow(
                [
                    step,
                    int(time.time()),
                    num_sequences,
                    total_tokens,
                    valid_tokens,
                    masked_tokens,
                    pad_tokens,
                    f"{mean_seq_len:.1f}",
                    min_seq_len,
                    max_seq_len,
                    f"{std_seq_len:.1f}",
                    batch_hash,
                    f"{gc_content:.4f}",
                    f"{n_content:.6f}",
                    f"{kmer_entropy:.4f}",
                    f"{kmer_cosine:.6f}",
                    f"{self._latest_grad_acc_sim:.6f}",
                ]
            )
            self._batch_file.flush()

    def log_summary(self, step: int) -> None:
        """Log a rolling summary of dataloader behavior."""
        if not self.enabled:
            return

        # Batch hash diversity: how many unique batches in the rolling window?
        recent_unique = len(set(self._recent_batch_hashes))
        recent_total = len(self._recent_batch_hashes)
        batch_diversity = recent_unique / max(recent_total, 1)

        # Duplicate batches overall
        total_duplicates = sum(1 for v in self._all_batch_hashes.values() if v > 1)

        # Sequence length distribution
        if self._seq_len_histogram:
            sl = np.array(self._seq_len_histogram[-10000:])  # Last 10k
            seq_len_stats = {
                "mean": float(sl.mean()),
                "std": float(sl.std()),
                "p10": float(np.percentile(sl, 10)),
                "p50": float(np.percentile(sl, 50)),
                "p90": float(np.percentile(sl, 90)),
                "min": int(sl.min()),
                "max": int(sl.max()),
            }
        else:
            seq_len_stats = {}

        # 4-mer cosine similarity stats (autocorrelation)
        kmer_cosine_stats = {}
        if self._kmer_cosine_history:
            kc = np.array(list(self._kmer_cosine_history))
            kmer_cosine_stats["mean"] = float(kc.mean())
            kmer_cosine_stats["std"] = float(kc.std())
            kmer_cosine_stats["p50"] = float(np.percentile(kc, 50))
            kmer_cosine_stats["min"] = float(kc.min())

        # k-mer entropy stats (species diversity proxy)
        kmer_entropy_stats = {}
        if self._kmer_entropy_history:
            ke = np.array(list(self._kmer_entropy_history))
            kmer_entropy_stats["mean"] = float(ke.mean())
            kmer_entropy_stats["std"] = float(ke.std())

        # Grad acc pairwise similarity stats (ESS proxy)
        grad_acc_stats = {}
        if self._grad_acc_sim_history:
            ga = np.array(list(self._grad_acc_sim_history))
            grad_acc_stats["mean"] = float(ga.mean())
            grad_acc_stats["std"] = float(ga.std())

        # GC content stats
        gc_stats = {}
        if self._gc_content_history:
            gc_arr = np.array(list(self._gc_content_history))
            gc_stats["gc_mean"] = float(gc_arr.mean())
            gc_stats["gc_std"] = float(gc_arr.std())

        summary = {
            "step": step,
            "tag": self.tag,
            "total_batches": self._total_batches,
            "total_tokens_seen": self._total_tokens_seen,
            "rolling_batch_diversity": batch_diversity,
            "total_unique_batches": len(self._all_batch_hashes),
            "total_duplicate_batches": total_duplicates,
            "seq_len_stats": seq_len_stats,
            "kmer_cosine_stats": kmer_cosine_stats,
            "kmer_entropy_stats": kmer_entropy_stats,
            "grad_acc_pairwise_sim_stats": grad_acc_stats,
            "gc_stats": gc_stats,
        }

        self._summary_file.write(json.dumps(summary) + "\n")
        self._summary_file.flush()

        logger.info(
            f"[DIAG:{self.tag}] step={step} batches={self._total_batches} "
            f"kmer_cosine={kmer_cosine_stats.get('mean', 0):.6f} "
            f"kmer_entropy={kmer_entropy_stats.get('mean', 0):.4f} "
            f"grad_acc_sim={grad_acc_stats.get('mean', 0):.6f} "
            f"gc_mean={gc_stats.get('gc_mean', 0):.4f} gc_std={gc_stats.get('gc_std', 0):.4f}"
        )

    def get_wandb_metrics(self) -> dict[str, float]:
        """Return the latest diagnostic metrics as a dict suitable for wandb.log().

        Call this after log_batch() to get per-step metrics that can be
        overlaid with loss curves in wandb for direct correlation analysis.

        Key metrics for comparing dataloaders:
          - diag/kmer4_cosine_prev: 4-mer cosine similarity between consecutive batches.
              HIGH = same genomic region (poor shuffling). This is the KEY autocorrelation metric.
          - diag/kmer4_entropy: Shannon entropy of 4-mer distribution (species diversity proxy).
              LOW = single species/region, HIGH = diverse content.
          - diag/grad_acc_pairwise_sim: Mean pairwise 4-mer cosine within a grad_acc window.
              HIGH = redundant gradient accumulation (ESS proxy).
          - diag/gc_content: GC% of current batch.
          - diag/gc_content_std100: Rolling std of GC% over 100 batches.

        Returns:
            Dictionary of metric name -> value. Empty dict if diagnostics disabled.
        """
        if not self.enabled:
            return {}

        metrics: dict[str, float] = {}

        # --- 4-mer cosine similarity (autocorrelation — most important metric) ---
        metrics["diag/kmer4_cosine_prev"] = self._latest_kmer_cosine
        if len(self._kmer_cosine_history) >= 10:
            recent_kc = np.array(list(self._kmer_cosine_history)[-100:])
            metrics["diag/kmer4_cosine_avg100"] = float(recent_kc.mean())
            metrics["diag/kmer4_cosine_std100"] = float(recent_kc.std())

        # --- k-mer entropy (species diversity proxy) ---
        metrics["diag/kmer4_entropy"] = self._latest_kmer_entropy
        if len(self._kmer_entropy_history) >= 10:
            recent_ke = np.array(list(self._kmer_entropy_history)[-100:])
            metrics["diag/kmer4_entropy_avg100"] = float(recent_ke.mean())

        # --- Grad accumulation window pairwise similarity (ESS proxy) ---
        metrics["diag/grad_acc_pairwise_sim"] = self._latest_grad_acc_sim
        if len(self._grad_acc_sim_history) >= 10:
            recent_ga = np.array(list(self._grad_acc_sim_history)[-100:])
            metrics["diag/grad_acc_sim_avg100"] = float(recent_ga.mean())

        # --- GC content ---
        metrics["diag/gc_content"] = self._latest_gc_content
        metrics["diag/n_content"] = self._latest_n_content
        if len(self._gc_content_history) >= 10:
            recent_gc = np.array(list(self._gc_content_history)[-100:])
            metrics["diag/gc_content_std100"] = float(recent_gc.std())
            metrics["diag/gc_content_mean100"] = float(recent_gc.mean())

        # --- Sequence fingerprint overlap (legacy) ---
        if self._fingerprint_overlaps:
            metrics["diag/seq_fingerprint_overlap"] = self._fingerprint_overlaps[-1]

        # Cumulative diversity
        if self._all_batch_hashes:
            total = sum(self._all_batch_hashes.values())
            unique = len(self._all_batch_hashes)
            metrics["diag/batch_diversity"] = unique / max(total, 1)

        return metrics

    def close(self) -> None:
        """Flush and close all log files."""
        if not self.enabled:
            return
        try:
            self._batch_file.close()
        except Exception:
            pass
        try:
            self._summary_file.close()
        except Exception:
            pass
        logger.info(f"[DIAGNOSTICS] Closed logs for tag={self.tag}")


class StreamingDatasetDiagnostics:
    """Additional diagnostics specific to HF streaming datasets.

    Tracks shard access patterns, buffer state, window ordering,
    and effective shuffle radius to understand where overfitting comes from.
    """

    def __init__(  # noqa: D107
        self,
        rank: int,
        log_dir: str | None = None,
        tag: str = "hf_streaming",
        enabled: bool = True,
        resume: bool = False,
    ) -> None:
        self.rank = rank
        self.tag = tag
        self.enabled = enabled and (rank == 0)

        if not self.enabled:
            return

        self.log_dir = Path(log_dir) if log_dir else Path(os.getcwd()) / "dataloader_diagnostics"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # --- Window ordering tracking ---
        self._shard_access_counts: Counter = Counter()
        self._sequence_access_counts: Counter = Counter()
        self._windows_per_sequence: defaultdict[str, int] = defaultdict(int)

        # --- Consecutive window similarity ---
        # Track same-source-sequence runs using overlap/prefix hash matching.
        # With stride=7992 and window=8192, consecutive windows from the same source
        # sequence share 200bp: window N's last 200 tokens == window N+1's first 200 tokens.
        # We detect this by comparing the previous window's overlap_hash (last 200 tokens)
        # against the current window's prefix_hash (first 200 tokens).
        self._prev_overlap_hash: str | None = None
        self._consec_same_source = 0  # Count of consecutive same-source-sequence pairs
        self._same_source_run_lengths: list[int] = []  # Length of each same-source run
        self._current_run_length = 1

        # Legacy: also track first-64-token hash matches (detects exact window duplicates)
        self._prev_window_hash: str | None = None
        self._consec_same_hash = 0

        # --- CSV for per-window tracking (sampled) ---
        window_csv_path = self.log_dir / f"window_order_{tag}_rank{rank}.csv"
        file_exists = window_csv_path.exists() and window_csv_path.stat().st_size > 0
        mode = "a" if (resume and file_exists) else "w"
        self._window_file = open(window_csv_path, mode, newline="")
        self._window_writer = csv.writer(self._window_file)
        if mode == "w":
            self._window_writer.writerow(
                [
                    "yield_idx",
                    "shard_idx",
                    "window_in_seq_idx",
                    "seq_len_tokens",
                    "window_token_hash",
                    "same_source_as_prev",
                ]
            )
        self._yield_count = 0

    def log_window(
        self,
        shard_idx: int | None = None,
        window_in_seq_idx: int | None = None,
        seq_len_tokens: int | None = None,
        window_token_hash: str | None = None,
        overlap_hash: str | None = None,
        prefix_hash: str | None = None,
    ) -> None:
        """Log a single window yield from the streaming dataset.

        Args:
            shard_idx: Which shard this window came from (if known).
            window_in_seq_idx: Position of this window within its source sequence.
            seq_len_tokens: Number of tokens in this window.
            window_token_hash: Hash of the first 64 tokens. Unique per window position
                since consecutive windows start stride=7992 tokens apart.
            overlap_hash: Hash of the last 200 tokens of this window. With stride=7992
                and window=8192, this region overlaps with the NEXT window's first 200
                tokens if they come from the same source sequence.
            prefix_hash: Hash of the first 200 tokens of this window. Compared against
                the PREVIOUS window's overlap_hash to detect same-source-sequence runs.
        """
        if not self.enabled:
            return

        self._yield_count += 1

        if shard_idx is not None:
            self._shard_access_counts[shard_idx] += 1

        # Detect same-source-sequence runs via overlap/prefix hash matching.
        # If the previous window's last-200-token hash matches this window's
        # first-200-token hash, these windows come from the same source sequence.
        same_source = False
        if self._prev_overlap_hash is not None and prefix_hash is not None:
            same_source = self._prev_overlap_hash == prefix_hash
        self._prev_overlap_hash = overlap_hash

        if same_source:
            self._consec_same_source += 1
            self._current_run_length += 1
        else:
            if self._current_run_length > 1:
                self._same_source_run_lengths.append(self._current_run_length)
            self._current_run_length = 1

        # Legacy: track exact window hash duplicates
        if window_token_hash is not None and window_token_hash == self._prev_window_hash:
            self._consec_same_hash += 1
        self._prev_window_hash = window_token_hash

        # Sample logging to avoid too much I/O
        if self._yield_count <= 100 or self._yield_count % 1000 == 0:
            self._window_writer.writerow(
                [
                    self._yield_count,
                    shard_idx if shard_idx is not None else "",
                    window_in_seq_idx if window_in_seq_idx is not None else "",
                    seq_len_tokens if seq_len_tokens is not None else "",
                    window_token_hash if window_token_hash is not None else "",
                    "1" if same_source else "0",
                ]
            )
            self._window_file.flush()

    def log_shard_summary(self, step: int) -> None:
        """Log summary of shard access patterns."""
        if not self.enabled:
            return

        total_accesses = sum(self._shard_access_counts.values())
        num_shards_accessed = len(self._shard_access_counts)

        if num_shards_accessed == 0:
            return

        counts = list(self._shard_access_counts.values())
        counts_arr = np.array(counts, dtype=np.float64)

        # Measure how uniformly shards are accessed (1.0 = perfect uniformity)
        expected_per_shard = total_accesses / num_shards_accessed
        uniformity = 1.0 - (counts_arr.std() / max(expected_per_shard, 1))

        # Same-source-sequence rate: what fraction of adjacent window pairs came from
        # the same parent genomic sequence? Detected via overlap/prefix hash matching
        # (previous window's last-200-token hash == current window's first-200-token hash).
        # High values mean the model sees correlated genomic regions in consecutive steps.
        consec_same_source_rate = self._consec_same_source / max(self._yield_count - 1, 1)

        # Same-source run length stats: how many consecutive windows from the same sequence?
        # For sequence shuffle (no window shuffle), this equals the number of windows per
        # sequence. For window shuffle, the buffer breaks up these runs.
        # Finalize current run if > 1
        run_lengths = list(self._same_source_run_lengths)
        if self._current_run_length > 1:
            run_lengths.append(self._current_run_length)

        run_stats = {}
        if run_lengths:
            rl = np.array(run_lengths, dtype=np.float64)
            run_stats = {
                "mean_run_length": float(rl.mean()),
                "max_run_length": int(rl.max()),
                "median_run_length": float(np.median(rl)),
                "num_runs": len(run_lengths),
            }

        # Legacy: exact window hash duplicate rate
        consec_same_hash_rate = self._consec_same_hash / max(self._yield_count - 1, 1)

        summary = {
            "step": step,
            "tag": self.tag,
            "total_window_yields": self._yield_count,
            "num_shards_accessed": num_shards_accessed,
            "total_shard_accesses": total_accesses,
            "shard_uniformity": uniformity,
            "shard_access_min": int(counts_arr.min()),
            "shard_access_max": int(counts_arr.max()),
            "shard_access_std": float(counts_arr.std()),
            "consecutive_same_source_count": self._consec_same_source,
            "consecutive_same_source_rate": consec_same_source_rate,
            "same_source_run_stats": run_stats,
            "consecutive_same_hash_count": self._consec_same_hash,
            "consecutive_same_hash_rate": consec_same_hash_rate,
            "top_shards": self._shard_access_counts.most_common(10),
        }

        logger.info(
            f"[SHARD_DIAG:{self.tag}] step={step} shards={num_shards_accessed} "
            f"uniformity={uniformity:.4f} same_source_rate={consec_same_source_rate:.4f} "
            f"same_hash_rate={consec_same_hash_rate:.4f} yields={self._yield_count}"
        )
        if run_stats:
            logger.info(
                f"[SHARD_DIAG:{self.tag}] same-source runs: "
                f"mean={run_stats['mean_run_length']:.1f} max={run_stats['max_run_length']} "
                f"median={run_stats['median_run_length']:.1f} count={run_stats['num_runs']}"
            )

        # Write to shared summary file
        summary_path = self.log_dir / f"shard_summary_{self.tag}_rank{self.rank}.jsonl"
        with open(summary_path, "a") as f:
            f.write(json.dumps(summary) + "\n")

    def close(self) -> None:
        """Flush and close all log files."""
        if not self.enabled:
            return
        try:
            self._window_file.close()
        except Exception:
            pass


class EdenDatasetDiagnostics:
    """Additional diagnostics specific to the ShardedEdenDataset.

    Tracks window access patterns, sequence/sample diversity,
    and compares the ordering to understand what makes Eden converge better.
    """

    def __init__(  # noqa: D107
        self,
        rank: int,
        log_dir: str | None = None,
        tag: str = "eden",
        enabled: bool = True,
    ) -> None:
        self.rank = rank
        self.tag = tag
        self.enabled = enabled and (rank == 0)

        if not self.enabled:
            return

        self.log_dir = Path(log_dir) if log_dir else Path(os.getcwd()) / "dataloader_diagnostics"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # --- Sequence/sample tracking ---
        self._sequence_access_counts: Counter = Counter()
        self._sample_access_counts: Counter = Counter()
        self._window_idx_access_counts: Counter = Counter()
        self._window_in_seq_histogram: Counter = Counter()

        # --- Ordering tracking ---
        self._access_order: list[tuple[int, str]] = []  # (window_idx, sequence_id)

        # --- CSV for per-window tracking (sampled) ---
        window_csv_path = self.log_dir / f"eden_window_order_{tag}_rank{rank}.csv"
        self._window_file = open(window_csv_path, "w", newline="")
        self._window_writer = csv.writer(self._window_file)
        self._window_writer.writerow(
            [
                "yield_idx",
                "window_idx",
                "sequence_id",
                "sample_id",
                "window_in_seq_idx",
                "seq_len_tokens",
            ]
        )
        self._yield_count = 0

    def log_window(
        self,
        window_idx: int,
        sequence_id: str,
        sample_id: str,
        window_in_seq_idx: int,
        seq_len_tokens: int | None = None,
    ) -> None:
        """Log a single window access from the Eden dataset."""
        if not self.enabled:
            return

        self._yield_count += 1
        self._sequence_access_counts[sequence_id] += 1
        self._sample_access_counts[sample_id] += 1
        self._window_idx_access_counts[window_idx] += 1
        self._window_in_seq_histogram[window_in_seq_idx] += 1
        self._access_order.append((window_idx, sequence_id))

        # Sample logging
        if self._yield_count <= 100 or self._yield_count % 1000 == 0:
            self._window_writer.writerow(
                [
                    self._yield_count,
                    window_idx,
                    sequence_id,
                    sample_id,
                    window_in_seq_idx,
                    seq_len_tokens if seq_len_tokens is not None else "",
                ]
            )
            self._window_file.flush()

    def log_summary(self, step: int) -> None:
        """Log summary of Eden access patterns."""
        if not self.enabled:
            return

        num_unique_sequences = len(self._sequence_access_counts)
        num_unique_samples = len(self._sample_access_counts)
        num_unique_windows = len(self._window_idx_access_counts)

        # Window repetition
        window_counts = list(self._window_idx_access_counts.values())
        if window_counts:
            wc = np.array(window_counts, dtype=np.float64)
            window_repeat_stats = {
                "mean": float(wc.mean()),
                "max": int(wc.max()),
                "windows_accessed_once": int((wc == 1).sum()),
                "windows_accessed_multiple": int((wc > 1).sum()),
            }
        else:
            window_repeat_stats = {}

        # Sample diversity
        sample_counts = list(self._sample_access_counts.values())
        if sample_counts:
            sc = np.array(sample_counts, dtype=np.float64)
            sample_uniformity = 1.0 - (sc.std() / max(sc.mean(), 1))
        else:
            sample_uniformity = 0.0

        # Effective shuffle radius from window index ordering
        shuffle_radius = 0.0
        if len(self._access_order) > 1:
            idxs = [x[0] for x in self._access_order[-10000:]]
            diffs = np.abs(np.diff(idxs))
            shuffle_radius = float(diffs.mean())

        # Consecutive same-sequence runs (indicates poor shuffling)
        consec_same_seq = 0
        max_run = 0
        current_run = 1
        for i in range(1, min(len(self._access_order), 10000)):
            if self._access_order[i][1] == self._access_order[i - 1][1]:
                current_run += 1
                consec_same_seq += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 1

        summary = {
            "step": step,
            "tag": self.tag,
            "total_yields": self._yield_count,
            "unique_sequences": num_unique_sequences,
            "unique_samples": num_unique_samples,
            "unique_windows": num_unique_windows,
            "window_repeat_stats": window_repeat_stats,
            "sample_uniformity": sample_uniformity,
            "effective_shuffle_radius": shuffle_radius,
            "consecutive_same_seq_pairs": consec_same_seq,
            "max_same_seq_run": max_run,
            "top_sequences": self._sequence_access_counts.most_common(10),
            "top_samples": self._sample_access_counts.most_common(10),
            "window_in_seq_distribution": self._window_in_seq_histogram.most_common(20),
        }

        logger.info(
            f"[EDEN_DIAG:{self.tag}] step={step} yields={self._yield_count} "
            f"uniq_seqs={num_unique_sequences} uniq_samples={num_unique_samples} "
            f"uniq_windows={num_unique_windows} shuffle_radius={shuffle_radius:.1f} "
            f"consec_same_seq={consec_same_seq} max_run={max_run} "
            f"sample_uniformity={sample_uniformity:.4f}"
        )

        summary_path = self.log_dir / f"eden_summary_{self.tag}_rank{self.rank}.jsonl"
        with open(summary_path, "a") as f:
            f.write(json.dumps(summary) + "\n")

    def close(self) -> None:
        """Flush and close all log files."""
        if not self.enabled:
            return
        try:
            self._window_file.close()
        except Exception:
            pass
