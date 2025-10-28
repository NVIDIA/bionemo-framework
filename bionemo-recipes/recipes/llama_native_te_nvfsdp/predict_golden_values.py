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

"""
Generate golden values (log probabilities) from LLAMA3 implementation.

This script generates per-token log probabilities for comparison with John's
bionemo implementation. Golden values are log probabilities that can be used
with torch.nn.functional.cross_entropy (just take the negative and apply reduction).
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import your model
from model import NVLlamaForCausalLM

# Import tokenizer - try HF tokenizer first, fall back to custom
try:
    from transformers import AutoTokenizer
    # Try to load the HF tokenizer from the nucleotide_tokenizer directory
    tokenizer_path = "nucleotide_tokenizer"
    HAS_HF_TOKENIZER = True
except ImportError:
    HAS_HF_TOKENIZER = False
    from ascii_tokenizer import NucleotideASCIITokenizer

# Try to import bionemo collator, fall back to simple version if not available
try:
    from bionemo.llm.lightning import batch_collator
    HAS_BIONEMO = True
except ImportError:
    HAS_BIONEMO = False
    print("Warning: bionemo not installed, using simple collation")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleFastaDataset(torch.utils.data.Dataset):
    """
    Simple FASTA dataset for generating golden values.
    
    Follows the same approach as John's SimpleFastaDataset in bionemo-evo2,
    with sequential segmentation according to context length.
    """
    
    def __init__(
        self,
        fasta_path: Path,
        tokenizer,  # Can be any tokenizer (HF or custom)
        seq_length: int = 8192,
        prepend_bos: bool = False,  # Match John's default (False)
    ):
        """Initialize the dataset.
        
        Args:
            fasta_path: Path to FASTA file
            tokenizer: Tokenizer instance
            seq_length: Maximum sequence length (context length)
            prepend_bos: Whether to prepend BOS token (John uses False)
        """
        super().__init__()
        self.fasta_path = fasta_path
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.prepend_bos = prepend_bos
        
        # Parse FASTA file and create windows
        self.sequences = self._parse_fasta()
        self.windows = self._create_windows()
        
        logger.info(f"Loaded {len(self.sequences)} sequences from {fasta_path}")
        logger.info(f"Created {len(self.windows)} windows of length {seq_length}")
    
    def _parse_fasta(self) -> Dict[str, str]:
        """Parse FASTA file into a dictionary."""
        sequences = {}
        current_header = None
        current_seq = []
        
        with open(self.fasta_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    # Save previous sequence
                    if current_header is not None:
                        sequences[current_header] = ''.join(current_seq)
                    # Start new sequence
                    current_header = line[1:]  # Remove '>'
                    current_seq = []
                else:
                    current_seq.append(line.upper())
            
            # Save last sequence
            if current_header is not None:
                sequences[current_header] = ''.join(current_seq)
        
        return sequences
    
    def _create_windows(self) -> List[Dict]:
        """
        Create windows from sequences by sequential segmentation.
        
        This follows John's approach: segment sequences according to context length.
        Each sequence is split into non-overlapping windows of seq_length.
        """
        windows = []
        
        for seq_idx, (header, sequence) in enumerate(self.sequences.items()):
            # Sequential segmentation: split into non-overlapping windows
            seq_len = len(sequence)
            
            for start_pos in range(0, seq_len, self.seq_length):
                end_pos = min(start_pos + self.seq_length, seq_len)
                window_seq = sequence[start_pos:end_pos]
                
                # Only add windows that have actual sequence data
                if len(window_seq) > 0:
                    windows.append({
                        'seq_idx': seq_idx,
                        'header': header,
                        'sequence': window_seq,
                        'start_pos': start_pos,
                        'end_pos': end_pos,
                        'window_id': f"{header}_{start_pos}:{end_pos}"
                    })
        
        return windows
    
    def __len__(self):
        """Get the length of the dataset."""
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get an item from the dataset.
        
        Returns a dictionary with:
        - tokens: input token IDs [seq_len]
        - loss_mask: mask indicating which positions to use for loss [seq_len]
        - seq_idx: index of the original sequence
        - window_id: identifier for this window
        """
        window = self.windows[idx]
        sequence = window['sequence']
        
        # Tokenize (without special tokens initially)
        token_ids = self.tokenizer.encode(sequence, add_special_tokens=False)
        
        # Prepend BOS if requested
        # Note: John does NOT use --prepend-bos in his command, so by default no prepending
        # If prepending: John's tokenizer uses PAD=0, BOS=1, EOS=2, SEP=3
        # He would prepend tokenizer.eod (EOS=2) when prepend_bos=True
        if self.prepend_bos:
            # Prepend token 2 to match Eden tokenizer's EOS
            token_ids = [2] + token_ids
            # Truncate to seq_length to match the expected input size
            token_ids = token_ids[:self.seq_length]
        else:
            # No prepending - just truncate if needed
            token_ids = token_ids[:self.seq_length]
        
        # Convert to tensor
        tokens = torch.tensor(token_ids, dtype=torch.long)
        
        # Create loss mask
        loss_mask = torch.ones_like(tokens)
        if self.prepend_bos:
            # Mask the prepended EOS token (first position)
            loss_mask[0] = 0
        
        return {
            'tokens': tokens,
            'loss_mask': loss_mask,
            'seq_idx': window['seq_idx'],
            'window_id': window['window_id'],
        }
    
    def write_idx_map(self, output_dir: Path):
        """Write the index map to match John's format."""
        # Use just the header (not window_id with position suffix) to match John's format
        idx_map = {window['header']: idx for idx, window in enumerate(self.windows)}
        with open(output_dir / "seq_idx_map.json", "w") as f:
            json.dump(idx_map, f, indent=2)


def collate_fn(batch):
    """
    Collate function for batching sequences.
    
    Uses bionemo's batch_collator if available, otherwise uses simple default collation.
    """
    if HAS_BIONEMO:
        # Use bionemo's batch_collator for consistency with John's notebook
        return batch_collator(batch)
    else:
        # Simple collation: stack tensors and collect metadata
        return {
            'tokens': torch.stack([item['tokens'] for item in batch]),
            'loss_mask': torch.stack([item['loss_mask'] for item in batch]),
            'seq_idx': torch.tensor([item['seq_idx'] for item in batch]),
            'window_id': [item['window_id'] for item in batch],
        }


@torch.no_grad()
def generate_golden_values(
    model: NVLlamaForCausalLM,
    dataloader: DataLoader,
    device: torch.device,
    output_dir: Path,
) -> Dict[str, torch.Tensor]:
    """
    Generate golden values (log probabilities) for all sequences.
    
    This follows John's approach in predict.py:
    1. Run forward pass to get logits
    2. Apply log_softmax to get log probabilities
    3. Gather log probabilities for the actual tokens
    4. Apply loss mask to zero out padding positions
    
    Args:
        model: The LLAMA3 model
        dataloader: DataLoader with sequences
        device: Device to run on
        output_dir: Directory to save results
    
    Returns:
        Dictionary with log probabilities and metadata
    """
    model.eval()
    
    all_log_probs = []
    all_loss_masks = []
    all_seq_indices = []
    all_window_ids = []
    
    logger.info("Generating golden values...")
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        # Move to device
        tokens = batch['tokens'].to(device)  # [batch_size, seq_len]
        loss_mask = batch['loss_mask'].to(device)  # [batch_size, seq_len]
        
        # Forward pass
        outputs = model(input_ids=tokens)
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]
        
        # Apply log_softmax to get log probabilities
        log_probs = F.log_softmax(logits, dim=-1)  # [batch_size, seq_len, vocab_size]
        
        # The model's predictions for input i land at output i.
        # To align: predictions are at [:, :-1], target tokens at [:, 1:]
        log_probs_aligned = log_probs[:, :-1]  # [batch_size, seq_len-1, vocab_size]
        target_tokens = tokens[:, 1:]  # [batch_size, seq_len-1]
        
        # Gather log probabilities for the actual tokens
        # This gives us the log probability that the model assigned to each token
        token_log_probs = torch.gather(
            log_probs_aligned,  # [batch_size, seq_len-1, vocab_size]
            dim=2,  # Gather along vocab dimension
            index=target_tokens.unsqueeze(-1),  # [batch_size, seq_len-1, 1]
        ).squeeze(-1)  # [batch_size, seq_len-1]
        
        # Apply loss mask (offset by 1 to match the alignment)
        loss_mask_aligned = loss_mask[:, 1:]  # [batch_size, seq_len-1]
        masked_log_probs = token_log_probs * loss_mask_aligned.float()
        
        # Store results
        all_log_probs.append(masked_log_probs.cpu())
        all_loss_masks.append(loss_mask_aligned.cpu())
        all_seq_indices.append(batch['seq_idx'].cpu())
        all_window_ids.extend(batch['window_id'])
    
    # Concatenate all batches
    results = {
        'log_probs_seqs': torch.cat(all_log_probs, dim=0),  # [total_samples, seq_len-1]
        'loss_mask': torch.cat(all_loss_masks, dim=0),  # [total_samples, seq_len-1]
        'seq_idx': torch.cat(all_seq_indices, dim=0),  # [total_samples]
    }
    
    logger.info(f"Generated log probabilities shape: {results['log_probs_seqs'].shape}")
    logger.info(f"Loss mask shape: {results['loss_mask'].shape}")
    
    # Save results
    output_file = output_dir / "predictions_llama3_native.pt"
    torch.save(results, output_file)
    logger.info(f"Saved golden values to {output_file}")
    
    # Also save window ID mapping
    window_id_file = output_dir / "window_id_map.json"
    with open(window_id_file, 'w') as f:
        json.dump({
            'window_ids': all_window_ids,
            'description': 'Window IDs corresponding to each row in predictions_llama3_native.pt'
        }, f, indent=2)
    logger.info(f"Saved window ID mapping to {window_id_file}")
    
    return results


def compute_metrics(results: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    Compute summary metrics from the golden values.
    
    These can be used to verify that the values make sense.
    """
    log_probs = results['log_probs_seqs']
    loss_mask = results['loss_mask']
    
    # Compute per-sequence negative log likelihood (NLL)
    # NLL = -log_prob (cross entropy with reduction='none')
    nll_per_token = -log_probs
    
    # Compute mean NLL per sequence
    num_tokens_per_seq = loss_mask.sum(dim=1).clamp(min=1.0)  # [batch_size]
    nll_per_seq = (nll_per_token * loss_mask).sum(dim=1) / num_tokens_per_seq
    
    # Compute perplexity
    mean_nll = nll_per_seq.mean().item()
    perplexity = torch.exp(torch.tensor(mean_nll)).item()
    
    metrics = {
        'mean_negative_log_likelihood': mean_nll,
        'perplexity': perplexity,
        'total_sequences': log_probs.shape[0],
        'sequence_length': log_probs.shape[1],
        'total_tokens': loss_mask.sum().item(),
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Generate golden values from LLAMA3 implementation"
    )
    parser.add_argument(
        "--fasta",
        type=Path,
        required=True,
        help="Path to FASTA file with test sequences"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to checkpoint directory or file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save golden values"
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=8192,
        help="Sequence length for windowing (default: 8192)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference (default: 1)"
    )
    parser.add_argument(
        "--prepend-bos",
        action="store_true",
        help="Prepend BOS/EOS token to sequences (should match John's setting, he uses False)"
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype (default: bfloat16)"
    )
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Map dtype string to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[args.torch_dtype]
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize tokenizer
    logger.info("Initializing tokenizer...")
    if HAS_HF_TOKENIZER:
        try:
            tokenizer = AutoTokenizer.from_pretrained("nucleotide_tokenizer")
            logger.info("Using HuggingFace tokenizer from nucleotide_tokenizer/")
        except Exception as e:
            logger.warning(f"Could not load HF tokenizer: {e}")
            from ascii_tokenizer import NucleotideASCIITokenizer
            tokenizer = NucleotideASCIITokenizer()
            logger.info("Using custom ASCII tokenizer")
    else:
        tokenizer = NucleotideASCIITokenizer()
        logger.info("Using custom ASCII tokenizer")
    
    # Create dataset
    logger.info(f"Loading dataset from {args.fasta}...")
    dataset = SimpleFastaDataset(
        fasta_path=args.fasta,
        tokenizer=tokenizer,
        seq_length=args.seq_length,
        prepend_bos=args.prepend_bos,
    )
    
    # Save index map
    dataset.write_idx_map(args.output_dir)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,  # Keep simple for now
    )
    
    # Load model
    logger.info(f"Loading model from {args.checkpoint}...")
    model = NVLlamaForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=torch_dtype,
        device_map=device,
    )
    model.eval()
    logger.info("Model loaded successfully")
    
    # Generate golden values
    results = generate_golden_values(
        model=model,
        dataloader=dataloader,
        device=device,
        output_dir=args.output_dir,
    )
    
    # Compute and save metrics
    metrics = compute_metrics(results)
    logger.info("Metrics:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value}")
    
    metrics_file = args.output_dir / "metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_file}")
    
    logger.info("Done!")


if __name__ == "__main__":
    main()

