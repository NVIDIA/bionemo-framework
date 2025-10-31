#!/usr/bin/env python3
"""
Generate golden values using NeMo/Megatron model with SIMPLE processing.

Uses the same approach as predict_golden_values.py but with NeMo model:
- Load model without distributed infrastructure
- Process sequences sequentially with batch_size=1
- Generate log probabilities same way as HF

This isolates if the 0.76 MAD is from:
- Model differences (HF vs NeMo/Megatron)
- Or BioNeMo's distributed processing infrastructure
"""

import argparse
import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, List
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import nemo.lightning as nl
from nemo.lightning import NeMoLogger, io
from nemo.lightning.ckpt_utils import ckpt_to_context_subdir
from nemo.collections.llm.inference.base import _setup_trainer_and_restore_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleFastaDataset(torch.utils.data.Dataset):
    """Simple FASTA dataset."""
    
    def __init__(self, fasta_path: Path, seq_length: int = 8192, prepend_bos: bool = False):
        super().__init__()
        self.fasta_path = fasta_path
        self.seq_length = seq_length
        self.prepend_bos = prepend_bos
        
        self.sequences = self._parse_fasta()
        self.windows = self._create_windows()
        
        logger.info(f"Loaded {len(self.sequences)} sequences")
        logger.info(f"Created {len(self.windows)} windows")
    
    def _parse_fasta(self) -> Dict[str, str]:
        sequences = {}
        current_header = None
        current_seq = []
        
        with open(self.fasta_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_header:
                        sequences[current_header] = ''.join(current_seq)
                    current_header = line[1:]
                    current_seq = []
                else:
                    current_seq.append(line.upper())
            
            if current_header:
                sequences[current_header] = ''.join(current_seq)
        
        return sequences
    
    def _create_windows(self) -> List[Dict]:
        windows = []
        for seq_idx, (header, sequence) in enumerate(self.sequences.items()):
            for start_pos in range(0, len(sequence), self.seq_length):
                end_pos = min(start_pos + self.seq_length, len(sequence))
                window_seq = sequence[start_pos:end_pos]
                
                if len(window_seq) > 0:
                    windows.append({
                        'seq_idx': seq_idx,
                        'header': header,
                        'sequence': window_seq,
                        'window_id': f"{header}_{start_pos}:{end_pos}"
                    })
        return windows
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx: int):
        window = self.windows[idx]
        sequence = window['sequence']
        
        # Tokenize as bytes
        tokens = torch.tensor([ord(c) for c in sequence], dtype=torch.long)
        
        if self.prepend_bos:
            bos = torch.tensor([2], dtype=torch.long)  # EOS token as BOS (Eden tokenizer convention)
            tokens = torch.cat([bos, tokens])
            tokens = tokens[:self.seq_length]
        else:
            tokens = tokens[:self.seq_length]
        
        # Pad if needed
        if len(tokens) < self.seq_length:
            padding = torch.zeros(self.seq_length - len(tokens), dtype=torch.long)
            tokens = torch.cat([tokens, padding])
        
        # Loss mask
        loss_mask = torch.ones_like(tokens, dtype=torch.bool)
        if len(sequence) < self.seq_length:
            loss_mask[len(sequence):] = False
        if self.prepend_bos:
            loss_mask[0] = False
        
        return {
            'tokens': tokens,
            'loss_mask': loss_mask,
            'header': window['header'],
        }


@torch.no_grad()
def generate_golden_values_nemo(model, dataloader, device):
    """Generate golden values using NeMo model."""
    model.eval()
    
    all_log_probs = []
    all_loss_masks = []
    all_headers = []
    
    # Compute RoPE once
    if hasattr(model, 'rotary_pos_emb'):
        rotary_pos_emb = model.rotary_pos_emb(8192)  # seq_length
    else:
        rotary_pos_emb = None
    
    logger.info("Generating predictions with NeMo...")
    for batch in tqdm(dataloader, desc="Processing"):
        tokens = batch['tokens'].to(device)  # [batch, seq]
        loss_mask = batch['loss_mask'].to(device)
        
        # NeMo expects [seq, batch, hidden], but we'll work with [batch, seq]
        # and let the embedding handle it
        
        # Get embeddings
        position_ids = torch.arange(tokens.shape[1], device=device).unsqueeze(0).expand(tokens.shape[0], -1)
        embeddings = model.embedding(tokens, position_ids)  # [seq, batch, hidden]
        
        # Run through all decoder layers
        hidden_states = embeddings
        for layer in model.decoder.layers:
            layer_out = layer(hidden_states, attention_mask=None, rotary_pos_emb=rotary_pos_emb)
            if isinstance(layer_out, tuple):
                hidden_states = layer_out[0]
            else:
                hidden_states = layer_out
        
        # Final layer norm
        hidden_states = model.decoder.final_layernorm(hidden_states)
        
        # Output projection to vocabulary
        # hidden_states is [seq, batch, hidden], transpose to [batch, seq, hidden]
        hidden_states = hidden_states.transpose(0, 1)
        
        # Get logits through output layer
        output = model.output_layer(hidden_states)  # [batch, seq, vocab]
        
        # Handle tuple output (output_layer might return (logits, bias))
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output
        
        # Log softmax
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Shift for next-token prediction
        log_probs_shifted = log_probs[:, :-1, :]
        tokens_shifted = tokens[:, 1:]
        loss_mask_shifted = loss_mask[:, 1:]
        
        # Gather log probs of correct tokens
        token_log_probs = torch.gather(
            log_probs_shifted, 2, tokens_shifted.unsqueeze(-1)
        ).squeeze(-1)
        
        all_log_probs.append(token_log_probs.cpu())
        all_loss_masks.append(loss_mask_shifted.cpu())
        all_headers.extend(batch['header'])
    
    # Concatenate
    log_probs_seqs = torch.cat(all_log_probs, dim=0)
    loss_mask = torch.cat(all_loss_masks, dim=0)
    
    return {
        'log_probs_seqs': log_probs_seqs,
        'loss_mask': loss_mask,
        'headers': all_headers,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="NeMo checkpoint directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seq-length", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prepend-bos", action="store_true")
    
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataset
    dataset = SimpleFastaDataset(args.fasta, args.seq_length, args.prepend_bos)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Setup NeMo trainer
    work_dir = Path(tempfile.mkdtemp())
    nemo_logger = NeMoLogger(log_dir=work_dir)
    
    trainer = nl.Trainer(
        devices=1,
        accelerator="gpu",
        strategy=nl.MegatronStrategy(),
        enable_checkpointing=False,
        logger=nemo_logger,
    )
    
    # Load model
    logger.info("Loading NeMo model...")
    model = io.load_context(path=ckpt_to_context_subdir(args.checkpoint), subpath="model")
    _setup_trainer_and_restore_model(path=args.checkpoint, trainer=trainer, model=model)
    
    megatron_model = model.module
    device = torch.device('cuda')
    logger.info("Model loaded")
    
    # Generate golden values
    results = generate_golden_values_nemo(megatron_model, dataloader, device)
    
    # Save
    torch.save(
        {'log_probs_seqs': results['log_probs_seqs'], 'loss_mask': results['loss_mask']},
        args.output_dir / 'predictions_nemo_simple.pt'
    )
    
    # Save seq_idx_map
    seq_idx_map = {header: idx for idx, header in enumerate(results['headers'])}
    with open(args.output_dir / 'seq_idx_map.json', 'w') as f:
        json.dump(seq_idx_map, f, indent=2)
    
    logger.info(f"✓ Saved to {args.output_dir}")
    logger.info(f"  Sequences: {len(results['headers'])}")


if __name__ == "__main__":
    main()

