from typing import Dict, List, Optional, Tuple
import random

import torch
import numpy as np
import torch.distributed as dist
from safetensors.torch import load_file
import json
from pathlib import Path

from src.tokenizer import Tokenizer
from src.tokenizer.mappings import AA_TABLE
from src.inference.base import BaseInference
from src.inference.task_types import TaskTypes
from src.inference.model_outputs import (
    MaskedLMOutput,
    MutationPredictionOutput,
    FitnessPredictionOutput,
    EmbeddingOutput,
    DownstreamPredictionOutput,
)
from src.models.encodon_pl import EncodonPL
from src.data.metadata import MetadataFields


class EncodonInference(BaseInference):
    """Inference class for Encodon models."""
    
    def configure_model(self):
        """Loads the model and tokenizer for inference."""
        if self.model is not None:
            return
        self.tokenizer = Tokenizer()
        
        state_dict = None
        hparams = None
        
        # Expect a full file path to either a .ckpt or a .safetensors file
        model_path = Path(self.model_path)
        if model_path.suffix.lower() not in [".ckpt", ".safetensors"]:
            raise ValueError(
                f"Expected a file path to a .ckpt or .safetensors file, got: {self.model_path}"
            )
        if not model_path.is_file():
            raise FileNotFoundError(f"Model file not found at {self.model_path}")

        suffix = model_path.suffix.lower()
        if suffix == ".safetensors":
            # Ensure config.json exists in the same directory
            config_path = model_path.parent / "config.json"
            if not config_path.exists():
                raise FileNotFoundError(
                    f"config.json is required to load safetensors checkpoint at: {config_path}"
                )
            if dist.is_initialized():
                broadcasted_objects = [None, None]
                if dist.get_rank() == 0:
                    state_dict = load_file(str(model_path))
                    with open(config_path, 'r') as f:
                        hparams = json.load(f)
                    broadcasted_objects = [state_dict, hparams]
                dist.broadcast_object_list(broadcasted_objects, src=0)
                state_dict, hparams = broadcasted_objects
            else:
                state_dict = load_file(str(model_path))
                with open(config_path, 'r') as f:
                    hparams = json.load(f)
        elif suffix == ".ckpt":
            # Load from PyTorch Lightning checkpoint
            if dist.is_initialized():
                broadcasted_objects = [None, None]
                if dist.get_rank() == 0:
                    ckpt = torch.load(self.model_path, map_location="cpu", weights_only=False)
                    hparams = ckpt.get("hyper_parameters")
                    state_dict = ckpt.get("state_dict")
                    broadcasted_objects = [state_dict, hparams]
                dist.broadcast_object_list(broadcasted_objects, src=0)
                state_dict, hparams = broadcasted_objects
            else:
                ckpt = torch.load(self.model_path, map_location="cpu", weights_only=False)
                hparams = ckpt.get("hyper_parameters")
                state_dict = ckpt.get("state_dict")
        else:
            raise ValueError(
                f"Unsupported model file type: {suffix}. Expected .ckpt or .safetensors"
            )

        # The hparams from lightning checkpoint might be nested.
        if 'hparams' in hparams:
            hparams = hparams['hparams']
        
        # For inference we don't need optimizer and scheduler, but EncodonPL expects them.
        def dummy_optimizer(params):
            return torch.optim.Adam(params)
        
        hparams['optimizer'] = dummy_optimizer
        hparams['scheduler'] = None

        self.model = EncodonPL(**hparams)
        self.model.configure_model(state_dict=state_dict)
        self.model.to(self.device)
        self.model.eval()
        
    def predict_mlm(self, batch, ids=None) -> Dict[str, np.ndarray]:
        """
        Predict masked tokens in a batch.
        
        Args:
            batch: Dictionary with INPUT_MASK and LABELS fields.
            ids: Optional sequence identifiers.
            
        Returns:
            MaskedLMOutput with predictions and labels at masked positions.
        """
        if MetadataFields.INPUT_MASK not in batch:
            raise ValueError(f"Batch missing required field: {MetadataFields.INPUT_MASK}")
        if MetadataFields.LABELS not in batch:
            raise ValueError(f"Batch missing required field: {MetadataFields.LABELS}")
        
        with torch.no_grad():
            output = self.model(batch)
            preds = output.logits
            if preds.dtype != torch.float:
                preds = preds.float()
            mask = batch[MetadataFields.INPUT_MASK].bool()
            y = batch[MetadataFields.LABELS]
            y = y[mask]
            preds = preds[mask]
            preds = preds.cpu().numpy()
            y = y.cpu().numpy()
            
        return MaskedLMOutput(preds=preds, labels=y, ids=ids)
        
    def predict_mutation(self, batch, ids=None) -> Dict[str, np.ndarray]:
        """
        Score variants by comparing log probabilities at the mutation position.
        
        Args:
            batch: Dictionary with REF_CODON_TOKS, ALT_CODON_TOKS, MUTATION_TOKEN_IDX.
            ids: Optional sequence identifiers.
            
        Returns:
            MutationPredictionOutput with ref/alt likelihoods and ratios.
        """
        required_fields = [
            MetadataFields.REF_CODON_TOKS,
            MetadataFields.ALT_CODON_TOKS,
            MetadataFields.MUTATION_TOKEN_IDX
        ]
        for field in required_fields:
            if field not in batch:
                raise ValueError(f"Batch missing required field for mutation prediction: {field}")
        
        with torch.no_grad():
            output = self.model(batch)
            preds = output.logits
            if preds.dtype != torch.float:
                preds = preds.float()
            ref_toks = batch[MetadataFields.REF_CODON_TOKS].view(-1)
            alt_toks = batch[MetadataFields.ALT_CODON_TOKS].view(-1)
            mutation_token_idx = batch[MetadataFields.MUTATION_TOKEN_IDX].view(-1)
            
            seq_len = preds.shape[1]
            if (mutation_token_idx >= seq_len).any() or (mutation_token_idx < 0).any():
                raise ValueError(
                    f"mutation_token_idx contains out-of-bounds indices. "
                    f"Valid range: [0, {seq_len-1}], got min={mutation_token_idx.min().item()}, "
                    f"max={mutation_token_idx.max().item()}"
                )
            
            batch_indices = torch.arange(preds.shape[0], device=preds.device)
            preds = preds[batch_indices, mutation_token_idx, :]
            preds = torch.nn.functional.log_softmax(preds, dim=-1)
            ref_likelihoods = preds[batch_indices, ref_toks]
            alt_likelihoods = preds[batch_indices, alt_toks]
            likelihood_ratios = ref_likelihoods - alt_likelihoods
        return MutationPredictionOutput(
            ref_likelihoods=ref_likelihoods.cpu().numpy(),
            alt_likelihoods=alt_likelihoods.cpu().numpy(),
            likelihood_ratios=likelihood_ratios.cpu().numpy(),
            ids=ids,
        )

    def predict_synom_agg_missense(self, batch, ids=None) -> Dict[str, np.ndarray]:
        masked_output = self.model(batch)
        poses = batch[MetadataFields.MUTATION_TOKEN_IDX].view(-1) # B
        ref_synom_mask = batch[MetadataFields.REF_SYNOM_MASK] # B, V
        alt_synom_mask = batch[MetadataFields.ALT_SYNOM_MASK] # B, V
        preds = masked_output.logits[torch.arange(masked_output.logits.size(0)), poses] # B, V
        ref_masked_preds = torch.where(ref_synom_mask.bool(), preds, torch.full_like(preds, float('-inf')))
        alt_masked_preds = torch.where(alt_synom_mask.bool(), preds, torch.full_like(preds, float('-inf')))
        ref_synom_preds = torch.max(ref_masked_preds, dim=-1).values
        alt_synom_preds = torch.max(alt_masked_preds, dim=-1).values
        preds = torch.sigmoid(ref_synom_preds - alt_synom_preds)

        return DownstreamPredictionOutput(
            predictions=preds.cpu().numpy(),
            ids=ids,
        )
        
    def extract_embeddings(self, batch, ids=None) -> Dict[str, np.ndarray]:
        """
        Extract sequence embeddings from the [CLS] token.
        
        Args:
            batch: Dictionary containing input_ids and attention_mask.
            ids: Optional sequence identifiers.
            
        Returns:
            EmbeddingOutput with embeddings array of shape (batch_size, hidden_size).
        """
        with torch.no_grad():
            output = self.model(batch, return_hidden_states=True)
            embeddings = output.all_hidden_states[-1]
            if embeddings.dtype != torch.float:
                embeddings = embeddings.float()
            embeddings = embeddings[:, 0, :].cpu().numpy()
        return EmbeddingOutput(embeddings=embeddings, ids=ids)
    
    def predict_fitness(self, batch, ids=None) -> Dict[str, np.ndarray]:
        """
        Compute sequence fitness as mean log-likelihood of tokens.
        
        Uses parallel scoring (all positions evaluated simultaneously).
        
        Args:
            batch: Dictionary containing INPUT_IDS field.
            ids: Optional sequence identifiers.
            
        Returns:
            FitnessPredictionOutput with fitness scores.
        """
        if MetadataFields.INPUT_IDS not in batch:
            raise ValueError(f"Batch missing required field: {MetadataFields.INPUT_IDS}")
        
        with torch.no_grad():
            output = self.model(batch)
            preds = output.logits
            if preds.dtype != torch.float:
                preds = preds.float()
            
            log_probs = torch.nn.functional.log_softmax(preds, dim=-1)
            selected_log_probs = log_probs.gather(-1, batch[MetadataFields.INPUT_IDS].unsqueeze(-1)).squeeze(-1)
            non_padding_mask = batch[MetadataFields.INPUT_IDS] != self.tokenizer.pad_token_id
            masked_log_probs = selected_log_probs * non_padding_mask
            log_likelihoods_sum = masked_log_probs.sum(dim=-1)
            non_padding_counts = non_padding_mask.sum(dim=-1).clamp(min=1)
            log_likelihoods_mean = (log_likelihoods_sum / non_padding_counts).cpu().numpy()
        return FitnessPredictionOutput(fitness=log_likelihoods_mean, ids=ids)

    def predict_fitness_autoregressive(self, batch, ids=None) -> FitnessPredictionOutput:
        """
        Compute fitness by masking one position at a time (O(L) forward passes).
        More accurate than parallel scoring for MLMs but much slower.
        """
        if MetadataFields.INPUT_IDS not in batch:
            raise ValueError(f"Batch missing required field: {MetadataFields.INPUT_IDS}")
        
        input_ids = batch[MetadataFields.INPUT_IDS]
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        pad_mask = input_ids != self.tokenizer.pad_token_id
        cls_mask = input_ids != self.tokenizer.cls_token_id
        sep_mask = input_ids != self.tokenizer.sep_token_id
        valid_mask = pad_mask & cls_mask & sep_mask
        
        log_likelihoods_sum = torch.zeros(batch_size, device=device)
        token_counts = torch.zeros(batch_size, device=device)
        
        with torch.no_grad():
            for pos in range(seq_len):
                pos_valid = valid_mask[:, pos]
                if not pos_valid.any():
                    continue
                
                masked_input_ids = input_ids.clone()
                masked_input_ids[:, pos] = self.tokenizer.mask_token_id
                
                masked_batch = {k: v for k, v in batch.items()}
                masked_batch[MetadataFields.INPUT_IDS] = masked_input_ids
                
                output = self.model(masked_batch)
                logits = output.logits
                if logits.dtype != torch.float:
                    logits = logits.float()
                
                log_probs = torch.nn.functional.log_softmax(logits[:, pos, :], dim=-1)
                true_tokens = input_ids[:, pos]
                selected_log_probs = log_probs.gather(-1, true_tokens.unsqueeze(-1)).squeeze(-1)
                
                log_likelihoods_sum += selected_log_probs * pos_valid.float()
                token_counts += pos_valid.float()
        
        token_counts = token_counts.clamp(min=1)
        log_likelihoods_mean = (log_likelihoods_sum / token_counts).cpu().numpy()
        
        return FitnessPredictionOutput(fitness=log_likelihoods_mean, ids=ids)

    def predict_fitness_left_to_right(self, batch, ids=None) -> FitnessPredictionOutput:
        """
        Compute P(seq) = P(x1) * P(x2|x1) * ... by masking current and right positions.
        Simulates autoregressive LM on a bidirectional model. O(L) forward passes.
        """
        if MetadataFields.INPUT_IDS not in batch:
            raise ValueError(f"Batch missing required field: {MetadataFields.INPUT_IDS}")
        
        input_ids = batch[MetadataFields.INPUT_IDS]
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        pad_mask = input_ids != self.tokenizer.pad_token_id
        cls_mask = input_ids != self.tokenizer.cls_token_id
        sep_mask = input_ids != self.tokenizer.sep_token_id
        valid_mask = pad_mask & cls_mask & sep_mask
        
        log_likelihoods_sum = torch.zeros(batch_size, device=device)
        token_counts = torch.zeros(batch_size, device=device)
        
        with torch.no_grad():
            for pos in range(seq_len):
                pos_valid = valid_mask[:, pos]
                if not pos_valid.any():
                    continue
                
                # Mask current position and everything to the right
                masked_input_ids = input_ids.clone()
                right_mask = valid_mask.clone()
                right_mask[:, :pos] = False
                masked_input_ids = torch.where(
                    right_mask,
                    torch.full_like(masked_input_ids, self.tokenizer.mask_token_id),
                    masked_input_ids
                )
                
                masked_batch = {k: v for k, v in batch.items()}
                masked_batch[MetadataFields.INPUT_IDS] = masked_input_ids
                
                output = self.model(masked_batch)
                logits = output.logits
                if logits.dtype != torch.float:
                    logits = logits.float()
                
                log_probs = torch.nn.functional.log_softmax(logits[:, pos, :], dim=-1)
                true_tokens = input_ids[:, pos]
                selected_log_probs = log_probs.gather(-1, true_tokens.unsqueeze(-1)).squeeze(-1)
                
                log_likelihoods_sum += selected_log_probs * pos_valid.float()
                token_counts += pos_valid.float()
        
        token_counts = token_counts.clamp(min=1)
        log_likelihoods_mean = (log_likelihoods_sum / token_counts).cpu().numpy()
        
        return FitnessPredictionOutput(fitness=log_likelihoods_mean, ids=ids)

    def predict_downstream(self, batch, ids=None) -> DownstreamPredictionOutput:
        """Predict using cross-attention head (requires use_downstream_head=True)."""
        with torch.no_grad():
            if not hasattr(self.model.model, 'cross_attention_head') or not hasattr(self.model.model, 'cross_attention_input_proj'):
                raise ValueError("Model does not have downstream cross-attention heads. Ensure the model was trained with use_downstream_head=True.")
            
            if MetadataFields.ATTENTION_MASK not in batch:
                raise ValueError(f"Batch missing required field: {MetadataFields.ATTENTION_MASK}")
            
            output = self.model(batch)
            hidden_states = output.last_hidden_state
            attention_mask = batch[MetadataFields.ATTENTION_MASK]
            
            projected_states = self.model.model.cross_attention_input_proj(hidden_states)
            query_input = projected_states[:, 0, :]
            key_value_input = projected_states
            preds = self.model.model.cross_attention_head(query_input, key_value_input, attention_mask)
            
            loss_type = getattr(self.model.hparams, 'loss_type', 'regression')
            
            if loss_type == "classification":
                preds_float = preds.float()
                probabilities = torch.nn.functional.softmax(preds_float, dim=-1).cpu().numpy()
                predicted_classes = preds_float.argmax(dim=-1).cpu().numpy()
                preds_np = preds_float.cpu().numpy()
                
                return DownstreamPredictionOutput(
                    predictions=preds_np,
                    probabilities=probabilities,
                    predicted_classes=predicted_classes,
                    ids=ids
                )
            else:
                preds = preds.squeeze(-1).float().cpu().numpy()
                return DownstreamPredictionOutput(predictions=preds, ids=ids)

    def _predict_step(self, batch, batch_idx):
        """Dispatch to appropriate prediction method based on task type."""
        ids = None
        if MetadataFields.ID in batch:
            ids = batch[MetadataFields.ID]
            del batch[MetadataFields.ID]
        
        if self.task_type == TaskTypes.MUTATION_PREDICTION:
            predict = self.predict_mutation
        elif self.task_type == TaskTypes.MASKED_LANGUAGE_MODELING:
            predict = self.predict_mlm
        elif self.task_type == TaskTypes.FITNESS_PREDICTION:
            predict = self.predict_fitness
        elif self.task_type == TaskTypes.EMBEDDING_PREDICTION:
            predict = self.extract_embeddings
        elif self.task_type == TaskTypes.DOWNSTREAM_PREDICTION:
            predict = self.predict_downstream
        elif self.task_type == TaskTypes.MISSENSE_PREDICTION:
            predict = self.predict_synom_agg_missense
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")
        
        outputs = predict(batch, ids)
        return outputs

    def build_generation_batch(
        self,
        seqs: List[str],
        target_num_codons: int,
        context_length: int,
    ) -> Dict[str, torch.Tensor]:
        """Build batch with [CLS] + tokens + [MASK]*remaining + [SEP] structure."""
        if not isinstance(seqs, list):
            raise TypeError(f"seqs must be a list, got {type(seqs).__name__}")
        if target_num_codons <= 0:
            raise ValueError(f"target_num_codons must be positive, got {target_num_codons}")
        if context_length <= 2:
            raise ValueError(f"context_length must be > 2, got {context_length}")
        
        batch_input_ids: List[np.ndarray] = []
        batch_attention_masks: List[np.ndarray] = []
        
        for seq_idx, s in enumerate(seqs):
            if not isinstance(s, str):
                raise TypeError(f"seqs[{seq_idx}] must be a string, got {type(s).__name__}")
            
            filled_tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(s))
            filled_tokens = np.asarray(filled_tokens, dtype=np.int64)
            filled_tokens = filled_tokens[:max(0, target_num_codons)]
            num_filled = int(filled_tokens.shape[0])
            
            remaining = max(0, target_num_codons - num_filled)
            composed = np.concatenate(
                [
                    np.asarray([self.tokenizer.cls_token_id], dtype=np.int64),
                    filled_tokens,
                    np.full((remaining,), self.tokenizer.mask_token_id, dtype=np.int64),
                    np.asarray([self.tokenizer.sep_token_id], dtype=np.int64),
                ],
                axis=0,
            )
            
            seq_len = int(min(len(composed), context_length))
            attn = np.zeros((context_length,), dtype=np.int64)
            attn[:seq_len] = 1
            
            if len(composed) > context_length:
                composed = composed[:context_length]
                composed[-1] = self.tokenizer.sep_token_id
            elif len(composed) < context_length:
                pad_len = context_length - len(composed)
                pad = np.full((pad_len,), self.tokenizer.pad_token_id, dtype=np.int64)
                composed = np.concatenate([composed, pad], axis=0)
            
            batch_input_ids.append(composed)
            batch_attention_masks.append(attn)
        
        input_ids = torch.tensor(np.stack(batch_input_ids, axis=0), dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(np.stack(batch_attention_masks, axis=0), dtype=torch.long, device=self.device)
        
        return {
            MetadataFields.INPUT_IDS: input_ids,
            MetadataFields.ATTENTION_MASK: attention_mask,
            MetadataFields.LABELS: input_ids.clone(),
        }

    @torch.no_grad()
    def get_next_codon_logits(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Get logits at the first [MASK] position for each sequence."""
        if MetadataFields.INPUT_IDS not in batch:
            raise ValueError(f"Batch missing required field: {MetadataFields.INPUT_IDS}")
        
        input_ids = batch[MetadataFields.INPUT_IDS]
        if input_ids.size(0) == 0:
            return torch.zeros((0, self.tokenizer.vocab_size), dtype=torch.float32, device=self.device)
        
        outputs = self.model(batch)
        logits = outputs.logits
        
        mask_id = self.tokenizer.mask_token_id
        mask_positions_bool = input_ids == mask_id
        
        has_mask = mask_positions_bool.any(dim=1)
        if not has_mask.all():
            missing_mask_indices = (~has_mask).nonzero(as_tuple=True)[0].tolist()
            raise ValueError(
                f"Sequences at indices {missing_mask_indices} have no mask tokens. "
                f"Cannot determine next codon position."
            )
        
        mask_positions = mask_positions_bool.int().argmax(dim=1)
        batch_indices = torch.arange(logits.size(0), device=self.device)
        next_token_logits = logits[batch_indices, mask_positions]
        
        return next_token_logits

    @torch.no_grad()
    def get_masked_codon_logits(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Get logits for all [MASK] positions. All sequences must have same mask count."""
        if MetadataFields.INPUT_IDS not in batch:
            raise ValueError(f"Batch missing required field: {MetadataFields.INPUT_IDS}")
        
        input_ids = batch[MetadataFields.INPUT_IDS]
        if input_ids.size(0) == 0:
            return torch.zeros((0, 0, self.tokenizer.vocab_size), dtype=torch.float32, device=self.device)
        
        outputs = self.model(batch)
        logits = outputs.logits
        
        mask_id = self.tokenizer.mask_token_id
        mask_positions_bool = input_ids == mask_id
        
        mask_counts = mask_positions_bool.sum(dim=1)
        if not (mask_counts == mask_counts[0]).all():
            raise ValueError(
                f"All sequences must have the same number of masked positions. "
                f"Found varying counts: {mask_counts.tolist()}"
            )
        
        num_masked = mask_counts[0].item()
        if num_masked == 0:
            batch_size = logits.size(0)
            vocab_size = logits.size(-1)
            return torch.zeros((batch_size, 0, vocab_size), dtype=logits.dtype, device=self.device)
        
        batch_size = logits.size(0)
        vocab_size = logits.size(-1)
        masked_logits = logits[mask_positions_bool].view(batch_size, num_masked, vocab_size)
        
        return masked_logits

    def build_random_mask_batch(
        self,
        seqs: List[str],
        mask_ratio: float,
        context_length: int,
        mask_positions: Optional[List[List[int]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Build batch with random or specified positions masked."""
        if not isinstance(seqs, list):
            raise TypeError(f"seqs must be a list, got {type(seqs).__name__}")
        if not seqs:
            raise ValueError("seqs cannot be empty")
        if not 0.0 <= mask_ratio <= 1.0:
            raise ValueError(f"mask_ratio must be between 0.0 and 1.0, got {mask_ratio}")
        if context_length <= 2:
            raise ValueError(f"context_length must be > 2, got {context_length}")
        if mask_positions is not None and len(mask_positions) != len(seqs):
            raise ValueError(f"mask_positions must be the same length as seqs, got {len(mask_positions)} and {len(seqs)}")
        batch_input_ids: List[np.ndarray] = []
        batch_attention_masks: List[np.ndarray] = []
        batch_labels: List[np.ndarray] = []
        batch_mask_indices: List[List[int]] = []
        
        for seq_idx, s in enumerate(seqs):
            if not isinstance(s, str):
                raise TypeError(f"seqs[{seq_idx}] must be a string, got {type(s).__name__}")
            if not s:
                raise ValueError(f"seqs[{seq_idx}] is empty. DNA sequences must be non-empty.")
            
            tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(s))
            tokens = np.asarray(tokens, dtype=np.int64)
            num_codons = len(tokens)
            if num_codons == 0:
                raise ValueError(
                    f"seqs[{seq_idx}] produced no tokens. "
                    f"Ensure the sequence contains valid codon triplets."
                )
            
            if mask_positions is not None:
                positions_to_mask = mask_positions[seq_idx]
                if not positions_to_mask:
                    raise ValueError(f"mask_positions[{seq_idx}] is empty. Must specify at least one position.")
                if min(positions_to_mask) < 0 or max(positions_to_mask) >= num_codons:
                    raise ValueError(
                        f"mask_positions[{seq_idx}] contains invalid positions {positions_to_mask}. "
                        f"Valid range: [0, {num_codons - 1}]"
                    )
            else:
                num_to_mask = max(1, int(num_codons * mask_ratio))
                num_to_mask = min(num_to_mask, num_codons)
                positions_to_mask = sorted(random.sample(range(num_codons), num_to_mask))
            
            batch_mask_indices.append(positions_to_mask)
            
            labels = tokens.copy()
            masked_tokens = tokens.copy()
            masked_tokens[positions_to_mask] = self.tokenizer.mask_token_id
            
            composed = np.concatenate(
                [
                    np.asarray([self.tokenizer.cls_token_id], dtype=np.int64),
                    masked_tokens,
                    np.asarray([self.tokenizer.sep_token_id], dtype=np.int64),
                ],
                axis=0,
            )
            
            labels_with_special = np.concatenate(
                [
                    np.asarray([self.tokenizer.cls_token_id], dtype=np.int64),
                    labels,
                    np.asarray([self.tokenizer.sep_token_id], dtype=np.int64),
                ],
                axis=0,
            )
            
            seq_len = int(min(len(composed), context_length))
            attn = np.zeros((context_length,), dtype=np.int64)
            attn[:seq_len] = 1
            
            if len(composed) > context_length:
                composed = composed[:context_length]
                composed[-1] = self.tokenizer.sep_token_id
                labels_with_special = labels_with_special[:context_length]
                labels_with_special[-1] = self.tokenizer.sep_token_id
            elif len(composed) < context_length:
                pad_len = context_length - len(composed)
                composed = np.concatenate([composed, np.full((pad_len,), self.tokenizer.pad_token_id, dtype=np.int64)], axis=0)
                labels_with_special = np.concatenate([labels_with_special, np.full((pad_len,), self.tokenizer.pad_token_id, dtype=np.int64)], axis=0)
            
            batch_input_ids.append(composed)
            batch_attention_masks.append(attn)
            batch_labels.append(labels_with_special)
        
        input_ids = torch.tensor(np.stack(batch_input_ids, axis=0), dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(np.stack(batch_attention_masks, axis=0), dtype=torch.long, device=self.device)
        labels = torch.tensor(np.stack(batch_labels, axis=0), dtype=torch.long, device=self.device)        
        return {
            MetadataFields.INPUT_IDS: input_ids,
            MetadataFields.ATTENTION_MASK: attention_mask,
            MetadataFields.LABELS: labels,
        }

    @torch.no_grad()
    def predict_and_fill_masks(
        self,
        batch: Dict[str, torch.Tensor],
        amino_acid_sequence: Optional[str] = None,
        temperature: float = 1.0,
        sample: bool = False,
    ) -> List[str]:
        """Fill [MASK] positions, optionally constrained to synonymous codons."""
        if temperature <= 0 and sample:
            raise ValueError(f"temperature must be positive when sample=True, got {temperature}")
        
        required_fields = [MetadataFields.INPUT_IDS, MetadataFields.LABELS]
        for field in required_fields:
            if field not in batch:
                raise ValueError(f"Batch missing required field: {field}")
        
        input_ids = batch[MetadataFields.INPUT_IDS]
        labels = batch[MetadataFields.LABELS]
        mask = batch[MetadataFields.INPUT_IDS] == self.tokenizer.mask_token_id
        if input_ids.size(0) == 0:
            return []
        
        outputs = self.model(batch)
        logits = outputs.logits
        
        filled_sequences: List[str] = []
        
        for i in range(input_ids.size(0)):
            seq_tokens = labels[i, :].cpu().numpy()
            sep_pos = np.where(seq_tokens == self.tokenizer.sep_token_id)[0]
            seq_len = sep_pos[0] if len(sep_pos) > 0 else len(seq_tokens)
            seq_tokens = seq_tokens[1:seq_len].copy()
            
            mask_indices = mask[i].nonzero(as_tuple=True)[0].tolist()
            for pos in mask_indices:
                if pos < 1 or pos >= seq_len:
                    continue
                token_logits = logits[i, pos].clone()
                codon_idx = pos - 1  # Adjust for CLS offset
                
                if amino_acid_sequence is not None and codon_idx < len(amino_acid_sequence):
                    aa = amino_acid_sequence[codon_idx]
                    synonymous_codons = self.tokenizer.aa_to_codon.get(aa)
                    if synonymous_codons:
                        valid_token_ids = [self.tokenizer.encoder[c] for c in synonymous_codons]
                        constraint_mask = torch.full_like(token_logits, float('-inf'))
                        constraint_mask[valid_token_ids] = 0
                        token_logits = token_logits + constraint_mask
                
                if sample and temperature > 0:
                    probs = torch.softmax(token_logits / temperature, dim=-1)
                    predicted_token = torch.multinomial(probs, 1).item()
                else:
                    predicted_token = token_logits.argmax().item()
                seq_tokens[codon_idx] = predicted_token
            
            codons = [self.tokenizer.decoder[int(t)] for t in seq_tokens]
            dna_seq = "".join(c for c in codons if c not in self.tokenizer.special_tokens)
            filled_sequences.append(dna_seq)
        
        return filled_sequences

    @torch.no_grad()
    def generate_bidirectional(
        self,
        dna_seqs: List[str],
        amino_acid_sequence: str,
        context_length: int,
        mask_ratio: float = 0.15,
        num_iterations: int = 10,
        temperature_start: float = 1.2,
        temperature_end: float = 0.5,
        bf16: bool = False,
        full_mask_argmax_indices: Optional[List[int]] = None,
        full_mask_sample_indices: Optional[List[int]] = None,
        sample_indices: Optional[List[int]] = None,
        batch_size: int = 32,
    ) -> List[str]:
        """Iteratively mask and predict positions with temperature annealing."""
        if len(dna_seqs) == 0:
            return []
        
        current_seqs = list(dna_seqs)
        num_codons = len(amino_acid_sequence)
        num_to_mask_per_iter = max(1, int(num_codons * mask_ratio))
        num_seqs = len(current_seqs)
        
        full_mask_argmax_set = set(full_mask_argmax_indices) if full_mask_argmax_indices else set()
        full_mask_sample_set = set(full_mask_sample_indices) if full_mask_sample_indices else set()
        full_mask_all = full_mask_argmax_set | full_mask_sample_set
        sample_set = set(sample_indices) if sample_indices is not None else full_mask_sample_set
        
        position_pools = [list(range(num_codons)) for _ in range(num_seqs)]
        for pool in position_pools:
            random.shuffle(pool)
        
        def process_in_batches(
            seqs: List[str], 
            mask_positions: List[List[int]], 
            temperature: float, 
            use_sample: bool,
        ) -> List[str]:
            """Process sequences in mini-batches and return filled sequences."""
            results = [None] * len(seqs)
            
            for batch_start in range(0, len(seqs), batch_size):
                batch_end = min(batch_start + batch_size, len(seqs))
                batch_seqs = seqs[batch_start:batch_end]
                batch_mask_positions = mask_positions[batch_start:batch_end]
                
                dtype = torch.bfloat16 if bf16 else torch.float32
                with torch.autocast(device_type=self.device.type, dtype=dtype):
                    batch = self.build_random_mask_batch(
                        batch_seqs, mask_ratio, context_length, mask_positions=batch_mask_positions
                    )
                    filled = self.predict_and_fill_masks(
                        batch, amino_acid_sequence, temperature=temperature, sample=use_sample
                    )
                
                for i, filled_seq in enumerate(filled):
                    results[batch_start + i] = filled_seq
            
            return results
        
        for iter_idx in range(num_iterations):
            if num_iterations > 1:
                progress = iter_idx / (num_iterations - 1)
                current_temp = temperature_start + (temperature_end - temperature_start) * progress
            else:
                current_temp = temperature_end
            
            # First iteration: handle fully-masked sequences specially
            if iter_idx == 0 and full_mask_all:
                if full_mask_argmax_set:
                    argmax_indices = sorted(full_mask_argmax_set)
                    argmax_seqs = [current_seqs[i] for i in argmax_indices]
                    argmax_mask_positions = [list(range(num_codons)) for _ in argmax_indices]
                    for idx in argmax_indices:
                        position_pools[idx] = []
                    filled = process_in_batches(argmax_seqs, argmax_mask_positions, temperature=1.0, use_sample=False)
                    for i, idx in enumerate(argmax_indices):
                        current_seqs[idx] = filled[i]
                
                if full_mask_sample_set:
                    sample_indices_list = sorted(full_mask_sample_set)
                    sample_seqs = [current_seqs[i] for i in sample_indices_list]
                    sample_mask_positions = [list(range(num_codons)) for _ in sample_indices_list]
                    for idx in sample_indices_list:
                        position_pools[idx] = []
                    filled = process_in_batches(sample_seqs, sample_mask_positions, temperature=temperature_start, use_sample=True)
                    for i, idx in enumerate(sample_indices_list):
                        current_seqs[idx] = filled[i]
                
                remaining_indices = [i for i in range(num_seqs) if i not in full_mask_all]
                if remaining_indices:
                    remaining_seqs = [current_seqs[i] for i in remaining_indices]
                    remaining_mask_positions = []
                    for idx in remaining_indices:
                        if len(position_pools[idx]) == 0:
                            position_pools[idx] = list(range(num_codons))
                            random.shuffle(position_pools[idx])
                        num_to_take = min(num_to_mask_per_iter, len(position_pools[idx]))
                        positions_to_mask = position_pools[idx][:num_to_take]
                        position_pools[idx] = position_pools[idx][num_to_take:]
                        remaining_mask_positions.append(sorted(positions_to_mask))
                    filled = process_in_batches(remaining_seqs, remaining_mask_positions, temperature=current_temp, use_sample=True)
                    for i, idx in enumerate(remaining_indices):
                        current_seqs[idx] = filled[i]
                
                continue
            
            # Regular iterations
            mask_positions_batch = []
            for seq_idx in range(num_seqs):
                if len(position_pools[seq_idx]) == 0:
                    position_pools[seq_idx] = list(range(num_codons))
                    random.shuffle(position_pools[seq_idx])
                num_to_take = min(num_to_mask_per_iter, len(position_pools[seq_idx]))
                positions_to_mask = position_pools[seq_idx][:num_to_take]
                position_pools[seq_idx] = position_pools[seq_idx][num_to_take:]
                mask_positions_batch.append(sorted(positions_to_mask))
            
            if sample_set:
                sample_indices_iter = sorted(sample_set)
                sample_seqs = [current_seqs[i] for i in sample_indices_iter]
                sample_masks = [mask_positions_batch[i] for i in sample_indices_iter]
                filled = process_in_batches(sample_seqs, sample_masks, temperature=current_temp, use_sample=True)
                for i, idx in enumerate(sample_indices_iter):
                    current_seqs[idx] = filled[i]
            
            argmax_indices_iter = [i for i in range(num_seqs) if i not in sample_set]
            if argmax_indices_iter:
                argmax_seqs = [current_seqs[i] for i in argmax_indices_iter]
                argmax_masks = [mask_positions_batch[i] for i in argmax_indices_iter]
                filled = process_in_batches(argmax_seqs, argmax_masks, temperature=1.0, use_sample=False)
                for i, idx in enumerate(argmax_indices_iter):
                    current_seqs[idx] = filled[i]
        
        return current_seqs

    @torch.no_grad()
    def generate_autoregressive(
        self,
        amino_acid_sequence: str,
        context_length: int,
        num_sequences: int = 1,
        seed_codons: Optional[List[str]] = None,
        temperature: float = 1.0,
        sample: bool = False,
        bf16: bool = False,
        batch_size: int = 32,
    ) -> List[str]:
        """Generate sequences using beam search with synonymous codon constraints."""
        num_codons = len(amino_acid_sequence)
        if seed_codons is None:
            seed_codons = [None] * num_codons
        
        beam: List[Tuple[float, str]] = [(0.0, "")]
        
        for pos in range(num_codons):
            if seed_codons[pos] is not None:
                beam = [(score, seq + seed_codons[pos]) for score, seq in beam]
                continue
            
            aa = amino_acid_sequence[pos]
            synonymous_codons = self.tokenizer.aa_to_codon.get(aa)
            if not synonymous_codons:
                raise ValueError(f"No synonymous codons found for amino acid '{aa}' at position {pos}")
            
            current_seqs = [seq for _, seq in beam]
            
            all_logits = []
            for batch_start in range(0, len(current_seqs), batch_size):
                batch_end = min(batch_start + batch_size, len(current_seqs))
                batch_seqs = current_seqs[batch_start:batch_end]
                
                if torch.cuda.is_available():
                    dtype = torch.bfloat16 if bf16 else torch.float32
                    with torch.autocast(device_type="cuda", dtype=dtype):
                        batch = self.build_generation_batch(batch_seqs, num_codons, context_length)
                        batch_logits = self.get_next_codon_logits(batch)
                else:
                    batch = self.build_generation_batch(batch_seqs, num_codons, context_length)
                    batch_logits = self.get_next_codon_logits(batch)
                
                all_logits.append(batch_logits)
            
            logits = torch.cat(all_logits, dim=0)
            
            valid_token_ids = []
            for c in synonymous_codons:
                token_id = self.tokenizer.encoder.get(c)
                if token_id is None:
                    raise ValueError(f"Codon '{c}' not found in tokenizer vocabulary")
                valid_token_ids.append(token_id)
            constraint_mask = torch.full((logits.size(-1),), float('-inf'), device=logits.device)
            constraint_mask[valid_token_ids] = 0
            logits = logits + constraint_mask.unsqueeze(0)
            
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            
            if sample and temperature > 0:
                new_beam = []
                for i, (cum_score, seq) in enumerate(beam):
                    probs = torch.softmax(logits[i] / temperature, dim=-1)
                    sampled_token = torch.multinomial(probs, 1).item()
                    codon = self.tokenizer.decoder.get(sampled_token, "")
                    token_log_prob = float(log_probs[i, sampled_token].item())
                    new_score = cum_score + token_log_prob
                    new_beam.append((new_score, seq + codon))
                beam = new_beam
            else:
                candidates = []
                for i, (cum_score, seq) in enumerate(beam):
                    for token_id in valid_token_ids:
                        codon = self.tokenizer.decoder.get(token_id, "")
                        token_log_prob = float(log_probs[i, token_id].item())
                        new_score = cum_score + token_log_prob
                        candidates.append((new_score, seq + codon))
                candidates.sort(key=lambda x: x[0], reverse=True)
                beam = candidates[:num_sequences]
        
        beam.sort(key=lambda x: x[0], reverse=True)
        return [seq for _, seq in beam[:num_sequences]]
