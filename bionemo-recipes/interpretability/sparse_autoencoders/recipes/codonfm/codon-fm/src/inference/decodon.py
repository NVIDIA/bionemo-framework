from typing import Optional, Dict, Any, List, Union, Tuple
from pathlib import Path
import json

import torch
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist
from safetensors.torch import load_file

from transformers.generation.logits_process import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
    MaxLengthCriteria,
)
from transformers.generation.beam_search import BeamSearchScorer

from src.tokenizer import Tokenizer
from src.tokenizer.mappings import AA_TABLE
from src.inference.base import BaseInference
from src.inference.task_types import TaskTypes
from src.inference.model_outputs import (
    EmbeddingOutput,
    FitnessPredictionOutput,
    MutationPredictionOutput,
    NextCodonPredictionOutput,
    SequenceGenerationOutput,
)
from src.models.decodon_pl import DecodonPL
from src.data.metadata import MetadataFields
from src.data.preprocess.codon_sequence import process_item_clm

# Tokenizer constants
# Special tokens: CLS=0, SEP=1, UNK=2, PAD=3, MASK=4
# Codons: 5-68 (64 codons)
SPECIAL_TOKEN_COUNT = 5
CODON_COUNT = 64
CODON_VOCAB_SIZE = SPECIAL_TOKEN_COUNT + CODON_COUNT  # 69
CODON_TOKEN_START = SPECIAL_TOKEN_COUNT  # 5
CODON_TOKEN_END = CODON_VOCAB_SIZE - 1  # 68


class DecodonInference(BaseInference):
    """Inference class for Decodon models."""
    
    def __init__(
        self, 
        model_path: str, 
        task_type: str, 
        organism_tokens_file: str = None,
        generation_config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(model_path, task_type)
        self.organism_tokens_file = organism_tokens_file
        # Store generation config for use in generate_from_batch
        self.generation_config = generation_config or {}
    
    def configure_model(self):
        """Load model weights from .ckpt or .safetensors file."""
        if self.model is not None:
            return
        self.tokenizer = Tokenizer(organism_tokens_file=self.organism_tokens_file)
        
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
                    ckpt = torch.load(self.model_path, map_location="cpu")
                    hparams = ckpt.get("hyper_parameters")
                    state_dict = ckpt.get("state_dict")
                    broadcasted_objects = [state_dict, hparams]
                dist.broadcast_object_list(broadcasted_objects, src=0)
                state_dict, hparams = broadcasted_objects
            else:
                ckpt = torch.load(self.model_path, map_location="cpu")
                hparams = ckpt.get("hyper_parameters")
                state_dict = ckpt.get("state_dict")
        else:
            raise ValueError(
                f"Unsupported model file type: {suffix}. Expected .ckpt or .safetensors"
            )

        # The hparams from lightning checkpoint might be nested.
        if hparams is not None and "hparams" in hparams:
            hparams = hparams["hparams"]

        if hparams is None:
            raise ValueError(
                f"Failed to load hyperparameters from checkpoint at '{self.model_path}'. "
                f"The checkpoint may be corrupted or in an unsupported format."
            )
        
        if state_dict is None:
            raise ValueError(
                f"Failed to load state_dict from checkpoint at '{self.model_path}'. "
                f"The checkpoint may be corrupted or in an unsupported format."
            )

        def dummy_optimizer(params):
            return torch.optim.Adam(params)
        
        hparams['optimizer'] = dummy_optimizer
        hparams['scheduler'] = None

        self.model = DecodonPL(**hparams)
        self.model.configure_model(state_dict=state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        
    def extract_embeddings(self, batch, ids=None) -> EmbeddingOutput:
        """
        Extract sequence embeddings from the model.
        
        Uses the last non-padding token position (similar to CLS token approach).
        
        Args:
            batch: Dictionary containing input_ids and attention_mask tensors.
            ids: Optional sequence identifiers.
            
        Returns:
            EmbeddingOutput with embeddings array of shape (batch_size, hidden_size).
        """
        with torch.no_grad():
            outputs = self.model(batch_data=batch, return_hidden_states=True)
            hidden_states = outputs.all_hidden_states[-1]
            if hidden_states.dtype != torch.float:
                hidden_states = hidden_states.float()
            
            input_ids = batch[MetadataFields.INPUT_IDS]
            non_pad_mask = input_ids != self.tokenizer.pad_token_id
            last_positions = non_pad_mask.sum(dim=1) - 1
            
            if (last_positions < 0).any():
                raise ValueError("Found sequence(s) with all padding tokens - cannot extract embeddings")
            
            batch_indices = torch.arange(input_ids.shape[0], device=input_ids.device)
            embeddings = hidden_states[batch_indices, last_positions, :].cpu().numpy()
            
        return EmbeddingOutput(embeddings=embeddings, ids=ids)

    def predict_mutation(self, batch, ids=None) -> MutationPredictionOutput:
        """
        Score variants by comparing log probabilities at the mutation position.
        
        Args:
            batch: Dictionary containing reference and alternative sequence data.
            ids: Optional sequence identifiers.
            
        Returns:
            MutationPredictionOutput with ref/alt likelihoods and ratios.
        """
        with torch.no_grad():
            attention_mask = batch[MetadataFields.ATTENTION_MASK][:, :-1]
            input_ids = batch[MetadataFields.INPUT_IDS][:, :-1]
            ref_batch = {
                MetadataFields.INPUT_IDS: input_ids,
                MetadataFields.ATTENTION_MASK: attention_mask
            }
            
            # Autoregressive: logits[i] predicts token[i+1], so adjust index
            mutation_token_idx = batch[MetadataFields.MUTATION_TOKEN_IDX].squeeze()
            mutation_logit_idx = mutation_token_idx - 1
            
            if (mutation_logit_idx < 0).any():
                raise ValueError("Mutation at position 0 cannot be scored—no prior context available")
            
            ref_codon_toks = batch[MetadataFields.REF_CODON_TOKS]
            alt_codon_toks = batch[MetadataFields.ALT_CODON_TOKS]
            
            ref_output = self.model(batch_data=ref_batch)
            logits = ref_output.logits
            
            batch_indices = torch.arange(logits.shape[0], device=logits.device)
            mutation_logits = logits[batch_indices, mutation_logit_idx, :]
            log_probs = torch.nn.functional.log_softmax(mutation_logits, dim=-1)
            
            ref_log_probs = log_probs[batch_indices, ref_codon_toks]
            alt_log_probs = log_probs[batch_indices, alt_codon_toks]
            
            ref_likelihoods = ref_log_probs.cpu().numpy()
            alt_likelihoods = alt_log_probs.cpu().numpy()
            likelihood_ratios = ref_likelihoods - alt_likelihoods
            
        return MutationPredictionOutput(
            ref_likelihoods=ref_likelihoods,
            alt_likelihoods=alt_likelihoods,
            likelihood_ratios=likelihood_ratios,
            ids=ids
        )
    
    def predict_next_codon(self, batch, ids=None) -> NextCodonPredictionOutput:
        """
        Predict the next codon in a sequence.
        
        Flattens predictions and targets for confusion score calculation.
        
        Args:
            batch: Dictionary containing input_ids, labels, and attention_mask.
            ids: Optional sequence identifiers.
            
        Returns:
            NextCodonPredictionOutput with flattened predictions and labels.
        """
        with torch.no_grad():
            output = self.model(batch)
            preds = output.logits
            targets = batch[MetadataFields.LABELS]
            attention_mask = batch[MetadataFields.ATTENTION_MASK]
            
            batch_size, seq_len, vocab_size = preds.shape
            preds_flat = preds.view(-1, vocab_size)
            targets_flat = targets.view(-1)
            mask_flat = attention_mask.view(-1)
            
            valid_mask = (targets_flat != self.model.hparams.ignore_index) & (mask_flat > 0)
            preds_valid = preds_flat[valid_mask]
            targets_valid = targets_flat[valid_mask]
            
            if ids is not None:
                ids_np = np.array(ids)
                ids_expanded = np.repeat(ids_np, seq_len)
                ids_valid = ids_expanded[valid_mask.cpu().numpy()]
            else:
                ids_valid = None
            
            return NextCodonPredictionOutput(
                preds=preds_valid.cpu().float().numpy(),
                labels=targets_valid.cpu().numpy(),
                ids=ids_valid
            )
    
    def generate_from_batch(
        self, 
        batch: Dict[str, torch.Tensor], 
        ids: Optional[List] = None,
    ) -> SequenceGenerationOutput:
        """
        Generate sequences from a batch of organism-ids prompts.
        
        Returns:
            SequenceGenerationOutput with generated_ids (padded to max length) and ids.
            No decoding is performed - only token IDs are returned for efficient storage.
        """
        # Get generation config (set during __init__ from runner.py args)
        max_new_tokens = self.generation_config.get('max_new_tokens', 2048)
        temperature = self.generation_config.get('temperature', 0.9)
        top_k = self.generation_config.get('top_k', 50)
        top_p = self.generation_config.get('top_p', 0.95)
        do_sample = self.generation_config.get('do_sample', True)
        use_stop_codons = self.generation_config.get('use_stop_codons', True)
        

        input_ids = batch[MetadataFields.INPUT_IDS]
        attention_mask = batch.get(MetadataFields.ATTENTION_MASK, None)
        # Get stop codon IDs if requested
        eos_token_id = self.get_stop_codon_token_ids("dna") if use_stop_codons else None
        
        with torch.no_grad():
            self.model.model.eval()
            
            # Generate sequences
            generated_ids = self.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                eos_token_id=eos_token_id,
            )
            
            # Pad all sequences to max_length
            # max_length = prompt length + max_new_tokens
            max_length = input_ids.shape[1] + max_new_tokens
            
            # Truncate if sequences are longer than max_length
            if generated_ids.shape[1] > max_length:
                generated_ids = generated_ids[:, :max_length]
            
            pad_length = max_length - generated_ids.shape[1]
            # pads on the right with pad_token_id
            if pad_length > 0:
                padded_ids = F.pad(
                    generated_ids,
                    (0, pad_length),
                    mode='constant',
                    value=self.tokenizer.pad_token_id
                )
            else:
                padded_ids = generated_ids

            generated_ids_np = padded_ids.cpu().numpy()
        
        return SequenceGenerationOutput(
            generated_ids=generated_ids_np,
            ids=ids
        )

    def _predict_step(self, batch, batch_idx):
        """Dispatch to appropriate prediction method based on task type."""
        ids = None
        if MetadataFields.ID in batch:
            ids = batch[MetadataFields.ID]
            del batch[MetadataFields.ID]
        
        if self.task_type == TaskTypes.EMBEDDING_PREDICTION:
            predict = self.extract_embeddings
        elif self.task_type == TaskTypes.MUTATION_PREDICTION:
            predict = self.predict_mutation
        elif self.task_type == TaskTypes.NEXT_CODON_PREDICTION:
            predict = self.predict_next_codon
        elif self.task_type == TaskTypes.SEQUENCE_GENERATION:
            predict = self.generate_from_batch
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")
        
        outputs = predict(batch, ids)
        return outputs

    def prepare_inputs_for_generation(
        self, input_ids: torch.LongTensor, attention_mask: Optional[torch.LongTensor] = None, **kwargs
    ) -> Dict[str, Any]:
        """
        Prepare inputs for generation.
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            **kwargs: Additional model-specific kwargs  
        """
        # Pad to make sequence length divisible by 8 (required for xformers)
        batch_size, seq_len = input_ids.shape
        pad_to_multiple = 8
        
        if seq_len % pad_to_multiple != 0:
            pad_len = pad_to_multiple - (seq_len % pad_to_multiple)
            
            padding = torch.full(
                (batch_size, pad_len),
                self.model.hparams.pad_token_id,
                dtype=input_ids.dtype,
                device=input_ids.device
            )
            input_ids = torch.cat([input_ids, padding], dim=1)
            
            if attention_mask is None:
                attention_mask = torch.ones(
                    (batch_size, seq_len),
                    dtype=torch.long,
                    device=input_ids.device
                )
            
            mask_padding = torch.zeros(
                (batch_size, pad_len),
                dtype=attention_mask.dtype,
                device=attention_mask.device
            )
            attention_mask = torch.cat([attention_mask, mask_padding], dim=1)
        elif attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
    
    def _get_logits_warper(
        self,
        temperature: float,
        top_k: Optional[int],
        top_p: float,
    ) -> LogitsProcessorList:
        """
        Create logits warper list for sampling.
        """
        warpers = LogitsProcessorList()
        
        # Add temperature warping
        if temperature is not None and temperature != 1.0:
            warpers.append(TemperatureLogitsWarper(temperature))
        
        # Add top-k filtering
        if top_k is not None and top_k > 0:
            warpers.append(TopKLogitsWarper(top_k=top_k, min_tokens_to_keep=1))
        
        # Add top-p filtering
        if top_p is not None and top_p < 1.0:
            warpers.append(TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=1))
        
        return warpers
    
    def get_stop_codon_token_ids(self, seq_type: str = "dna") -> List[int]:
        """
        Get token IDs for stop codons.
        
        Stop codons signal the end of protein translation:
        - DNA: TAA, TAG, TGA
        - RNA: UAA, UAG, UGA
        
        Example:
            >>> inference = DecodonInference(model_path, task_type)
            >>> inference.configure_model()
            >>> stop_ids = inference.get_stop_codon_token_ids()
            >>> generated = inference.generate(input_ids, eos_token_id=stop_ids)
        """
        if seq_type.lower() == "dna":
            stop_codons = ["TAA", "TAG", "TGA"]
        elif seq_type.lower() == "rna":
            stop_codons = ["UAA", "UAG", "UGA"]
        else:
            raise ValueError(f"seq_type must be 'dna' or 'rna', got: {seq_type}")
        
        return [self.tokenizer.encoder[codon] for codon in stop_codons]
    
    def get_organism_token_id(self, organism_id: Union[str, int]) -> int:
        """
        Get token ID for a specific organism.
        
        Organism tokens are formatted as "<taxid>" where taxid is the NCBI taxonomy ID.
            
        Example:
            >>> inference = DecodonInference(model_path, task_type, organism_tokens_file)
            >>> inference.configure_model()
            >>> human_token = inference.get_organism_token_id("9606")  # Homo sapiens
            >>> marmatoe_token = inference.get_organism_token_id("1499973")     # Escherichia marmotae
        """
        organism_key = f"<{organism_id}>"
        if organism_key not in self.tokenizer.encoder:
            raise KeyError(
                f"Organism ID '{organism_id}' not found in tokenizer vocabulary. "
                f"Make sure organism_tokens_file was provided and contains this organism."
            )
        return self.tokenizer.encoder[organism_key]
    
    def list_available_organisms(self) -> Dict[str, int]:
        """
        List all available organism tokens and their IDs.
        
        Returns:
            Dictionary mapping organism taxonomy IDs to token IDs
            
        Example:
            >>> organisms = inference.list_available_organisms()
            >>> print(organisms)
            {'9606': 852, '562': 123, ...}
        """
        organisms = {}
        for token_str, token_id in self.tokenizer.encoder.items():
            # Organism tokens are formatted as "<taxid>"
            if token_str.startswith("<") and token_str.endswith(">"):
                inner = token_str[1:-1]
                # Check if inner is a numeric ID (taxonomy ID)
                if inner.isdigit():
                    organisms[inner] = token_id
        return organisms
    
    def get_start_codon_token_id(self, seq_type: str = "dna") -> int:
        """
        Get token ID for the standard start codon (ATG/AUG).

        Example:
            >>> start_id = inference.get_start_codon_token_id()
            >>> # Use as first codon after organism token for generation
        """
        if seq_type.lower() == "dna":
            start_codon = "ATG"
        elif seq_type.lower() == "rna":
            start_codon = "AUG"
        else:
            raise ValueError(f"seq_type must be 'dna' or 'rna', got: {seq_type}")
        
        return self.tokenizer.encoder[start_codon]
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = None,
        max_length: Optional[int] = 2048,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 1.0,
        do_sample: bool = True,
        num_beams: int = 1,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        early_stopping: bool = False,
        num_return_sequences: int = 1,
        length_penalty: float = 1.0,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate sequences using the DeCodon model.
        
        This method follows the HuggingFace generate pattern and supports:
        - Greedy decoding (do_sample=False, num_beams=1)
        - Multinomial sampling (do_sample=True, num_beams=1)
        - Beam search (num_beams>1, do_sample=False)
        
        Generation stops when EITHER condition is met:
        1. Maximum number of new tokens is reached (max_new_tokens)
        2. A stop token/codon is generated (eos_token_id)
        
        Args:
            input_ids: Input token IDs of shape (batch_size, sequence_length)
            attention_mask: Attention mask of shape (batch_size, sequence_length)
            max_new_tokens: Maximum number of new tokens to generate
            max_length: Maximum total length of generated sequences (alternative to max_new_tokens)
            temperature: Temperature for sampling (higher = more random). Default: 1.0
            top_k: Keep only top k tokens during sampling. Default: 50
            top_p: Keep tokens with cumulative probability <= top_p. Default: 1.0
            do_sample: Whether to sample or use greedy decoding. Default: True
            num_beams: Number of beams for beam search. Default: 1
            pad_token_id: ID of padding token. Defaults to model's pad_token_id.
            eos_token_id: ID(s) of end-of-sequence/stop codon token(s). 
                         If None, generation only stops at max_length.
                         For stop codons, use [53, 55, 61] (TAA, TAG, TGA for DNA).
            early_stopping: Whether to stop when EOS is generated (used in beam search). Default: False
            num_return_sequences: Number of sequences to return per input. Default: 1
                                  For sampling, inputs will be duplicated internally.
            length_penalty: Exponential penalty to length for beam search. Default: 1.0
            **kwargs: Additional keyword arguments
            
        Returns:
            Generated token IDs of shape (batch_size*num_return_sequences, sequence_length + generated_length)

        Note: generated_length may be less than max_new_tokens if stop tokens are encountered
        """
        # Validate input
        if input_ids.numel() == 0:
            raise ValueError("input_ids cannot be empty")
        if input_ids.shape[1] == 0:
            raise ValueError("input_ids must have at least one token (sequence length > 0)")
        
        # Set pad_token_id from config if not provided
        if pad_token_id is None:
            pad_token_id = self.model.hparams.pad_token_id
        
        # Determine max_length for stopping criteria
        if max_new_tokens is None and max_length is not None:
            max_new_tokens = max_length - input_ids.shape[1]
        elif max_new_tokens is None:
            max_new_tokens = 2046  # Default fallback
        
        max_new_tokens = max(1, max_new_tokens)  # Ensure at least 1 token
        final_max_length = input_ids.shape[1] + max_new_tokens
        

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        batch_size = input_ids.shape[0]
        
        # Set model to eval mode
        self.model.model.eval()
        
        # Determine generation mode
        is_greedy_gen_mode = (num_beams == 1) and (do_sample is False)
        is_sample_gen_mode = (num_beams == 1) and (do_sample is True)
        is_beam_gen_mode = (num_beams > 1) and (do_sample is False)
        
        # Prepare logits processors and stopping criteria
        logits_processor = LogitsProcessorList()
        
        stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=final_max_length)])
        
        # Prepare model kwargs
        model_kwargs = {
            "attention_mask": attention_mask,
        }
        
        # Run generation based on mode
        if is_greedy_gen_mode:
            if num_return_sequences > 1:
                raise ValueError(
                    f"num_return_sequences has to be 1, but is {num_return_sequences} when doing greedy search."
                )
            return self.greedy_search(
                input_ids,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                **model_kwargs,
            )
        elif is_sample_gen_mode:
            logits_warper = self._get_logits_warper(
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            
            # Expand inputs for multiple return sequences in sampling mode
            if num_return_sequences > 1:
                input_ids, model_kwargs = self._expand_inputs_for_generation(
                    input_ids=input_ids,
                    expand_size=num_return_sequences,
                    attention_mask=attention_mask,
                )
            
            return self.sample(
                input_ids,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                **model_kwargs,
            )
        elif is_beam_gen_mode:
            if num_return_sequences > num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")
            
            # Initialize beam scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=num_beams,
                device=input_ids.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_return_sequences,
            )
            
            # Expand input_ids and model_kwargs for beam search
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=num_beams,
                attention_mask=attention_mask,
            )
            
            return self.beam_search(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                **model_kwargs,
            )
        else:
            raise ValueError(f"Unsupported generation mode: num_beams={num_beams}, do_sample={do_sample}")
    
    def greedy_search(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        **model_kwargs,
    ) -> torch.LongTensor:
        """
        Generate sequences using greedy decoding.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, sequence_length)
            logits_processor: Optional logits processors to apply
            stopping_criteria: Optional stopping criteria
            pad_token_id: Padding token ID
            eos_token_id: End-of-sequence token ID(s)
            **model_kwargs: Additional model inputs (e.g., attention_mask)
            
        Returns:
            Generated token IDs of shape (batch_size, final_sequence_length)
        """
        # Initialize processors
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        
        # Convert eos_token_id to set for efficient lookup
        if isinstance(eos_token_id, list):
            eos_token_id_tensor = torch.tensor(eos_token_id, dtype=torch.long, device=input_ids.device)
        elif eos_token_id is not None:
            eos_token_id_tensor = torch.tensor([eos_token_id], dtype=torch.long, device=input_ids.device)
        else:
            eos_token_id_tensor = None
        
        # Track which sequences are finished
        batch_size = input_ids.shape[0]
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        
        # Main generation loop
        while True:
            # Track where the real sequence ends (before padding)
            real_seq_length = input_ids.shape[1]
            
            # Prepare model inputs (may add padding for xformers)
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            
            outputs = self.model.model(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs.get("attention_mask"),
                return_hidden_states=False,
            )
            
            # Get next token logits from the last real token position (not padding)
            next_token_logits = outputs.logits[:, real_seq_length - 1, :]
            
            # Restrict to codon vocabulary + special tokens
            next_token_logits = next_token_logits[:, :CODON_VOCAB_SIZE]
            
            # Apply logits processors
            next_token_scores = logits_processor(input_ids, next_token_logits)
            
            # Greedy selection
            next_tokens = torch.argmax(next_token_scores, dim=-1)
            
            # Update tokens for finished sequences (use pad token)
            if eos_token_id_tensor is not None and pad_token_id is not None:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            
            # Append next tokens to input_ids
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            
            # Update attention mask
            if "attention_mask" in model_kwargs:
                model_kwargs["attention_mask"] = torch.cat(
                    [model_kwargs["attention_mask"], 
                     unfinished_sequences[:, None]], 
                    dim=-1
                )
            
            # Check for EOS tokens
            if eos_token_id_tensor is not None:
                # Check if next_tokens is in eos_token_id list
                is_eos = torch.isin(next_tokens, eos_token_id_tensor)
                unfinished_sequences = unfinished_sequences.mul((~is_eos).long())
            
            # Stop if all sequences are finished or stopping criteria met
            stop_criteria_result = stopping_criteria(input_ids, None)

            if isinstance(stop_criteria_result, torch.Tensor):
                should_stop = bool(stop_criteria_result.all())
            else:
                should_stop = bool(stop_criteria_result)
            
            if unfinished_sequences.max().item() == 0 or should_stop:
                break
        
        return input_ids
    
    def sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        **model_kwargs,
    ) -> torch.LongTensor:
        """
        Generate sequences using multinomial sampling.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, sequence_length)
            logits_processor: Optional logits processors to apply
            logits_warper: Optional logits warpers for sampling
            stopping_criteria: Optional stopping criteria
            pad_token_id: Padding token ID
            eos_token_id: End-of-sequence token ID(s)
            **model_kwargs: Additional model inputs (e.g., attention_mask)
            
        Returns:
            Generated token IDs of shape (batch_size, final_sequence_length)
        """
        # Initialize processors
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        
        # Convert eos_token_id to tensor for efficient lookup
        if isinstance(eos_token_id, list):
            eos_token_id_tensor = torch.tensor(eos_token_id, dtype=torch.long, device=input_ids.device)
        elif eos_token_id is not None:
            eos_token_id_tensor = torch.tensor([eos_token_id], dtype=torch.long, device=input_ids.device)
        else:
            eos_token_id_tensor = None
        
        # Track which sequences are finished
        batch_size = input_ids.shape[0]
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        
        # Main generation loop
        while True:
            # Track real sequence length before padding
            real_seq_length = input_ids.shape[1]
            
            # Prepare model inputs (may add padding)
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            
            # Forward pass
            outputs = self.model.model(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs.get("attention_mask"),
                return_hidden_states=False,
            )
            
            # Get next token logits from the last real token position (not padding)
            next_token_logits = outputs.logits[:, real_seq_length - 1, :]
            
            # Restrict to codon vocabulary + special tokens
            # Organism tokens should not be predicted
            next_token_logits = next_token_logits[:, :CODON_VOCAB_SIZE]
            
            # Apply logits processors (e.g., repetition penalty)
            next_token_scores = logits_processor(input_ids, next_token_logits)
            
            # Apply logits warpers (temperature, top-k, top-p)
            next_token_scores = logits_warper(input_ids, next_token_scores)
            
            # Sample from the distribution
            probs = F.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            
            # Update tokens for finished sequences (use pad token)
            if eos_token_id_tensor is not None and pad_token_id is not None:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            
            # Append next tokens to input_ids
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            
            # Update attention mask
            if "attention_mask" in model_kwargs:
                model_kwargs["attention_mask"] = torch.cat(
                    [model_kwargs["attention_mask"], 
                     unfinished_sequences[:, None]], 
                    dim=-1
                )
            
            # Check for EOS tokens
            if eos_token_id_tensor is not None:
                # Check if next_tokens is in eos_token_id list
                is_eos = torch.isin(next_tokens, eos_token_id_tensor)
                unfinished_sequences = unfinished_sequences.mul((~is_eos).long())
            
            # Stop if all sequences are finished or stopping criteria met
            stop_criteria_result = stopping_criteria(input_ids, None)
            # Handle both scalar and tensor results from stopping_criteria
            if isinstance(stop_criteria_result, torch.Tensor):
                should_stop = bool(stop_criteria_result.all())
            else:
                should_stop = bool(stop_criteria_result)
            
            if unfinished_sequences.max().item() == 0 or should_stop:
                break
        
        return input_ids
    
    @staticmethod
    def _expand_inputs_for_generation(
        input_ids: torch.LongTensor,
        expand_size: int = 1,
        attention_mask: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """
        Expand inputs for generation (e.g., for beam search).
        
        Args:
            input_ids: Input token IDs of shape (batch_size, sequence_length)
            expand_size: Number of times to expand each batch element
            attention_mask: Optional attention mask
            **model_kwargs: Additional model kwargs
            
        Returns:
            Tuple of (expanded_input_ids, expanded_model_kwargs)
        """
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)
        
        model_kwargs_out = {}
        if attention_mask is not None:
            model_kwargs_out["attention_mask"] = attention_mask.index_select(0, expanded_return_idx)
        
        return input_ids, model_kwargs_out
    
    def beam_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamSearchScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **model_kwargs,
    ) -> torch.LongTensor:
        """
        Generate sequences using beam search decoding.
        
        Args:
            input_ids: Input token IDs of shape (batch_size * num_beams, sequence_length)
            beam_scorer: BeamSearchScorer instance for managing beam hypotheses
            logits_processor: Optional logits processors to apply
            stopping_criteria: Optional stopping criteria
            pad_token_id: Padding token ID
            eos_token_id: End-of-sequence token ID
            **model_kwargs: Additional model inputs (e.g., attention_mask)
            
        Returns:
            Generated token IDs of shape (batch_size * num_return_sequences, final_sequence_length)
        """
        # Initialize processors
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        
        if len(stopping_criteria) == 0:
            import warnings
            warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever")
        
        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams
        
        batch_beam_size, cur_len = input_ids.shape
        
        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )
        
        # Initialize beam scores
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))
        
        # Main generation loop
        while True:
            # Track real sequence length before padding
            real_seq_length = input_ids.shape[1]
            
            # Prepare model inputs (may add padding)
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = self.model.model(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs.get("attention_mask"),
                return_hidden_states=False,
            )
            
            # Get next token logits from the last real token position (not padding)
            next_token_logits = outputs.logits[:, real_seq_length - 1, :]
            
            # Restrict to codon vocabulary + special tokens
            next_token_logits = next_token_logits[:, :CODON_VOCAB_SIZE]
            
            # Convert logits to log probabilities
            next_token_scores = F.log_softmax(next_token_logits, dim=-1)
            
            # Apply logits processors
            next_token_scores_processed = logits_processor(input_ids, next_token_scores)
            
            # Add beam scores
            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores)
            
            # Reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)
            
            # Get top 2*num_beams tokens
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )
            
            # Calculate which beam each token came from
            next_indices = torch.div(next_tokens, vocab_size, rounding_mode='floor')
            next_tokens = next_tokens % vocab_size
            
            # Process beam hypotheses
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]
            
            # Update input_ids for next iteration
            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            
            # Update attention mask
            if "attention_mask" in model_kwargs:
                model_kwargs["attention_mask"] = torch.cat(
                    [model_kwargs["attention_mask"][beam_idx, :],
                     model_kwargs["attention_mask"].new_ones((batch_size * num_beams, 1))],
                    dim=-1
                )
            
            cur_len = cur_len + 1
            
            # Check stopping criteria
            stop_criteria_result = stopping_criteria(input_ids, None)
            # Handle both scalar and tensor results from stopping_criteria
            if isinstance(stop_criteria_result, torch.Tensor):
                should_stop = bool(stop_criteria_result.all())
            else:
                should_stop = bool(stop_criteria_result)
            
            if beam_scorer.is_done or should_stop:
                break
        
        # Finalize beam search
        # Extract max_length from stopping criteria (MaxLengthCriteria stores it)
        finalize_max_length = cur_len
        for criterion in stopping_criteria:
            if hasattr(criterion, 'max_length'):
                finalize_max_length = criterion.max_length
                break
        
        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=finalize_max_length,
        )
        
        return sequence_outputs["sequences"]
    
    def _is_organism_token(self, token_id: int) -> bool:
        """Check if token_id is an organism token (format: <NUMBER>)."""
        if token_id is None:
            return False
        token = self.tokenizer.decoder.get(token_id, "")
        special_tokens = {'<CLS>', '<SEP>', '<UNK>', '<PAD>', '<MASK>'}
        if token.startswith('<') and token.endswith('>') and token not in special_tokens:
            inner = token[1:-1]
            return inner.isdigit() or inner in getattr(self.tokenizer, 'organism_tokens', {})
        return False
    
    def _normalize_organism_token(self, organism_token: Optional[Union[str, int]]) -> Optional[str]:
        """Convert organism token to standard <NUMBER> format."""
        if organism_token is None:
            return None
        if isinstance(organism_token, (int, np.integer)):
            return f"<{int(organism_token)}>"
        token_str = str(organism_token)
        if token_str.startswith("<") and token_str.endswith(">"):
            return token_str
        return f"<{token_str}>"
    
    def build_generation_batch(
        self,
        seqs: List[str],
        target_num_codons: int,
        context_length: int,
        organism_token: Optional[Union[str, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Build model inputs for autoregressive generation.
        
        Tokenizes partial sequences and prepends organism token if not present.
        
        Args:
            seqs: List of DNA sequences (partial sequences during generation).
            target_num_codons: Total number of codons in target sequence.
            context_length: Maximum context length for the model.
            organism_token: Optional organism token (e.g., "9606", 9606, or "<9606>").
            
        Returns:
            Dictionary containing input_ids and attention_mask tensors.
        """
        if not isinstance(seqs, list):
            raise TypeError(f"seqs must be a list, got {type(seqs).__name__}")
        if context_length <= 2:
            raise ValueError(f"context_length must be > 2, got {context_length}")
        
        normalized_org_token = self._normalize_organism_token(organism_token)
        org_token_id = None
        if normalized_org_token is not None:
            org_token_id = self.tokenizer.encoder.get(normalized_org_token)
            if org_token_id is None:
                available_organisms = [
                    token for token in self.tokenizer.encoder.keys() 
                    if token.startswith('<') and token.endswith('>') 
                    and token not in ['<CLS>', '<SEP>', '<UNK>', '<PAD>', '<MASK>']
                ]
                raise ValueError(
                    f"Organism token '{normalized_org_token}' not found in tokenizer. "
                    f"{len(available_organisms)} organism tokens are available."
                )
        
        batch_input_ids: List[np.ndarray] = []
        batch_attention_masks: List[np.ndarray] = []
        
        for seq in seqs:
            token_ids = self.tokenizer.encode(seq)
            
            if len(token_ids) > 0 and self._is_organism_token(token_ids[0]):
                org_token_from_seq = self.tokenizer.decoder.get(token_ids[0])
                if seq.startswith(org_token_from_seq):
                    seq = seq[len(org_token_from_seq):]
                org_token_to_use = org_token_from_seq
            else:
                org_token_to_use = normalized_org_token
            
            if org_token_to_use is None:
                raise ValueError(
                    "No organism token provided and sequence does not start with one. "
                    "Please provide organism_token argument."
                )
            
            processed = process_item_clm(
                seq=seq,
                context_length=context_length,
                tokenizer=self.tokenizer,
                organism_token=org_token_to_use
            )
            batch_input_ids.append(processed['input_ids'])
            batch_attention_masks.append(processed['attention_mask'])
        
        input_ids = torch.tensor(np.stack(batch_input_ids), dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(np.stack(batch_attention_masks), dtype=torch.long, device=self.device)
        
        return {
            MetadataFields.INPUT_IDS: input_ids,
            MetadataFields.ATTENTION_MASK: attention_mask,
        }
    
    def get_next_codon_logits(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Get logits for the next codon position for each sequence in the batch.
        
        For autoregressive models, returns logits at the last non-padding position.
        
        Args:
            batch: Dictionary containing input_ids and attention_mask tensors.
            
        Returns:
            Tensor of shape (batch_size, vocab_size) with next-token logits.
        """
        if MetadataFields.INPUT_IDS not in batch:
            raise ValueError(f"Batch missing required field: {MetadataFields.INPUT_IDS}")
        
        input_ids = batch[MetadataFields.INPUT_IDS]
        attention_mask = batch[MetadataFields.ATTENTION_MASK]
        
        if input_ids.size(0) == 0:
            return torch.zeros((0, self.tokenizer.vocab_size), dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            outputs = self.model(batch_data=batch)
            logits = outputs.logits
        
        last_positions = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(logits.size(0), device=logits.device)
        next_logits = logits[batch_indices, last_positions, :]
        
        return next_logits
    
    def predict_fitness(self, batch) -> FitnessPredictionOutput:
        """
        Compute sequence fitness as the average log probability of each token.
        
        For autoregressive models, sums the log probabilities of predicting
        each token given the previous tokens.
        
        Args:
            batch: Dictionary containing input_ids and attention_mask.
            
        Returns:
            FitnessPredictionOutput with fitness scores.
        """
        input_ids = batch[MetadataFields.INPUT_IDS]
        attention_mask = batch[MetadataFields.ATTENTION_MASK]
        
        with torch.no_grad():
            outputs = self.model(batch_data=batch)
            logits = outputs.logits
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            
            # Autoregressive shift: logits[i] predicts token[i+1]
            shifted_logits = log_probs[:, :-1, :]
            shifted_targets = input_ids[:, 1:]
            shifted_mask = attention_mask[:, 1:]
            
            target_log_probs = shifted_logits.gather(
                dim=-1, index=shifted_targets.unsqueeze(-1)
            ).squeeze(-1)
            
            target_log_probs = target_log_probs * shifted_mask.float()
            num_scored_tokens = shifted_mask.sum(dim=-1)
            fitness = target_log_probs.sum(dim=-1) / num_scored_tokens.clamp(min=1)
            
        return FitnessPredictionOutput(fitness=fitness.cpu().numpy())
    
    @torch.no_grad()
    def generate_with_aa_constraints(
        self,
        amino_acid_sequence: str,
        num_sequences: int = 1,
        organism_token: Optional[str] = None,
        temperature: float = 1.0,
        sample: bool = False,
        seed_codons: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Generate DNA sequences with synonymous codon constraints.
        
        This method generates sequences one codon at a time, enforcing that each
        generated codon is synonymous with the corresponding amino acid in the
        target sequence.
        
        Args:
            amino_acid_sequence: Target amino acid sequence (determines codon constraints).
            num_sequences: Number of sequences to generate (beam width).
            organism_token: Optional organism token to prepend (e.g., "Homo_sapiens").
            temperature: Sampling temperature (only used if sample=True).
            sample: If True, sample from distribution; if False, use beam search.
            seed_codons: Optional list of seed codons (None means predict).
            
        Returns:
            List of generated DNA sequences that translate to amino_acid_sequence.
        """
        num_codons = len(amino_acid_sequence)
        if seed_codons is None:
            seed_codons = [None] * num_codons
        
        initial_tokens = []
        if organism_token is not None:
            normalized = self._normalize_organism_token(organism_token)
            org_token_id = self.tokenizer.encoder.get(normalized)
            if org_token_id is None:
                raise ValueError(f"Organism token '{organism_token}' not found in tokenizer")
            initial_tokens.append(org_token_id)
        
        beam: List[Tuple[float, List[int], int]] = [(0.0, list(initial_tokens), 0)]
        
        for pos in range(num_codons):
            if seed_codons[pos] is not None:
                token_id = self.tokenizer.encoder.get(seed_codons[pos])
                if token_id is not None:
                    new_beam = []
                    for beam_id, (score, tokens, parent_id) in enumerate(beam):
                        new_beam.append((score, tokens + [token_id], beam_id))
                    beam = new_beam
                continue
            
            aa = amino_acid_sequence[pos]
            synonymous_codons = AA_TABLE.DNA.get(aa, [])
            if not synonymous_codons:
                raise ValueError(f"No synonymous codons found for amino acid '{aa}' at position {pos}")
            
            valid_token_ids = [
                self.tokenizer.encoder.get(c) 
                for c in synonymous_codons 
                if self.tokenizer.encoder.get(c) is not None
            ]
            
            batch_input_ids = []
            batch_attention_masks = []
            for _, tokens, _ in beam:
                batch_input_ids.append(list(tokens))
                batch_attention_masks.append([1] * len(tokens))
            
            if not batch_input_ids:
                raise ValueError("Beam is empty—cannot continue generation")
            
            # Pad to multiple of 8 for Flash Attention compatibility
            max_len = max(len(t) for t in batch_input_ids)
            padded_len = ((max_len + 7) // 8) * 8
            for i in range(len(batch_input_ids)):
                pad_len = padded_len - len(batch_input_ids[i])
                batch_input_ids[i] = batch_input_ids[i] + [self.tokenizer.pad_token_id] * pad_len
                batch_attention_masks[i] = batch_attention_masks[i] + [0] * pad_len
            
            input_ids = torch.tensor(batch_input_ids, dtype=torch.long, device=self.device)
            attention_mask = torch.tensor(batch_attention_masks, dtype=torch.long, device=self.device)
            
            batch = {
                MetadataFields.INPUT_IDS: input_ids,
                MetadataFields.ATTENTION_MASK: attention_mask,
            }
            outputs = self.model(batch_data=batch)
            logits = outputs.logits
            
            last_positions = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(logits.size(0), device=logits.device)
            next_logits = logits[batch_indices, last_positions, :]
            
            constraint_mask = torch.full((next_logits.size(-1),), float('-inf'), device=self.device)
            constraint_mask[valid_token_ids] = 0
            next_logits = next_logits + constraint_mask.unsqueeze(0)
            
            log_probs = torch.nn.functional.log_softmax(next_logits, dim=-1)
            
            if sample and temperature > 0:
                new_beam = []
                for i, (cum_score, tokens, _) in enumerate(beam):
                    probs = torch.softmax(next_logits[i] / temperature, dim=-1)
                    sampled_token = torch.multinomial(probs, 1).item()
                    token_log_prob = float(log_probs[i, sampled_token].item())
                    new_score = cum_score + token_log_prob
                    new_beam.append((new_score, tokens + [sampled_token], i))
                beam = new_beam
            else:
                candidates = []
                for i, (cum_score, tokens, _) in enumerate(beam):
                    for token_id in valid_token_ids:
                        token_log_prob = float(log_probs[i, token_id].item())
                        new_score = cum_score + token_log_prob
                        candidates.append((new_score, tokens + [token_id], i))
                candidates.sort(key=lambda x: x[0], reverse=True)
                top_candidates = candidates[:num_sequences]
                beam = [(score, tokens, idx) for idx, (score, tokens, _) in enumerate(top_candidates)]
        
        beam.sort(key=lambda x: x[0], reverse=True)
        dna_sequences = []
        for score, tokens, _ in beam[:num_sequences]:
            codons = []
            for token_id in tokens:
                token = self.tokenizer.decoder.get(token_id, "")
                if len(token) == 3 and all(c in 'ACGT' for c in token):
                    codons.append(token)
            dna_sequences.append("".join(codons))
        
        return dna_sequences