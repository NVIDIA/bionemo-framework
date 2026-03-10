from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .codon_embedding import CodonEmbedding
from .encoder_layer import EncoderLayer
from .decodon_config import DeCodonConfig

@dataclass
class DeCodonOutput:
    """
    Base class for DeCodon model's outputs.
    """
    logits: torch.FloatTensor = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    all_hidden_states: Optional[torch.FloatTensor] = None

class DeCodon(nn.Module):
    """
    DeCodon is a transformer-based decoder model for codon sequences (GPT-style).

    It consists of a codon embedding layer, a stack of transformer decoder layers,
    and a configurable prediction head (updated to match cdsFM).
    """
    def __init__(self, config: DeCodonConfig):
        """
        Initializes the DeCodon model.

        Args:
            config: A configuration object containing model hyperparameters.
        """
        super().__init__()
        self.config = config
        self.embeddings = CodonEmbedding(config)
        self.layers = nn.ModuleList(
            [EncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.output_ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.cls = nn.Sequential(
            nn.Linear(config.hidden_size, 69),
        )
        self._init_weights()

    def reset_cls_parameters(self):
        """Resets the parameters of the classification head."""
        for module in self.cls.modules():
            if isinstance(module, nn.Linear):
                # We don't use the name-based scaling for the classification head
                gain = self.config.initializer_range * math.sqrt(math.log(2 * self.config.num_hidden_layers))
                nn.init.xavier_normal_(module.weight, gain=gain)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    def _get_extended_attention_mask(
        self,
        attention_mask: torch.Tensor,
        input_shape: tuple[int],
        device: torch.device,
        dtype: torch.float,
    ) -> torch.Tensor:
        """
        Creates a broadcastable attention mask from a 2D or 3D input mask.
        The resulting mask is suitable for use in self-attention mechanisms where
        it can be added to the attention scores.

        - If `attention_mask` is 2D (batch_size, seq_length), it's expanded to
          (batch_size, 1, 1, seq_length).
        - If `attention_mask` is 3D (batch_size, seq_length, seq_length), it's expanded to
          (batch_size, 1, seq_length, seq_length).

        The mask values are inverted (1s become 0s, 0s become a large negative number),
        so that masked positions have a large negative value and non-masked positions are 0.
        """
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )
        extended_attention_mask = extended_attention_mask.to(dtype=dtype, device=device)
        extended_attention_mask = torch.where(extended_attention_mask == 0, float('-inf'), 0)
        return extended_attention_mask

    def _init_weights(self):
        """
        Initializes the weights of the model using the MAGNETO initialization scheme.
        """
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                is_qk = 'query' in name or 'key' in name
                scale_factor = math.sqrt(math.log(2 * self.config.num_hidden_layers))
                scale_value = self.config.initializer_range * scale_factor
                gain = 1.0 if is_qk else scale_value
                nn.init.xavier_normal_(module.weight, gain=gain)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.padding_idx is not None:
                    module.weight.data[self.config.pad_token_id].zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
        **kwargs
    ) -> DeCodonOutput:
        """
        Performs the forward pass of the DeCodon model.

        Args:
            input_ids: Tensor of input token ids. Shape (batch_size, sequence_length).
            attention_mask: Optional mask (1 for real tokens, 0 for padding). Shape (batch_size, sequence_length).
            return_hidden_states: Whether to return all hidden states.
            debug: Whether to enable debug mode.

        Returns:
            A `DeCodonOutput` object containing the logits and the last hidden state.
        """
        hidden_states = self.embeddings(input_ids=input_ids)
        input_shape = hidden_states.size()[:-1]

        extended_attention_mask: torch.Tensor = self._get_extended_attention_mask(
            attention_mask, input_shape, device=input_ids.device, dtype=next(self.parameters()).dtype
        )
        all_hidden_states = []
        for layer_module in self.layers:
            layer_outputs = layer_module(
                hidden_states,
                extended_attention_mask,
                is_causal=True,
            )
            hidden_states = layer_outputs
            if return_hidden_states:
                all_hidden_states.append(hidden_states)
        
        sequence_output = hidden_states
        sequence_output = self.output_ln(sequence_output)
        prediction_scores = self.cls(sequence_output)
        
        return DeCodonOutput(
            logits=prediction_scores,
            last_hidden_state=hidden_states,
            all_hidden_states=all_hidden_states if return_hidden_states else None,
        )