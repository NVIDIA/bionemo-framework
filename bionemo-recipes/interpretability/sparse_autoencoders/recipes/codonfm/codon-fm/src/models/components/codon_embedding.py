from typing import Optional

import torch
import torch.nn as nn

class CodonEmbedding(nn.Module):
    """Codon Embedding layer.

    This module takes codon IDs as input and returns their embeddings.
    It includes an embedding layer, layer normalization, and dropout.
    """

    def __init__(self, config):
        """Initializes the CodonEmbedding module.

        Args:
            config: A configuration object with the following attributes:
                - vocab_size (int): The size of the vocabulary.
                - hidden_size (int): The size of the hidden embeddings.
                - pad_token_id (int): The ID of the padding token.
                - layer_norm_eps (float): The epsilon value for layer normalization.
                - hidden_dropout_prob (float): The dropout probability.
                - apply_post_emb_ln (bool): Whether to apply layer normalization after the embedding.
        """
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.apply_post_emb_ln = getattr(config, "apply_post_emb_ln", True)
        if self.apply_post_emb_ln:
            self.post_ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        

    def forward(
        self,
        input_ids: Optional[torch.LongTensor],
    ) -> torch.Tensor:
        """Forward pass for the CodonEmbedding module.

        Args:
            input_ids (Optional[torch.LongTensor]): A tensor of shape (batch_size, sequence_length)
                containing the input codon IDs.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, sequence_length, hidden_size)
                representing the embeddings of the input codons.
        """
        embeddings = self.word_embeddings(input_ids)
        if self.apply_post_emb_ln:
            embeddings = self.post_ln(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings