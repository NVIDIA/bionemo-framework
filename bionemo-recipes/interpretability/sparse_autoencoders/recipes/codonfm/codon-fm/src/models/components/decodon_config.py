from typing import Optional

class DeCodonConfig:
    """Configuration class for DeCodon model.
    
    This class handles the configuration for the DeCodon model, including model architecture
    parameters and training settings. Simplified to match codon-fm pipeline requirements.
    """
    
    model_type = "decodon"

    def __init__(
        self,
        vocab_size: int = 69,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 3,
        classifier_dropout: Optional[float] = None,
        gamma_init: float = 0.1,
        rotary_theta: float = 1e4,
        lm_type: str = "gpt",
        use_rotary_emb: bool = True,
        apply_post_emb_ln: bool = False,  # GPT-style: no LayerNorm after embeddings
        max_position_embeddings: int = 2048,
        **kwargs,
    ):
        """Initialize DeCodonConfig.
        
        Args:
            vocab_size: Size of the vocabulary.
            hidden_size: Size of the hidden layers.
            num_hidden_layers: Number of hidden layers.
            num_attention_heads: Number of attention heads.
            intermediate_size: Size of the intermediate layer in the transformer.
            hidden_act: Activation function for the hidden layer.
            hidden_dropout_prob: Dropout probability for hidden layers.
            attention_probs_dropout_prob: Dropout probability for attention layers.
            initializer_range: Range for weight initialization.
            layer_norm_eps: Epsilon for layer normalization.
            pad_token_id: ID of the padding token.
            classifier_dropout: Dropout probability for classifier.
            gamma_init: Initial value for gamma.
            rotary_theta: Theta value for rotary embeddings.
            lm_type: Type of language model head ("gpt" or "bert").
            use_rotary_emb: Whether to use rotary embeddings.
            apply_post_emb_ln: Whether to apply LayerNorm after embeddings (False for GPT-style).
            
        """
        # Validate parameters
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_attention_heads ({num_attention_heads})"
            )
            
        if hidden_act not in ["gelu", "relu", "silu", "gelu_new"]:
            raise ValueError(f"hidden_act must be one of: gelu, relu, silu, gelu_new, got {hidden_act}")

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.classifier_dropout = classifier_dropout
        self.gamma_init = gamma_init
        self.rotary_theta = rotary_theta
        self.lm_type = lm_type
        self.use_rotary_emb = use_rotary_emb
        self.apply_post_emb_ln = apply_post_emb_ln
        self.max_position_embeddings = max_position_embeddings
        # Store any additional kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
        
    def to_dict(self):
        return self.__dict__.copy()