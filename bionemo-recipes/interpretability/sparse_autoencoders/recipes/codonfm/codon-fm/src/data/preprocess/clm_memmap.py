from typing import List, Dict, Any, Optional

import numpy as np

from src.data.metadata import MetadataFields

def process_item(tokenizer: Any,
                sequence_tokens: np.ndarray,
                context_length: int,
                organism_token: Optional[str] = None,
                codon_weights = None,
                ignore_index: int = -100,
                **kwargs) -> Dict[str, np.ndarray]:
    """
    Prepare GPT-style input with organism token prepended and SEP token appended.

    Args:
        tokenizer: DeCodonTokenizer with `set_organism_tokens()` already called.
        sequence_tokens: np.ndarray of token IDs (e.g., [1, 2, 3])
        context_length: Total length for padding (including organism token and SEP token)
        organism_token: String like "9606" (optional)
        codon_weights: Codon weights (unused in CLM but included for compatibility)
        ignore_index: Value to ignore in loss

    Returns:
        Dict with input_ids, labels, attention_mask
    """
    # Check if organism token is provided and in tokenizer
    if organism_token is None:
        raise ValueError("Organism token must be provided for CLM preprocessing")
    
    # normalize organism token
    if isinstance(organism_token, (int, np.integer)):
        organism_token_key = f"<{int(organism_token)}>"
    else:
        token_str = str(organism_token)
        organism_token_key = token_str if (token_str.startswith("<") and token_str.endswith(">")) else f"<{token_str}>"
    
    if organism_token_key not in tokenizer.encoder:
        available_organisms = [token for token in tokenizer.encoder.keys() if token.startswith('<') and token.endswith('>') and token not in ['<CLS>', '<SEP>', '<UNK>', '<PAD>', '<MASK>']]
        num_available = len(available_organisms)
        raise ValueError(f"Organism token '{organism_token_key}' not found in tokenizer. {num_available} organism tokens are available in the tokenizer.")
    
    org_id = tokenizer.encoder[organism_token_key]
    sep_id = tokenizer.sep_token_id
    
    sequence_tokens = np.asarray(sequence_tokens).astype(np.int64)
    input_ids = sequence_tokens[:context_length]
    
    # Check for empty sequence
    if len(input_ids) == 0:
        raise ValueError("Empty sequence provided. CLM preprocessing requires at least one token for training.")
    
    # Add organism token to front and SEP token to end
    # The SEP token is added only if the sequence is not long enough to reach the context length and the sequence ends with a stop codon.
    # [ORG, 1, 2, 3, SEP]
    if len(input_ids) < context_length and input_ids[-1] in [tokenizer.encoder["TAG"], tokenizer.encoder["TAA"], tokenizer.encoder["TGA"]]:
        input_ids = np.append(input_ids, sep_id)
    input_ids = np.insert(input_ids, 0, org_id)
    
    labels = input_ids[1:]
    input_ids = input_ids[:-1]
    
    # Pad if needed
    pad_len = context_length - len(input_ids)
    if pad_len > 0:
        input_ids = np.pad(input_ids, (0, pad_len), constant_values=tokenizer.pad_token_id)
        labels = np.pad(labels, (0, pad_len), constant_values=ignore_index)

    attention_mask = (input_ids != tokenizer.pad_token_id).astype(int)
    return {
        MetadataFields.INPUT_IDS: input_ids.astype(np.int64),
        MetadataFields.LABELS: labels.astype(np.int64),
        MetadataFields.ATTENTION_MASK: attention_mask.astype(np.int64),
    }