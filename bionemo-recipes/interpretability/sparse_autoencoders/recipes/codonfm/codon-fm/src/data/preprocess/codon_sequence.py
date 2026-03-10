from typing import Optional
import numpy as np


def process_item(seq, context_length, tokenizer):
    """Process item for MLM (bidirectional) models like EnCodon.
    
    Adds CLS token at start and SEP token at end.
    """
    input_sequence_toks = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(seq))
    input_sequence_toks = np.asarray(input_sequence_toks)[:(context_length - 2)]
    
    # - add CLS token
    input_sequence_toks = np.insert(input_sequence_toks, 0, tokenizer.cls_token_id)
    
    # - add SEP token
    input_sequence_toks = np.append(input_sequence_toks, tokenizer.sep_token_id)
    
    attention_mask = np.ones(context_length, dtype=np.int64)
    attention_mask[len(input_sequence_toks):] = 0
    # Pad/truncate to context_length using numpy
    input_sequence_toks = np.pad(input_sequence_toks, (0, max(0, context_length - len(input_sequence_toks))), 
                                 mode='constant', constant_values=tokenizer.pad_token_id)
    # input_sequence_toks = input_sequence_toks[:context_length]
    return {
        'input_ids': np.asarray(input_sequence_toks, dtype=np.int64),
        'attention_mask': np.asarray(attention_mask, dtype=np.int64),
    }


def process_item_clm(seq: str, context_length: int, tokenizer, organism_token: str):
    """Process item for CLM (autoregressive) models like DeCodon.
    
    Prepends organism token and appends SEP token if sequence ends with stop codon.
    
    Args:
        seq: DNA sequence string (codons will be tokenized)
        context_length: Total length for padding
        tokenizer: Tokenizer with organism tokens loaded
        organism_token: Organism token string (e.g., "9606" for human)
        
    Returns:
        Dict with input_ids, attention_mask
    """
    # Normalize organism token format
    if isinstance(organism_token, (int, np.integer)):
        organism_token_key = f"<{int(organism_token)}>"
    else:
        token_str = str(organism_token)
        organism_token_key = token_str if (token_str.startswith("<") and token_str.endswith(">")) else f"<{token_str}>"
    
    if organism_token_key not in tokenizer.encoder:
        available_organisms = [
            token for token in tokenizer.encoder.keys() 
            if token.startswith('<') and token.endswith('>') 
            and token not in ['<CLS>', '<SEP>', '<UNK>', '<PAD>', '<MASK>']
        ]
        raise ValueError(
            f"Organism token '{organism_token_key}' not found in tokenizer. "
            f"{len(available_organisms)} organism tokens are available."
        )
    
    org_id = tokenizer.encoder[organism_token_key]
    sep_id = tokenizer.sep_token_id
    
    # Tokenize the sequence
    input_sequence_toks = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(seq))
    input_sequence_toks = np.asarray(input_sequence_toks, dtype=np.int64)
    
    # Truncate to leave room for organism token
    input_ids = input_sequence_toks[:(context_length - 1)]
    
    if len(input_ids) == 0:
        raise ValueError("Empty sequence provided. CLM preprocessing requires at least one token.")
    
    # Add SEP token if sequence ends with stop codon and there's room
    stop_codon_ids = [tokenizer.encoder.get("TAG"), tokenizer.encoder.get("TAA"), tokenizer.encoder.get("TGA")]
    if len(input_ids) < (context_length - 1) and input_ids[-1] in stop_codon_ids:
        input_ids = np.append(input_ids, sep_id)
    
    # Prepend organism token: [ORG, codon1, codon2, ..., SEP]
    input_ids = np.insert(input_ids, 0, org_id)
    
    # Pad if needed
    pad_len = context_length - len(input_ids)
    if pad_len > 0:
        input_ids = np.pad(input_ids, (0, pad_len), constant_values=tokenizer.pad_token_id)
    
    # Truncate if too long
    input_ids = input_ids[:context_length]
    
    attention_mask = (input_ids != tokenizer.pad_token_id).astype(np.int64)
    
    return {
        'input_ids': input_ids.astype(np.int64),
        'attention_mask': attention_mask.astype(np.int64),
    }