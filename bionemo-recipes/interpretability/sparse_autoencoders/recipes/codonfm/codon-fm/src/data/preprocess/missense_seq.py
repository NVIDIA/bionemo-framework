import numpy as np
from src.data.metadata import MetadataFields
from src.data.preprocess.mlm_memmap import process_item as process_item_mlm
from src.data.metadata import SynomCodons

def process_item(cds_sequence, center_variant, other_variants, tokenizer, N, context_length, use_weights=False,
                  mask_ref=True, mlm_probability=0.3, mask_replace_prob=0.8, random_replace_prob=0.1):
    tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(cds_sequence))
    synom_codons = SynomCodons()
    half_context = (context_length - 2) // 2
    center_pos = center_variant['codon_position']
    left = max(0,center_pos - half_context)
    right = min(len(tokens), center_pos + half_context)
    tokens = tokens[left:right]

    mlm_items = process_item_mlm(tokenizer, tokens.copy(), context_length, 
                                  mlm_probability, mask_replace_prob, random_replace_prob)
    mlm_masked_tokens = mlm_items[MetadataFields.INPUT_IDS]
    mlm_input_mask = mlm_items[MetadataFields.INPUT_MASK]

    
    tokens = np.insert(tokens, 0, tokenizer.cls_token_id)
    # - add SEP token
    tokens = np.append(tokens, tokenizer.sep_token_id)
    
    variants = [center_variant] + other_variants
    variants.sort(key=lambda v:v['codon_position'])

    mask_poses = np.array([v['codon_position'] for v in variants]) - left + 1
    assert np.all(mask_poses > 0) and np.all(mask_poses < len(tokens)-1), "Masked positions are out of bounds"
    masked_tokens = tokens.copy()
    if mask_ref:
        masked_tokens[mask_poses] = tokenizer.mask_token_id
    ref_codons = [v['ref_codon'] for v in variants]
    alt_codons = [v['alt_codon'] for v in variants]
    ref_codon_toks = tokenizer.convert_tokens_to_ids(ref_codons)
    alt_codon_toks = tokenizer.convert_tokens_to_ids(alt_codons)
    labels = np.array([v['label'] for v in variants])
    ref_synom_codon_toks = [tokenizer.convert_tokens_to_ids(synom_codons.get_synonymous_codons(ref_codon)) for ref_codon in ref_codons]
    alt_synom_codon_toks = [tokenizer.convert_tokens_to_ids(synom_codons.get_synonymous_codons(alt_codon)) for alt_codon in alt_codons]

    variant_weights = np.zeros(N, dtype=np.float32)
    for i in range(len(labels)):
        variant_weights[i] = variants[i]['weight'] if use_weights else 1.0

    ref_synom_mask = np.zeros((N, tokenizer.vocab_size), dtype=bool)
    alt_synom_mask = np.zeros((N, tokenizer.vocab_size), dtype=bool)
    for i in range(len(labels)):
        ref_synom_mask[i, ref_synom_codon_toks[i]] = True
        alt_synom_mask[i, alt_synom_codon_toks[i]] = True

    if len(labels) < N:
        labels = np.pad(labels, (0, N-len(labels)), mode='constant', constant_values=-100)
        ref_codon_toks = np.pad(ref_codon_toks, (0, N-len(ref_codon_toks)), mode='constant', constant_values=tokenizer.pad_token_id)
        alt_codon_toks = np.pad(alt_codon_toks, (0, N-len(alt_codon_toks)), mode='constant', constant_values=tokenizer.pad_token_id)
        mask_poses = np.pad(mask_poses, (0, N-len(mask_poses)), mode='constant', constant_values=-100)


    attention_mask = np.ones(context_length, dtype=bool)
    attention_mask[len(masked_tokens):] = 0
    to_pad = context_length - len(masked_tokens)
    if to_pad > 0:
        masked_tokens = np.pad(masked_tokens, (0, to_pad), mode='constant', constant_values=tokenizer.pad_token_id)
    assert len(masked_tokens) == context_length, "Masked tokens are not the same length as the context length"

    items = {
        MetadataFields.INPUT_IDS: np.asarray(masked_tokens, dtype=np.int64),
        MetadataFields.REF_CODON_TOKS: np.asarray(ref_codon_toks, dtype=np.int64),
        MetadataFields.ALT_CODON_TOKS: np.asarray(alt_codon_toks, dtype=np.int64),
        MetadataFields.ATTENTION_MASK: np.asarray(attention_mask, dtype=np.int64),
        MetadataFields.MUTATION_TOKEN_IDX: np.asarray(mask_poses, dtype=np.int64),
        MetadataFields.LABELS: np.asarray(labels, dtype=np.int64),
        MetadataFields.INPUT_MASK: mlm_input_mask.astype(bool),
        'ref_synom_mask': np.asarray(ref_synom_mask, dtype=bool),
        'alt_synom_mask': np.asarray(alt_synom_mask, dtype=bool),
        'variant_weights': np.asarray(variant_weights, dtype=np.float32),
    }
    return items