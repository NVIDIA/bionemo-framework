import numpy as np

from src.data.metadata import MetadataFields, SynomCodons

def _construct_sentence(ref_seq, codon_position, ref_codon, alt_codon, context_length, tokenizer, mask_mutation, use_alt):
    assert not (mask_mutation and use_alt), "Cannot mask mutation and use alt sequence at the same time"
    input_sequence_toks = tokenizer.tokenize(ref_seq)
    input_sequence_toks = tokenizer.convert_tokens_to_ids(input_sequence_toks)
    input_sequence_toks = np.asarray(input_sequence_toks)[:(context_length - 2)]
    
    # - add CLS token
    input_sequence_toks = np.insert(input_sequence_toks, 0, tokenizer.cls_token_id)
    
    # - add SEP token
    input_sequence_toks = np.append(input_sequence_toks, tokenizer.sep_token_id)
    
    ref_codon_toks = tokenizer.convert_tokens_to_ids(ref_codon)
    alt_codon_toks = tokenizer.convert_tokens_to_ids(alt_codon)

    mutation_token_idx = int(codon_position + 1)
    if (0 < mutation_token_idx < len(input_sequence_toks)-1):
        if mask_mutation:
            # - mask the mutation token
            input_sequence_toks[mutation_token_idx] = tokenizer.mask_token_id
        if use_alt:
            input_sequence_toks[mutation_token_idx] = alt_codon_toks
    else:
        raise ValueError(f"Mutation token index {mutation_token_idx} is out of bounds for input sequence of length {len(input_sequence_toks)}")
    
    attention_mask = np.ones(context_length, dtype=np.int64)
    attention_mask[len(input_sequence_toks):] = 0
    input_sequence_toks = np.pad(input_sequence_toks, (0, max(0, context_length - len(input_sequence_toks))), 
                                 mode='constant', constant_values=tokenizer.pad_token_id)
    input_sequence_toks = input_sequence_toks[:context_length]
    return input_sequence_toks, ref_codon_toks, alt_codon_toks, attention_mask, mutation_token_idx

def _construct_sentence_clm(ref_seq, codon_position, ref_codon, alt_codon, context_length, tokenizer, mask_mutation, use_alt, organism_token):
    assert not (mask_mutation and use_alt), "Cannot mask mutation and use alt sequence at the same time"
    assert organism_token is not None, "Organism token is required for CLM"
    organism_token = organism_token.strip("<>")
    organism_token = f"<{organism_token}>"
    assert organism_token in tokenizer.encoder, f"Organism token {organism_token} not found in tokenizer"
    
    input_sequence_toks = tokenizer.tokenize(ref_seq)
    input_sequence_toks = tokenizer.convert_tokens_to_ids(input_sequence_toks)
    input_sequence_toks = np.asarray(input_sequence_toks)[:context_length]
    add_sep = False
    if input_sequence_toks[-1] in [tokenizer.encoder["TAG"], tokenizer.encoder["TAA"], tokenizer.encoder["TGA"]]:
        if len(input_sequence_toks) < context_length:
            add_sep = True
        else:
            pass
    
    # - add organism token
    input_sequence_toks = np.insert(input_sequence_toks, 0, tokenizer.encoder[organism_token])
    if add_sep and len(input_sequence_toks) < context_length:
        input_sequence_toks = np.append(input_sequence_toks, tokenizer.sep_token_id)
    ref_codon_toks = tokenizer.convert_tokens_to_ids(ref_codon)
    alt_codon_toks = tokenizer.convert_tokens_to_ids(alt_codon)

    mutation_token_idx = codon_position + 1
    
    attention_mask = np.ones(context_length + 1, dtype=np.int64)
    attention_mask[len(input_sequence_toks):] = 0
    input_sequence_toks = np.pad(input_sequence_toks, (0, max(0, (context_length + 1) - len(input_sequence_toks))), 
                                 mode='constant', constant_values=tokenizer.pad_token_id)
    return input_sequence_toks, ref_codon_toks, alt_codon_toks, attention_mask, mutation_token_idx

def mlm_process_item(ref_seq, codon_position, ref_codon, alt_codon, context_length, tokenizer, mask_mutation=True):    
    input_sequence_toks, ref_codon_toks, alt_codon_toks, attention_mask, mutation_token_idx = _construct_sentence(ref_seq, 
                                                                                                                  codon_position, 
                                                                                                                  ref_codon, 
                                                                                                                  alt_codon, 
                                                                                                                  context_length, 
                                                                                                                  tokenizer, 
                                                                                                                  mask_mutation=mask_mutation, 
                                                                                                                  use_alt=False)
    
    return {
        MetadataFields.INPUT_IDS: np.asarray(input_sequence_toks, dtype=np.int64),
        MetadataFields.REF_CODON_TOKS: np.asarray(ref_codon_toks, dtype=np.int64),
        MetadataFields.ALT_CODON_TOKS: np.asarray(alt_codon_toks, dtype=np.int64),
        MetadataFields.ATTENTION_MASK: np.asarray(attention_mask, dtype=np.int64),
        MetadataFields.MUTATION_TOKEN_IDX: np.asarray([mutation_token_idx], dtype=np.int64),
    }

def clm_process_item(ref_seq, codon_position, ref_codon, alt_codon, context_length, tokenizer, organism_token, **kwargs):
    input_sequence_toks, ref_codon_toks, alt_codon_toks, attention_mask, mutation_token_idx = _construct_sentence_clm(ref_seq, 
                                                                                                                  codon_position, 
                                                                                                                  ref_codon, 
                                                                                                                  alt_codon, 
                                                                                                                  context_length, 
                                                                                                                  tokenizer, 
                                                                                                                  mask_mutation=False, 
                                                                                                                  use_alt=False, 
                                                                                                                  organism_token=organism_token)
    return {
        MetadataFields.INPUT_IDS: np.asarray(input_sequence_toks, dtype=np.int64),
        MetadataFields.REF_CODON_TOKS: np.asarray(ref_codon_toks, dtype=np.int64),
        MetadataFields.ALT_CODON_TOKS: np.asarray(alt_codon_toks, dtype=np.int64),
        MetadataFields.ATTENTION_MASK: np.asarray(attention_mask, dtype=np.int64),
        MetadataFields.MUTATION_TOKEN_IDX: np.asarray([mutation_token_idx], dtype=np.int64),
    }


def likelihood_process_item(ref_seq, codon_position, ref_codon, alt_codon, context_length, tokenizer):    
    input_sequence_toks, _, _, attention_mask, _ = _construct_sentence(ref_seq, 
                                                                       codon_position, 
                                                                       ref_codon, 
                                                                       alt_codon, 
                                                                       context_length, 
                                                                       tokenizer, 
                                                                       mask_mutation=False, 
                                                                       use_alt=True)
    
    return {
        MetadataFields.INPUT_IDS: np.asarray(input_sequence_toks, dtype=np.int64),
        MetadataFields.ATTENTION_MASK: np.asarray(attention_mask, dtype=np.int64),
    }

def missense_inference_process_item(ref_seq, codon_position, ref_codon, alt_codon, context_length, tokenizer):    
    input_sequence_toks, ref_codon_toks, alt_codon_toks, attention_mask, mutation_token_idx = _construct_sentence(ref_seq, 
                                                                                                                  codon_position, 
                                                                                                                  ref_codon, 
                                                                                                                  alt_codon, 
                                                                                                                  context_length, 
                                                                                                                  tokenizer, 
                                                                                                                  mask_mutation=True, 
                                                                                                                  use_alt=False)
    
    synoms = SynomCodons()
    ref_synom_codon_toks = tokenizer.convert_tokens_to_ids(synoms.get_synonymous_codons(ref_codon))
    alt_synom_codon_toks = tokenizer.convert_tokens_to_ids(synoms.get_synonymous_codons(alt_codon))
    ref_synom_mask = np.zeros((tokenizer.vocab_size), dtype=bool)
    alt_synom_mask = np.zeros((tokenizer.vocab_size), dtype=bool)
    ref_synom_mask[ref_synom_codon_toks] = True
    alt_synom_mask[alt_synom_codon_toks] = True 
    return {
        MetadataFields.INPUT_IDS: np.asarray(input_sequence_toks, dtype=np.int64),
        MetadataFields.REF_CODON_TOKS: np.asarray(ref_codon_toks, dtype=np.int64),
        MetadataFields.ALT_CODON_TOKS: np.asarray(alt_codon_toks, dtype=np.int64),
        MetadataFields.ATTENTION_MASK: np.asarray(attention_mask, dtype=np.int64),
        MetadataFields.MUTATION_TOKEN_IDX: np.asarray([mutation_token_idx], dtype=np.int64),
        MetadataFields.REF_SYNOM_MASK: ref_synom_mask,
        MetadataFields.ALT_SYNOM_MASK: alt_synom_mask,
    }