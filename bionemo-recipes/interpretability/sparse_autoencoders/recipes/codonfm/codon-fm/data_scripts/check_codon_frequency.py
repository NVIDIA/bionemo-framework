# %%
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path

import sys
sys.path.append('/workspace/codon_fm')
from src.tokenizer import Tokenizer

data_path = Path('/data/ncbi/processed_unfiltered')
tax_ids_to_remove = json.load(open('/data/ncbi/taxids_to_remove.json'))
metadata = json.load(open(data_path / 'metadata.json'))
tokenizer = Tokenizer()


groups = set([x['file_name'][:-4] for x in metadata['file_metadata']])
counts = {g:np.zeros(tokenizer.vocab_size) for g in groups}
for fm, cm in tqdm(zip(metadata['file_metadata'], metadata['chunks']), total=len(metadata['file_metadata'])):
    group = fm['file_name'][:-4]
    if group in tax_ids_to_remove:
        curr_taxids_to_remove = set(tax_ids_to_remove[group])
    else:
        curr_taxids_to_remove = set()
    mmap = np.memmap( data_path / cm['sequences']['path'],
                     dtype=cm['sequences']['dtype'],
                     mode='r',
                     shape=tuple(cm['sequences']['shape']))
    idx_mmap = np.memmap(data_path / cm['index']['path'],  
                        dtype=cm['index']['dtype'],
                                 mode='r',
                                 shape=tuple(cm['index']['shape']))
    for start, end, taxid in idx_mmap:
        if taxid in curr_taxids_to_remove:
            continue
        seq = mmap[start:end]
        idx, count = np.unique(seq, return_counts=True)
        counts[group][idx] += count

# %%
for g in counts:
    counts[g] = counts[g].tolist()
json.dump(counts, open('codon_counts_nopathogen.json', 'w'))
