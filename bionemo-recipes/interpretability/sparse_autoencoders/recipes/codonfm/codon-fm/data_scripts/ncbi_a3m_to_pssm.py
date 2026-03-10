import re
import mmap
import numpy as np
from glob import glob
from concurrent.futures import ProcessPoolExecutor  # Changed to process pool
import polars as pl
import gc
import os
import sys
sys.path.append('/workspace/codon-fm')
from src.tokenizer import Tokenizer
import json
import numpy as np
import torch
from pathlib import Path
from typing import Callable, List

import numpy as np
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
import argparse


def get_curr_mat(groups, indices, seq_data):
    sep_id = tokenizer.encode([tokenizer.special_tokens_map['sep_token']])[0]
    curr_mat = np.zeros([len(seq_data[0]), len(tokenizer)], dtype='uint16')
    for g, i, seq in zip(groups, indices, seq_data):
        orig_encoded = dataset.get_by_group(g,i)
        # orig_encoded = getitem(dataset, g,i)
        orig_aa = tokenizer.convert_ids_to_aa(orig_encoded)
        s_nogap = seq.replace('-','').replace('X','').upper()
         
        try:
            idx = orig_aa.index(s_nogap)
        except:
            print(groups[0], indices[0], g,i,orig_aa,s_nogap, seq, sep='\n')
            raise
        mat_i = 0
        for c in seq:
            if c == '-':
                curr_mat[mat_i, sep_id] += 1
                mat_i += 1
            elif ord('A') <= ord(c) <= ord('Z'):
                # co = orig_seq[idx*3:(idx+1)*3]
                co = orig_encoded[idx]
                curr_mat[mat_i, co] += 1
                idx += 1
                mat_i += 1
            elif ord('a') <= ord(c) <= ord('z'):
                idx += 1
            else:
                assert False

    return curr_mat

# Precompile regex patterns outside main logic
PATTERN_GROUP = re.compile(rb'([A-Za-z_]+)_\d+$', re.M)
PATTERN_INDEX = re.compile(rb'_(\d+)$', re.M)


# Precompute group indices using numpy memmap
def load_checkpoint(save_path):
    try:
        with open(os.path.join(save_path, "checkpoint.json"), "r") as f:
            return json.load(f)["processed_batches"]
    except FileNotFoundError:
        return []

def save_checkpoint(save_path, processed_batches):
    with open(os.path.join(save_path, "checkpoint.json"), "w") as f:
        json.dump({"processed_batches": processed_batches}, f)

def process_cluster_batch(args):
    """Process batch of indices using byte-level operations"""
    
    batch_indices, batch_idx, save_path, a3m = args
    # with open(f"/data/codonfm_msa/ncbi/{spec}SeqClust_rep.msa.a3m", "rb") as f:
    with open(a3m, "rb") as f:
        main_mmap = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    batch_results = []
    meta = []
    # Use numpy array for batch processing
    indices = a3m_idx[batch_indices]
    byte_ranges = np.column_stack([indices[:, 1], indices[:, 2]])
    
    for start, end in byte_ranges:
        # Process bytes directly without full decode
        # with open(f"/data/codonfm_msa/ncbi/{spec}SeqClust_rep.msa.a3m", "rb") as f:
        #     main_mmap = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        chunk = main_mmap[start:end].strip(b'\x00').strip()
        records = chunk.split(b'>')[1:]  # Skip empty first element
        
        # Vectorized processing using numpy
        headers = np.array([rec.split(b'\n', 1)[0] for rec in records])
        seq_data = np.array([rec.split(b'\n', 1)[1].replace(b'\n', b'').decode() for rec in records])
        if len(headers) <= 1:
            continue
        # Regex extraction using vectorized operations
        groups = [PATTERN_GROUP.search(header).group(1).decode() for header in headers]
        indices = [int(PATTERN_INDEX.search(header).group(1)) for header in headers]
        
        # Fetch sequences in bulk
        mat = get_curr_mat(groups[1:], indices[1:], seq_data[1:])
        meta.append([groups[0], indices[0], mat.shape[0]])
        batch_results.append(mat)
    output_meta_file = os.path.join(save_path, f"batch_meta_{batch_idx}.json")  # Customize the filename format as needed
    with open(output_meta_file, "w") as f:
        json.dump(meta, f)  # Save results in JSON format

    output_mat_file = os.path.join(save_path,  f"batch_meta_{batch_idx}.json")
    output_mat = np.concatenate(batch_results, axis=0)
    np.save(output_mat_file, output_mat)
    main_mmap.close()

    del output_mat, batch_results, chunk, main_mmap
    gc.collect()
    return batch_idx


def optimized_main(save_path, spec, n_workers=8, batch_size=1000):
    """Optimized parallel processing with batch prefetching"""
    print(save_path)
    if not os.path.exists(save_path):
    
        os.makedirs(save_path, exist_ok=True)
    else:
        print(save_path,' exists')
        
    processed_batches = load_checkpoint(save_path)
    
    total = len(a3m_idx)
    indices = np.arange(total)
    batches = [
        (indices[i:i + batch_size], i // batch_size, save_path, spec)  # Include batch index
        for i in range(0, total, batch_size) if i//batch_size not in processed_batches
    ]
    
    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            with tqdm(total=len(batches), desc="Processing Batches") as pbar:
                for batch_idx in executor.map(process_cluster_batch, batches):
                    processed_batches.append(batch_idx)
                    save_checkpoint(save_path, processed_batches)
                    pbar.update(1)
        
    else:
        with tqdm(total=len(batches), desc="Processing Batches") as pbar:
            for args in batches:
                batch_idx = process_cluster_batch(args)
                processed_batches.append(batch_idx)
                save_checkpoint(save_path, processed_batches)
                pbar.update(1)
    return 


class CodonMmapDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_path: str,
                 cache_path: str,
                 tokenizer: Callable,
                 seed: int = 42):

        self.data_path = Path(data_path)
        self.metadata_path = self.data_path / "metadata.json"
        self.tokenizer = tokenizer
        self.cache_path = Path(cache_path)

        with open(self.metadata_path, 'r') as f:
            metadata = json.load(f)
            
        self.chunks_metadata = metadata['chunks']

        self.group_offset = {}
        curr = 0
        for item in metadata['file_metadata']:
            gr = item['file_name'].split('.csv')[0]
            if gr not in self.group_offset:
                self.group_offset[gr] = curr
            curr += item['end']-item['start']+1


        if self.cache_path.exists():
            print("Loading cached global indices...")
            self.global_indices = np.load(cache_path, allow_pickle=True)#.tolist()
        else:
            self.indices_mmaps = []
            for chunk in self.chunks_metadata:
                
                idx_mmap_path = self.data_path / chunk['index']['path']
    
                idx_mmap = np.memmap(idx_mmap_path,
                                     dtype=chunk['index']['dtype'],
                                     mode='r',
                                     shape=tuple(chunk['index']['shape']))
    
                self.indices_mmaps.append(idx_mmap)
            print("Computing global indices for subsequences...")
            self.global_indices = []
            for chunk_id, idx_mmap in enumerate(self.indices_mmaps):
                for seq_idx in tqdm(range(len(idx_mmap))):
                    seq_start, seq_end, taxid = idx_mmap[seq_idx]

                    self.global_indices.append((chunk_id, seq_start, seq_end))

            np.save(self.cache_path, np.array(self.global_indices, dtype=np.uint32))
            print(f"Cached global indices saved at {cache_path}")

    def get_by_group(self, group, group_idx):
        idx = self.group_offset[group] + group_idx
        # return idx
        return self.__getitem__(idx)
    
    def __len__(self):
        return len(self.global_indices)

    def __getitem__(self, idx):
        chunk_id, start_token_idx, end_token_idx = self.global_indices[idx]
        chunk = self.chunks_metadata[chunk_id]
        seq_mmap_path = self.data_path / chunk['sequences']['path']
        seq_mmap = np.memmap(seq_mmap_path,
                                 dtype=chunk['sequences']['dtype'],
                                 mode='r',
                                 shape=tuple(chunk['sequences']['shape']))
        sequence_tokens = seq_mmap[start_token_idx:end_token_idx]
        return sequence_tokens


def parse_args():
    """
    Parse command-line arguments for the script.
    """
    parser = argparse.ArgumentParser(description="Run optimized dataset processing.")
    parser.add_argument(
        '--a3m',
        required=True,
        type=str,
        help="The spec (e.g., 'vertebrate_other') to be processed."
    )
    parser.add_argument(
        '--out_pre',
        required=True,
        type=str,
        help="output prefix"
    )
    parser.add_argument(
        '--n_workers',
        type=int,
        default=94,
        help="Number of worker processes to use (default: 94)."
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1000,
        help="Batch size for processing (default: 1000)."
    )
    return parser.parse_args()

if __name__=='__main__':
    args = parse_args()

    # a3m_idx = np.loadtxt(f'/data/codonfm_msa/ncbi/{args.spec}SeqClust_rep.msa.a3m.index', delimiter='\t', dtype=np.int64)
    a3m_idx = np.loadtxt(args.a3m+'.index', delimiter='\t', dtype=np.uint64)
    if len(a3m_idx.shape) < 2:
        a3m_idx = a3m_idx.reshape([-1,3])
    a3m_idx[:,2] = a3m_idx[:,1] + a3m_idx[:,2]

    tokenizer = Tokenizer(
            cls_token="<CLS>",
            bos_token="<CLS>",
            sep_token="<SEP>",
            unk_token="<UNK>",
            pad_token="<PAD>",
            mask_token="<MASK>",
            padding_side="right",
            truncation="right",
            seq_type="dna",
        )
    dataset = CodonMmapDataset('/data/ncbi_processed/', 
                               cache_path='/data/ncbi_processed/global_index.cache.npy',
                              tokenizer=tokenizer)
    
    optimized_main(args.out_pre, args.a3m, args.n_workers, batch_size=args.batch_size)
