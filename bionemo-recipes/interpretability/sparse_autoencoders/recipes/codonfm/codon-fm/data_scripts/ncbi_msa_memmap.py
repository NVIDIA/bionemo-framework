import pandas as pd
import os,sys
import numpy as np
from glob import glob
from tqdm import tqdm
import json
import re
import argparse

def save_checkpoint(value, filename="msa_mmap_checkpoint.txt"):
    with open(filename, "w") as file:
        file.write(str(value))

# Function to load the checkpoint
def load_checkpoint(filename="msa_mmap_checkpoint.txt"):
    if os.path.exists(filename):
        with open(filename, "r") as file:
            return int(file.read().strip())
    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Create MSA memmap files")
    parser.add_argument('--checkpoint', type=str, default='msa_mmap_checkpoint.txt', help='Checkpoint file to save the progress')
    parser.add_argument('--mmap_dir', type=str, default='/data/codonfm_msa/ncbi_pssm/mmap', help='Directory to save the memmap files')
    parser.add_argument('--chunk_size', type=int, default=100_000_000, help='Size of each chunk')
    parser.add_argument('--data_dir_pattern', type=str, default='/data/codonfm_msa/ncbi_pssm/temp_*', help='Directory containing the data files')
    return parser.parse_args()


if __name__=='__main__':
    args = parse_args()
    mmap_dir = args.mmap_dir
    os.makedirs(mmap_dir, exist_ok=True)
    # Use the compiled regex to search the input string

    compiled_pattern = re.compile(r'batch_meta_(\d+)\.json')
    chunk_size = args.chunk_size

    chunk_counter = 0
    checked_chunk_counter = load_checkpoint(args.checkpoint)
    all_dirs = sorted(glob(args.data_dir_pattern))
    for data_dir in tqdm(all_dirs):
        meta_files = sorted(glob(f'{data_dir}/batch_meta*.json'), key=lambda k: int(compiled_pattern.search(k).group(1)))
        concat_groups = []
        lengths = []
        curr_length = 0
        curr_group = []
        for meta_file in meta_files:
            with open(meta_file, 'r') as f:
                meta = json.load(f)
        
                length = sum([x[2] for x in meta])
                if curr_length + length > chunk_size:
                    concat_groups.append(curr_group)
                    lengths.append(curr_length)
                    curr_length = 0
                    curr_group = []
                curr_group.append(meta_file)
                curr_length += length
        if curr_group:
            concat_groups.append(curr_group)
            lengths.append(curr_length)
        
        for group, length in zip(concat_groups, lengths):
            if checked_chunk_counter and chunk_counter <= checked_chunk_counter:
                chunk_counter += 1
                continue
            group_meta = []
            offset = 0
            group_data = np.memmap(os.path.join(mmap_dir, f'chunk_{chunk_counter}.mmap'), dtype=np.uint16, mode='w+', shape=(length, 69))
            for mfile in group:
                with open(mfile,'r') as f:
                    meta = json.load(f)
                meta = pd.DataFrame(meta)
                meta.columns = ['org_group', 'org_index', 'seq_len']
                edges = np.cumsum([0] + meta['seq_len'].tolist()) + offset
                meta['start'] = edges[:-1]
                meta['end'] = edges[1:]
        
                group_meta.append(meta)
                temp = np.load(mfile+'.npy')
                assert  meta['seq_len'].sum() == temp.shape[0]
                group_data[offset:offset+temp.shape[0], :] = temp[:]
                group_data.flush()
                offset += temp.shape[0]
            group_meta = pd.concat(group_meta)
            group_meta.to_csv(os.path.join(mmap_dir, f'chunk_{chunk_counter}.csv'), sep=',',index=False)
            save_checkpoint(chunk_counter, args.checkpoint)
            chunk_counter += 1
                