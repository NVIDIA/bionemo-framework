import numpy as np
import polars as pl
import re
import pathlib
import json
from typing import List, Callable, Tuple, Optional
import numpy.typing as npt
from tqdm import tqdm

def load_train_val_test_indices_by_group(keys: List | npt.NDArray | pl.DataFrame,
                                metadata_path: str | pathlib.Path,
                                train_val_test_ratio: List[float] = [0.9998, 0.0002, 0.00],
                                post_fix: str = '',
                                seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    metadata_path = pathlib.Path(metadata_path)

    train_idx_path = metadata_path.parent / f'train_idx{post_fix}.npy'
    val_idx_path   = metadata_path.parent / f'val_idx{post_fix}.npy'
    test_idx_path  = metadata_path.parent / f'test_idx{post_fix}.npy'

    if train_idx_path.exists() and val_idx_path.exists() and test_idx_path.exists():
        print("Loading existing train/val/test splits from disk...")
        train_idx = np.load(train_idx_path)
        val_idx   = np.load(val_idx_path)
        test_idx  = np.load(test_idx_path)
    else:
        if not isinstance(keys, pl.DataFrame):
            keys = pl.DataFrame(keys)
        shuffled = keys.unique().sample(fraction=1, shuffle=True, seed=seed)

        # Calculate the partition sizes
        total_groups = len(shuffled)
        partition_1_size = int(total_groups * train_val_test_ratio[0])
        partition_2_size = int(total_groups * train_val_test_ratio[1])
        # Assign partitions
        partitions = [1] * partition_1_size + [2] * partition_2_size + [3] * (total_groups - partition_1_size - partition_2_size)
        shuffled = shuffled.with_columns(pl.Series('partition', partitions))

        # Map the partitions back to the original dataframe
        keys = keys.join(shuffled, on=keys.columns, how='left', validate='m:1').with_row_index('row_index')
        train_idx = keys.filter(pl.col('partition') == 1).select('row_index').to_numpy().flatten()
        val_idx = keys.filter(pl.col('partition') == 2).select('row_index').to_numpy().flatten()
        test_idx = keys.filter(pl.col('partition') == 3).select('row_index').to_numpy().flatten()
        
        np.save(train_idx_path, train_idx)
        np.save(val_idx_path, val_idx)
        np.save(test_idx_path, test_idx)

    return train_idx, val_idx, test_idx

def get_group_bits(groups_to_use):
    all_groups = ['Primates', 'archaea', 'bacteria', 'fungi', 'invertebrate', 'plant', 'protozoa', 'vertebrate_mammalian', 'vertebrate_other', 'viral']
    group_bits = np.power(2, np.arange(len(all_groups)))
    group_to_bits = dict(zip(all_groups, group_bits))
    
    try:
        return sum([group_to_bits[g] for g in groups_to_use])
    except KeyError as e:
        available_groups = list(group_to_bits.keys())
        raise ValueError(f"Unknown group {e}. Available groups: {available_groups}")

def load_train_val_test_indices_proportional(metadata_path, global_keys, global_indices, 
                                post_fix: str = '',
                               train_val_test_ratio=[0.9998, 0.0002, 0.00], seed=42):

    train_idx_path = metadata_path.parent / f'train_idx{post_fix}.npy'
    val_idx_path   = metadata_path.parent / f'val_idx{post_fix}.npy'
    test_idx_path  = metadata_path.parent / f'test_idx{post_fix}.npy'

    if train_idx_path.exists() and val_idx_path.exists() and test_idx_path.exists():
        print("Loading existing train/val/test splits from disk...")
        train_indices = np.load(train_idx_path)
        val_indices   = np.load(val_idx_path)
        test_indices  = np.load(test_idx_path)
    else:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)['file_metadata']
        groups = sorted(list(set([metadata[i]['file_name'].split('.')[0] for i in range(len(metadata))])))
        count_mat = np.zeros((max(global_keys)+1, len(groups)))
        for idx, cluster_idx in zip(global_indices, global_keys):
            group = metadata[idx[0]]['file_name'].split('.')[0]
            count_mat[cluster_idx, groups.index(group)] += 1
        val_size = int(len(global_keys) * train_val_test_ratio[1])
        test_size = int(len(global_keys) * train_val_test_ratio[2])
        val_indices, val_clusters = np.array([]), []
        test_indices, test_clusters = np.array([]), []
        if val_size > 0:
            val_indices, val_clusters = sample_clusters_by_size(count_mat, global_keys, used_clusters=[], 
                                                                target_size=val_size, seed=seed)
        if test_size > 0:
            test_indices, test_clusters = sample_clusters_by_size(count_mat, global_keys, 
                                                                used_clusters=val_clusters, 
                                                                target_size=test_size,
                                                                seed=seed)
        train_indices = np.setdiff1d(np.arange(len(global_keys)), np.concatenate([val_indices, test_indices]))

        np.save(train_idx_path, train_indices)
        np.save(val_idx_path, val_indices)
        np.save(test_idx_path, test_indices)
    return train_indices, val_indices, test_indices

def sample_clusters_by_size(count_mat, global_keys, used_clusters: Optional[List] = None, target_size=30000, seed=42):
    """
    Sample clusters starting from organisms with fewest sequences, randomly selecting clusters
    to meet target proportions for each organism.
    
    Args:
        count_mat: matrix of cluster counts per organism
        global_keys: array of cluster IDs for each sequence
        used_clusters: list of cluster IDs that have already been selected
        target_size: desired number of sequences in sample
        seed: random seed for reproducibility
    
    Returns:
        tuple of (array of indices corresponding to selected sequences, selected clusters)
    """
    if used_clusters is None:
        used_clusters = []
        
    np.random.seed(seed)
    
    total_per_organism = count_mat.sum(axis=0)
    
    total_sum = total_per_organism.sum()
    if total_sum == 0:
        raise ValueError("No sequences found in count matrix")
    
    target_proportions = total_per_organism / total_sum
    
    org_indices = np.argsort(total_per_organism)
    
    selected_clusters = []
    current_counts = np.zeros_like(total_per_organism)
    current_total = 0
    
    # Process organisms from rarest to most common
    for org_idx in tqdm(org_indices):
        # Calculate target number of sequences for this organism
        target_count = max(1, int(target_size * target_proportions[org_idx]))
        
        # Find clusters containing this organism
        valid_clusters = np.where(count_mat[:, org_idx] > 0)[0]
        
        # Remove already selected clusters
        valid_clusters = list(set(valid_clusters) - set(selected_clusters) - set(used_clusters))
        
        # Randomly shuffle available clusters
        np.random.shuffle(valid_clusters)
        
        # Keep adding clusters until we reach target count
        for cluster_idx in valid_clusters:
            if current_counts[org_idx] >= target_count:
                break
                
            selected_clusters.append(cluster_idx)
            current_counts += count_mat[cluster_idx]
            current_total += count_mat[cluster_idx].sum()
    
    print(f"\nFinal selection:")
    print(f"Selected {len(selected_clusters)} clusters")
    print(f"Total sequences: {current_total:,}")
    print("\nDistribution comparison:")
    print("Target vs Actual proportions:")
    for i in range(len(target_proportions)):
        actual_prop = current_counts[i] / current_total if current_total > 0 else 0
        print(f"Organism {i}: {target_proportions[i]:.3f} vs {actual_prop:.3f}")
    
    # Check for organisms with no sequences
    zeros = np.where(current_counts == 0)[0]
    if len(zeros) > 0:
        print("\nWARNING: The following organisms have no sequences:", zeros)
    
    # Get indices corresponding to selected clusters using boolean indexing
    selected_indices = np.where(np.isin(global_keys, selected_clusters))[0]
    
    return np.array(selected_indices), selected_clusters
