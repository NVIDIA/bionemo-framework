import os
import polars as pl
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from typing import Callable
import copy
from pathlib import Path
from typing import List, Tuple
from collections import defaultdict
import logging
from src.data.metadata import MetadataFields
logger = logging.getLogger(__name__)

class MissenseDataset(Dataset):
    def __init__(self, data_path, tokenizer, process_item: Callable, num_variants_per_seq=5,
                 train_val_test_ratio=[0.8, 0.1, 0.1], context_length=2048, use_weights=False,
                 center_weight_threshold=0.3, use_paiv1=False, use_am=False, n_per_benign=1,
                 random_mask_prob=0.3, mask_replace_prob=0.8, random_replace_prob=0.1,
                 seed=42, **kwargs):
        self.num_variants_per_seq = num_variants_per_seq
        self.seed = seed
        self.context_length = context_length
        self.tokenizer = tokenizer
        self.process_item = process_item
        self.random_mask_prob = random_mask_prob
        self.mask_replace_prob = mask_replace_prob
        self.random_replace_prob = random_replace_prob
        self.use_weights = use_weights
        self.use_paiv1 = use_paiv1
        self.use_am = use_am
        self.center_weight_threshold = center_weight_threshold
        self.n_per_benign = n_per_benign
        self.rng = np.random.default_rng(seed)

        post_fix = ""
        if use_paiv1:
            assert self.use_weights, "use_weights must be True if use_paiv1 is True"
        if use_am:
            assert self.use_weights, "use_weights must be True if use_am is True"
        if use_weights:
            post_fix += '.af_weighted'
            if use_paiv1:
                post_fix += '.with_paiv1'
            if use_am:
                post_fix += '.with_am'
            post_fix += f'.npb{self.n_per_benign}' if self.n_per_benign > 1 else ''
            post_fix += f'.center{self.center_weight_threshold:.1f}'
        benign_path = os.path.join(data_path, "benign" + post_fix + ".csv")
        pathogenic_path = os.path.join(data_path, "sampled_pathogenic" + post_fix + ".csv")
        transcripts_path = os.path.join(data_path, "transcripts.tsv")
        benign = pl.from_pandas(pd.read_csv(benign_path)).with_columns(pl.lit(0).alias('label'))
        pathogenic = pl.from_pandas(pd.read_csv(pathogenic_path)).with_columns(pl.lit(1).alias('label'))
        variants = pl.concat([benign, pathogenic], how='vertical')
        variants = variants.with_columns(
                pl.format("{},{}:{}>{}", pl.col('chrom'), pl.col('genomic_pos'), pl.col('ref'), pl.col('alt')).alias('key'),
                (pl.col('var_rel_pos_in_cds') // 3).alias('codon_position'),
            )
        self.unique_variants = sorted(variants.unique(subset=['key'])['key'].to_list())
        if self.use_weights:
            center_variants = sorted(variants.filter(pl.col('weight')>center_weight_threshold).unique(subset=['key'])['key'].to_list())
            center_variants = set(center_variants)
            self.center_idxs = np.array([i for i, v in enumerate(self.unique_variants) if v in center_variants])
        else:
            self.center_idxs = np.arange(len(self.unique_variants))
        self.variant_key_to_idx = {key: idx for idx, key in enumerate(self.unique_variants)}

        # Create mapping from variant key to its index in unique_variants
        transcripts = pl.from_pandas(pd.read_csv(transcripts_path, sep='\t')).with_columns(pl.col('name').alias('transcript_id'))
        key_to_transcripts_df = (
                    variants
                    .select(['key', 'transcript_id'])
                    .group_by('key')
                    .agg(pl.col('transcript_id').unique().alias('transcripts'))
                )
        variant_key_to_transcripts = {
            k: sorted(ts) if isinstance(ts, list) else list(ts)
            for k, ts in zip(key_to_transcripts_df['key'].to_list(), key_to_transcripts_df['transcripts'].to_list())
        }   

        self.variant_key_to_transcripts = variant_key_to_transcripts
        transcript_ids = transcripts.select('transcript_id').to_series().to_list()
        cds_sequences = transcripts.select('cds_sequence').to_series().to_list()
        self.transcript_to_cds = {t: s for t, s in zip(transcript_ids, cds_sequences)}

        # Precompute mappings to avoid per-call Polars ops
        # 1) (key, transcript_id) -> variant dict for direct center variant lookup
        # 2) transcript_id -> { codon_position -> [variant_dict, ...] } for neighbor sampling
        self.key_transcript_to_variant = {}
        self.transcript_to_pos_variants = {}
        for v in variants.to_dicts():
            tid = v['transcript_id']
            pos = v['codon_position']
            self.key_transcript_to_variant[(v['key'], tid)] = v
            self.transcript_to_pos_variants.setdefault(tid, {}).setdefault(pos, []).append(v)

        self.train_idx, self.val_idx, self.test_idx = None, None, None
        if train_val_test_ratio:
            self.train_idx, self.val_idx, self.test_idx = self.load_train_val_test_indices_by_pos(data_path, train_val_test_ratio, seed)
        self.idxs = self.train_idx if self.train_idx is not None else np.arange(len(self.unique_variants))
        self.reset_idxs()

    def reset_idxs(self):
        self.idxs_for_sampling = np.intersect1d(self.center_idxs, self.idxs)
        self.allowed_variant_indices = set(self.idxs)
        
    def __len__(self):
        return len(self.idxs_for_sampling)
    
    def __getitem__(self, idx):
        # Center variant key
        center_key = self.unique_variants[self.idxs_for_sampling[idx]]
        transcripts = self.variant_key_to_transcripts.get(center_key, [])
        assert len(transcripts) > 0, f"No transcript found for variant {center_key}"

        # Choose a transcript and fetch data via precomputed dicts
        selected_transcript = self.rng.choice(transcripts)
        cds_sequence = self.transcript_to_cds[selected_transcript]
        center_variant = self.key_transcript_to_variant[(center_key, selected_transcript)]

        sampled = []

        items = self.process_item(cds_sequence, center_variant, sampled, 
                                tokenizer=self.tokenizer, 
                                N=self.num_variants_per_seq,
                                use_weights=self.use_weights)

        items[MetadataFields.ID] = self.idxs_for_sampling[idx]
        return items

    def load_train_val_test_indices_by_pos(self, data_path: str, train_val_test_ratio: List[float], seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        data_path = Path(data_path)
        dir_path = data_path if data_path.is_dir() else data_path.parent
        post_fix = "_pos_split" + "_weighted" if self.use_weights else ""
        post_fix += "_paiv1" if self.use_paiv1 else ""
        post_fix += "_am" if self.use_am else ""
        post_fix += f'_npb{self.n_per_benign}' if self.n_per_benign > 1 else ''
        post_fix += f'_center{self.center_weight_threshold:.1f}' if self.use_weights else ""
        if train_val_test_ratio[2] == 0:
            post_fix += "_notest"
        train_idx_path = dir_path / f'train_idx{post_fix}.npy'
        val_idx_path = dir_path / f'val_idx{post_fix}.npy'
        test_idx_path = dir_path / f'test_idx{post_fix}.npy'
        if train_idx_path.exists() and val_idx_path.exists() and test_idx_path.exists():
            train_idx = np.load(train_idx_path)
            val_idx = np.load(val_idx_path)
            test_idx = np.load(test_idx_path)
        else:
            if len(train_val_test_ratio) != 3:
                raise ValueError("train_val_test_ratio must have 3 values")
            # normalize to sum to 1
            train_val_test_ratio = np.array(train_val_test_ratio)
            if train_val_test_ratio.sum() <= 0:
                raise ValueError("train_val_test_ratio must sum to a positive number")
            train_val_test_ratio = train_val_test_ratio / train_val_test_ratio.sum()
            logger.info(f"train_val_test_ratio: {train_val_test_ratio}")

            # Split variants by genomic position (chromosome + genomic_pos)
            # First, get unique genomic positions across all variants
            unique_positions = set()
            variant_to_pos = {}
            
            for variant_idx, variant_key in enumerate(self.unique_variants):
                # Get position info from any transcript (all should have same genomic coords)
                pos_key = variant_key.split(':')[0]
                unique_positions.add(pos_key)
                variant_to_pos[variant_idx] = pos_key

            # Convert to sorted list for deterministic splitting
            sorted_positions = sorted(list(unique_positions))
            num_positions = len(sorted_positions)
            logger.info(f"Number of unique genomic positions: {num_positions}")

            # Split positions by ratio
            rng = np.random.RandomState(seed)
            position_indices = rng.permutation(num_positions)
            
            train_pos_size = int(train_val_test_ratio[0] * num_positions)
            val_pos_size = int(train_val_test_ratio[1] * num_positions)

            train_position_indices = position_indices[:train_pos_size]
            val_position_indices = position_indices[train_pos_size:train_pos_size+val_pos_size]
            test_position_indices = position_indices[train_pos_size+val_pos_size:]

            # Map position indices back to position keys
            train_positions = set(sorted_positions[i] for i in train_position_indices)
            val_positions = set(sorted_positions[i] for i in val_position_indices)
            test_positions = set(sorted_positions[i] for i in test_position_indices)

            # Assign variants to splits based on their genomic positions
            train_idx = []
            val_idx = []
            test_idx = []

            for variant_idx, pos_key in variant_to_pos.items():
                if pos_key in train_positions:
                    train_idx.append(variant_idx)
                elif pos_key in val_positions:
                    val_idx.append(variant_idx)
                elif pos_key in test_positions:
                    test_idx.append(variant_idx)

            train_idx = np.array(sorted(train_idx))
            val_idx = np.array(sorted(val_idx))
            test_idx = np.array(sorted(test_idx))

            logger.info(f"Position-based split:")
            logger.info(f"Train variants: {len(train_idx)}, Val variants: {len(val_idx)}, Test variants: {len(test_idx)}")
            logger.info(f"Total variants: {len(train_idx) + len(val_idx) + len(test_idx)} (expected: {len(self.unique_variants)})")

            # Save the splits
            np.save(train_idx_path, train_idx)
            np.save(val_idx_path, val_idx)
            np.save(test_idx_path, test_idx)

        return train_idx, val_idx, test_idx

    def copy(self):
        new_copy = copy.copy(self)
        new_copy.rng = np.random.default_rng(self.seed)
        return new_copy

    def get_num_samples(self, split):
        if split == "train":
            return self.get_train_num_samples()
        elif split == "valid":
            return self.get_val_num_samples()
        elif split == "test":
            return self.get_test_num_samples()
        else:
            raise ValueError(f"Invalid split: {split}")

    def get_train_num_samples(self):
        l = len(self.center_idxs) if self.train_idx is None else len(np.intersect1d(self.center_idxs, self.train_idx))
        return l

    def get_val_num_samples(self):
        l = len(self.center_idxs) if self.val_idx is None else len(np.intersect1d(self.center_idxs, self.val_idx))
        return l

    def get_test_num_samples(self):
        l = len(self.center_idxs) if self.test_idx is None else len(np.intersect1d(self.center_idxs, self.test_idx))
        return l

    def get_train(self, process_item):
        """modifies indices to correspond to `train` split"""
        assert self.train_idx is not None, "train_idx is not loaded"
        self.process_item = process_item
        self.idxs = self.train_idx
        self.reset_idxs()
        return self
    
    def get_validation(self, process_item):
        """modifies indices to correspond to `valid` split"""
        assert self.val_idx is not None, "val_idx is not loaded"
        copy = self.copy()
        copy.process_item = process_item
        copy.idxs = self.val_idx
        copy.reset_idxs()
        copy.num_variants_per_seq = 1
        return copy
    
    def get_test(self, process_item):
        """modifies indices to correspond to `test` split"""
        assert self.test_idx is not None, "test_idx is not loaded"
        copy = self.copy()
        copy.process_item = process_item
        copy.idxs = self.test_idx
        copy.reset_idxs()
        copy.num_variants_per_seq = 1
        return copy