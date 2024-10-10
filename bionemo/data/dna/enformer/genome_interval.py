# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from pathlib import Path
from random import random, randrange

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from pyfaidx import Fasta
from torch.utils.data import Dataset


# helper functions


def exists(val):
    return val is not None


def identity(t):
    return t


def cast_list(t):
    return t if isinstance(t, list) else [t]


def coin_flip():
    return random() > 0.5


# genomic function transforms

seq_indices_embed = torch.zeros(256).long()
seq_indices_embed[ord("a")] = 0
seq_indices_embed[ord("c")] = 1
seq_indices_embed[ord("g")] = 2
seq_indices_embed[ord("t")] = 3
seq_indices_embed[ord("n")] = 4
seq_indices_embed[ord("A")] = 0
seq_indices_embed[ord("C")] = 1
seq_indices_embed[ord("G")] = 2
seq_indices_embed[ord("T")] = 3
seq_indices_embed[ord("N")] = 4
seq_indices_embed[ord(".")] = -1

one_hot_embed = torch.zeros(256, 4)
one_hot_embed[ord("a")] = torch.Tensor([1.0, 0.0, 0.0, 0.0])
one_hot_embed[ord("c")] = torch.Tensor([0.0, 1.0, 0.0, 0.0])
one_hot_embed[ord("g")] = torch.Tensor([0.0, 0.0, 1.0, 0.0])
one_hot_embed[ord("t")] = torch.Tensor([0.0, 0.0, 0.0, 1.0])
one_hot_embed[ord("n")] = torch.Tensor([0.0, 0.0, 0.0, 0.0])
one_hot_embed[ord("A")] = torch.Tensor([1.0, 0.0, 0.0, 0.0])
one_hot_embed[ord("C")] = torch.Tensor([0.0, 1.0, 0.0, 0.0])
one_hot_embed[ord("G")] = torch.Tensor([0.0, 0.0, 1.0, 0.0])
one_hot_embed[ord("T")] = torch.Tensor([0.0, 0.0, 0.0, 1.0])
one_hot_embed[ord("N")] = torch.Tensor([0.0, 0.0, 0.0, 0.0])
one_hot_embed[ord(".")] = torch.Tensor([0.25, 0.25, 0.25, 0.25])

reverse_complement_map = torch.Tensor([3, 2, 1, 0, 4]).long()


def torch_fromstring(seq_strs):
    batched = not isinstance(seq_strs, str)
    seq_strs = cast_list(seq_strs)
    np_seq_chrs = [np.fromstring(t, dtype=np.uint8) for t in seq_strs]
    seq_chrs = list(map(torch.from_numpy, np_seq_chrs))
    return torch.stack(seq_chrs) if batched else seq_chrs[0]


def str_to_seq_indices(seq_strs):
    seq_chrs = torch_fromstring(seq_strs)
    return seq_indices_embed[seq_chrs.long()]


def str_to_one_hot(seq_strs):
    seq_chrs = torch_fromstring(seq_strs)
    return one_hot_embed[seq_chrs.long()]


def seq_indices_to_one_hot(t, padding=-1):
    is_padding = t == padding
    t = t.clamp(min=0)
    one_hot = F.one_hot(t, num_classes=5)
    out = one_hot[..., :4].float()
    out = out.masked_fill(is_padding[..., None], 0.25)
    return out


# augmentations


def seq_indices_reverse_complement(seq_indices):
    complement = reverse_complement_map[seq_indices.long()]
    return torch.flip(complement, dims=(-1,))


def one_hot_reverse_complement(one_hot):
    *_, n, d = one_hot.shape
    assert d == 4, "must be one hot encoding with last dimension equal to 4"
    return torch.flip(one_hot, (-1, -2))


class FastaInterval:
    """
    Returns subsequence described by location (sequence name, start position,
    end position)
    """

    def __init__(self, *, fasta_file, context_length=None, return_seq_indices=False, shift_augs=None, rc_aug=False):
        self.seqs = Fasta(fasta_file)
        self.return_seq_indices = return_seq_indices
        self.context_length = context_length
        self.shift_augs = shift_augs
        self.rc_aug = rc_aug

    def __call__(self, chr_name, start, end, return_augs=False):
        interval_length = end - start
        chromosome = self.seqs[chr_name]
        chromosome_length = len(chromosome)

        if exists(self.shift_augs):
            min_shift, max_shift = self.shift_augs
            max_shift += 1

            min_shift = max(start + min_shift, 0) - start
            max_shift = min(end + max_shift, chromosome_length) - end

            rand_shift = randrange(min_shift, max_shift)
            start += rand_shift
            end += rand_shift

        left_padding = right_padding = 0

        if exists(self.context_length) and interval_length < self.context_length:
            extra_seq = self.context_length - interval_length

            extra_left_seq = extra_seq // 2
            extra_right_seq = extra_seq - extra_left_seq

            start -= extra_left_seq
            end += extra_right_seq

            if start < 0:
                left_padding = -start
                start = 0

            if end > chromosome_length:
                right_padding = end - chromosome_length
                end = chromosome_length

        seq = ("." * left_padding) + str(chromosome[start:end]) + ("." * right_padding)

        if self.return_seq_indices:
            if self.rc_aug and coin_flip():
                seq = seq_indices_reverse_complement(seq)

            return str_to_seq_indices(seq)

        one_hot = str_to_one_hot(seq)

        rc_aug = self.rc_aug and coin_flip()

        if rc_aug:
            one_hot = one_hot_reverse_complement(one_hot)

        if not return_augs:
            return one_hot

        # returns the shift integer as well as the bool (for whether reverse complement was activated)
        # for this particular genomic sequence

        rand_shift_tensor = torch.tensor([rand_shift])
        rand_aug_bool_tensor = torch.tensor([rc_aug])

        return one_hot, rand_shift_tensor, rand_aug_bool_tensor


class FastaDataset(Dataset):
    def __init__(self, fasta_file: str, context_length: int):
        self.seqs = Fasta(fasta_file)
        self.context_length = context_length

    def __getitem__(self, idx):
        fa_record = self.seqs[idx]
        seq_length = len(fa_record)

        extra_seq = self.context_length - seq_length

        extra_left_seq = extra_seq // 2
        extra_right_seq = extra_seq - extra_left_seq

        seq = ("." * extra_left_seq) + str(fa_record) + ("." * extra_right_seq)
        return {"name": fa_record.name, "seq": str_to_one_hot(seq)}

    def __len__(self):
        return len(self.seqs.records)


class GenomeIntervalDataset(Dataset):
    def __init__(
        self,
        bed_file,
        fasta_file,
        filter_df_fn=identity,
        chr_bed_to_fasta_map={},
        context_length=None,
        return_seq_indices=False,
        shift_augs=None,
        rc_aug=False,
        return_augs=False,
    ):
        super().__init__()
        bed_path = Path(bed_file)
        assert bed_path.exists(), "path to .bed file must exist"

        df = pl.read_csv(str(bed_path), sep="\t", has_header=False)
        df = filter_df_fn(df)
        self.df = df

        # if the chromosome name in the bed file is different than the keyname in the fasta
        # can remap on the fly
        self.chr_bed_to_fasta_map = chr_bed_to_fasta_map

        self.fasta = FastaInterval(
            fasta_file=fasta_file,
            context_length=context_length,
            return_seq_indices=return_seq_indices,
            shift_augs=shift_augs,
            rc_aug=rc_aug,
        )

        self.return_augs = return_augs

    def __len__(self):
        return len(self.df)

    def __getitem__(self, ind):
        interval = self.df.row(ind)
        chr_name, start, end = (interval[0], interval[1], interval[2])
        chr_name = self.chr_bed_to_fasta_map.get(chr_name, chr_name)
        return self.fasta(chr_name, start, end, return_augs=self.return_augs)
