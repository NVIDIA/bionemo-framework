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


import os
from copy import deepcopy
from functools import lru_cache

import numpy as np
import pyfastx
from nemo.collections.nlp.data.language_modeling.text_memmap_dataset import TextMemMapDataset
from nemo.core import Dataset
from omegaconf import open_dict
from torch import distributed
from torch.utils.data import ConcatDataset

from bionemo.core import BioNeMoDataModule
from bionemo.data.dataloader import (
    KmerBertCollate,
    KmerTokenizerAdapter,
    LengthTruncator,
    SpanMasking,
)
from bionemo.data.dataloader.kmer_collate import DeterministicLengthTruncator
from bionemo.data.mapped_dataset import MappedDataset, ResamplingMappedDataset
from bionemo.data.utils import (
    DatasetBuilderSpec,
    FormattedDatasetFactory,
    expand_dataset_paths,
    handle_index,
)
from bionemo.tokenizer import KmerTokenizer


class _InMemoryFastxBackend:
    def __init__(self):
        self.records = {}

    def write(self, fastx):
        for seq in fastx:
            self.records[seq.name] = seq.seq

    def fetch(self, contig, start, end):
        return self.records[contig][start:end]


class _FetchFastxBackend:
    def __init__(self):
        self.fastx = None

    def write(self, fastx):
        self.fastx = fastx

    def fetch(self, contig, start, end):
        return self.fastx.fetch(contig, (start + 1, end))


BACKENDS = {
    "file": _FetchFastxBackend,
    "memory": _InMemoryFastxBackend,
}


class SafePyfastxFasta:
    """SafePyfastxFasta provides safety pytorch distributed by ensuring the pyfastx indexes are created only once.
    Previously, when more than one device was configured, a race condition would occur where all workers would try to build an index.
    If an index file already existed, the worker would attempt to use it, leading to a crash. This extension guards against this scenario.

    Since the original pyfastx is marked as final, we must override getattribute to mimic inheritence.
    """

    def __init__(self, *args, **kwargs):
        if distributed.is_available():
            if distributed.get_rank() == 0:
                self.fasta = pyfastx.Fasta(*args, **kwargs)
            distributed.barrier()
            if distributed.get_rank() != 0:
                self.fasta = pyfastx.Fasta(*args, **kwargs)
        else:
            return pyfastx.Fasta(*args, **kwargs)

    def __getattribute__(self, __name: str):
        fa = object.__getattribute__(self, "fasta")
        if __name == "fasta":
            return fa
        else:
            return getattr(fa, __name)

    def __iter__(self):
        for seq in self.fasta:
            yield seq


class PyfastxFastaDataset(Dataset):
    """
    Constructs a dataset for a pyfastx.Fasta

    See FASTA format definition here:
    http://www.ncbi.nlm.nih.gov/blast/fasta.shtml

    E.g., the following is a valid FASTA
    >seq1
    ACGTAC
    GAGATA
    >seq2
    TACATA

    A useful access would be, e.g.,
    >>> fasta = pyfasx.Fasta('example.fa')
    >>> dataset = PyfastxFastaDataset(fasta, max_length=4)
    >>> dataset[0]
    'ACGT'
    >>> dataset[4]
    'ACGA'
    >>> dataset[9]
    'ATA'
    >>> dataset[13]
    'ACAT'

    """

    def __init__(self, fastx_file, max_length, backend="file"):
        """

        Arguments:
            fastx_file (pyfastx.Fasta): Fasta object for the dataset.
            max_length (int): Maximum length of sub-sequences to fetch from the
                FASTA.
            backend (Literal['file', 'memory']): If 'file',
                the data is read from file on the fly with `pyfastx`. If
                'memory', the data is stored in memory.

        """
        self.fastx = fastx_file
        self.max_length = max_length
        self.backend = BACKENDS[backend]()
        self.backend.write(self.fastx)
        bins = [(seq.name, len(seq)) for seq in self.fastx]
        self.name_bins = [bin[0] for bin in bins]
        self.length_bins = np.cumsum([bin[1] for bin in bins])

    def fetch(self, id_, start, end):
        return self.backend.fetch(id_, start, end)

    def ids(self):
        return list(self.fastx.keys())

    def __len__(self):
        """Return the length of the dataset."""
        return self.length_bins[-1]

    def __getitem__(self, idx):
        """Return the subsequence starting at idx

        Returns:
            Dict: containing the sequence, the contig/description,
            and the position at the index.
        """
        idx = handle_index(self, idx)

        contig_idx = min(np.digitize(idx, self.length_bins), len(self.length_bins) - 1)
        contig_offset = self.length_bins[contig_idx - 1] if contig_idx > 0 else 0
        contig_name = self.name_bins[contig_idx]
        # wrap in int because floats here will cause errors in `fetch`
        position_start = int(idx - contig_offset)
        position_end = int(position_start + self.max_length)
        text = self.backend.fetch(contig_name, position_start, position_end)
        return {
            "seq": text,
            "contig": contig_name,
            "start": position_start,
        }


class InternallyIndexedFastaMemMapDataset(Dataset):
    """
    See FASTA format definition here:
    http://www.ncbi.nlm.nih.gov/blast/fasta.shtml
    E.g., the following is a valid FASTA
    >seq1
    ACGTAC
    GAGATA
    >seq2
    TACATA
    A useful access would be, e.g.,
    >>> dataset = InternallyIndexedFastaMemMapDataset(files, max_length=4)
    >>> dataset[0]
    'ACGT'
    >>> dataset[4]
    'ACGA'
    >>> dataset[9]
    'TACA'
    >>> dataset[11]
    'CATA'
    """

    def __init__(
        self,
        dataset_paths,
        max_length,
        workers=None,
        sort_dataset_paths=True,
        index_mapping_dir=None,
    ):
        """
        Args:
            dataset_paths (List[str]): FASTA files to be included.
                Must be unzipped.
            max_length (int): Length of sequences to load. Sequences will be
                less than `max_length` if they are in the last `max_length` - 1
                bases in the entry.
            workers (Optional[int]): Number of works used to build the file
                indexes.
            sort_dataset_paths (Optional[bool]): If True, the files are sorted
                in lexicographic order for accession.
        """
        self.max_length = max_length
        self.dataset_paths = dataset_paths
        self._dataset = TextMemMapDataset(
            dataset_paths,
            newline_int=ord(">"),
            header_lines=1,
            workers=workers,
            sort_dataset_paths=sort_dataset_paths,
            index_mapping_dir=index_mapping_dir,
        )

        self.n_cumulative_entries = self._dataset.midx_bins
        self.characters_and_entry_idx_by_file = self._dataset.mdata_midx_list
        self.n_fasta_entries = self._dataset.midx_bins[-1]

        # TODO: saving indices.
        # splitting by file is nice because it makes the datasets composable,
        # but the current indices make use of global positions. In order to
        # properly store indices, we would need to ensure proper globalization/
        # relativization of the indices on load/save respectively.

        self.n_seq_bases_per_entry = np.zeros(len(self._dataset), dtype=int)
        self.all_newlines = []
        for idx in range(len(self._dataset)):
            _, mdata, start, end = self.get_fileid_mdata_start_end(self._dataset, idx)
            segment = mdata[start:end]
            # newlines tracks the locations of newline characters
            newlines = np.where(segment == ord("\n"))[0]
            # make correction to the index if there are no newlines at end of file
            if newlines[-1] != len(segment) - 1:
                newlines = np.concatenate([newlines, [len(segment)]])

            # n_seq_chars is the index for the number of bases that have been
            # observed in a sample at the end of each line
            # the calculation is the number of cumulative characters in the sample
            # at the end of the line, minus the initial line length
            # (newlines[0], because this contains the description of the
            # sequence but no bases) and minus the cumulative
            # number of newline characters (np.arange(...)) so these are not
            # counted toward the number of sequenced bases.
            n_seq_chars = newlines - (newlines[0] + np.arange(len(newlines)))
            self.n_seq_bases_per_entry[idx] = n_seq_chars[-1] + (self.n_seq_bases_per_entry[idx - 1] if idx > 0 else 0)
            self.all_newlines.append(newlines)

        self.non_newline_chars = self.n_seq_bases_per_entry[-1]
        # length_bins for compatibility with discretize
        self.length_bins = self.n_seq_bases_per_entry

    def __getitem__(self, index):
        index = handle_index(self, index)
        # TODO: this lookup is currently a little roundabout: it tells you which entry
        # we want to go to, then uses that to lookup the file and get the entry position
        # within the file. We can probably restructure the indices to go directly
        # to the entry without needing to know its global position
        entry = np.digitize(index, self.n_seq_bases_per_entry, right=False)
        position_within_entry = index - (self.n_seq_bases_per_entry[entry - 1] if entry > 0 else 0)
        end_position_within_entry = position_within_entry + min(
            self.n_seq_bases_per_entry[entry] - index, self.max_length
        )

        _, mdata, start, _ = self.get_fileid_mdata_start_end(self._dataset, entry)
        # there should be an index to find the start and end of a given section within
        #  the flattened all_newlines index
        # currently this looks like: [array([ 4, 11, 19]), array([ 4, 10]),
        #   array([ 6, 15, 19]), array([ 6, 14, 21])]
        # we want it too look like: [4, 11, 19, 4, 10, 6, 15, 19, 6, 14, 21]
        # with a structure like:
        # starts: [0, 3, 5, 8, -1]
        newlines = self.all_newlines[entry]
        n_seq_chars = newlines - (newlines[0] + np.arange(len(newlines)))
        start_line = np.digitize(position_within_entry, n_seq_chars, right=False) - 1
        end_line = np.digitize(end_position_within_entry, n_seq_chars, right=False) - 1

        consume_leading_chars = position_within_entry - n_seq_chars[start_line]
        consume_trailing_chars = end_position_within_entry - n_seq_chars[end_line]
        seq_start = start + newlines[start_line] + consume_leading_chars + 1
        seq_end = start + newlines[end_line] + consume_trailing_chars + 1

        return {
            "seq": mdata[seq_start:seq_end].tobytes().decode().replace("\n", ""),
            "contig": mdata[start : start + newlines[0]].tobytes().decode(),
            "start": position_within_entry,
        }

    @staticmethod
    def get_entries_per_file(dataset: TextMemMapDataset):
        entries_per_file = np.concatenate([dataset.midx_bins[0:1], dataset.midx_bins[1:] - dataset.midx_bins[:-1]])
        return entries_per_file

    def __len__(self):
        return self.non_newline_chars

    @staticmethod
    def get_fileid_mdata_start_end(dataset: TextMemMapDataset, idx):
        # Identify the file containing the record
        file_id = np.digitize(idx, dataset.midx_bins, right=False)
        base_idx = dataset.midx_bins[file_id - 1] if file_id > 0 else 0
        file_idx = idx - base_idx + dataset._header_lines
        mdata, midx = dataset.mdata_midx_list[file_id]
        # load sample
        if file_idx == 0:
            i = 0
            j = midx[0]
        else:
            i = midx[file_idx - 1] + 1  # ignore newline
            j = midx[file_idx]
        return file_id, mdata, i, j


class DiscretizeFastaDataset(MappedDataset):
    def __init__(self, dataset: PyfastxFastaDataset):
        """
        Produces a discretized version of a `FastaDataset`.

        `DiscretizeFastaDataset` wraps a `FastaDatset` to produce non-overlapping
        subsections, e.g.: a dataset built on `ACGT` with sequence length of 2
        will only yield `AC` and `GT`. This is in contrast to the
        `FastaDataset`, which provides overlapping sections of the sequence,
        e.g., `ACGT` with a sequence length of 2 will return `AC`, `CG`, `GT`,
        and `T`.

        Arguments:
            dataset (FastaDataset): a dataset to discretize. The chunk size
                used to discretize the dataset will be inferred from `dataset`.

        """
        super().__init__(dataset, None)

    def fetch(self, id_, start, end):
        return self._dataset.fetch(id_, start, end)

    def ids(self):
        return self._dataset.ids()

    @staticmethod
    def create_sample_mapping(dataset, num_samples):
        """Sample mapping used to remap a dataset into discrete sections.

        Arguments:
            dataset (FastaDataset): dataset to discretize
            num_samples (None): Unused.

        Returns:
            sample_mapping (ArrayLike[int]): If `sample_mapping[i] == j`,
            the `i`th entry in this dataset will be `j`th entry of the original
            dataset.

        """
        first_bin = dataset.length_bins[0]
        rest_bin = dataset.length_bins[1:] - dataset.length_bins[:-1]
        bins = np.append(first_bin, rest_bin)
        # shortcut for ceil division
        n_seqs_per_bin = -(bins // -dataset.max_length)
        sample_mapping = np.zeros(n_seqs_per_bin.sum(), dtype=int)
        pos = 0
        for i, n_seqs in enumerate(n_seqs_per_bin):
            initial_start = 0 if i == 0 else dataset.length_bins[i - 1]
            for j in range(n_seqs):
                start = initial_start + j * dataset.max_length
                sample_mapping[pos] = start
                pos += 1
        return sample_mapping


class ConcatFastaDataset(Dataset):
    def __init__(self, files, max_length, backend="file", uppercase=False, transforms=None):
        """
        Constructs a dataset consisting of multiple FASTA files.

        Arguments:
            files (List[str]): Each entry contains a filepath to a FASTA file.
                The files can be zipped (gzip) or not zipped. An arbtrary
                mixture of both zipped and not zipped is supported.
            max_length (int): Maximum length of sub-sequences to fetch from the
                FASTA.
            backend (Literal['file', 'memory']): Either 'file' or 'memory'. If 'file',
                the data is read from file on the fly with `pyfastx`. If
                'memory', the data is stored in memory.
            uppercase (Optional[bool]). If True, the sequences are coerced to
                uppercase. Otherwise, they are provided in the original case
                from the file.
            transforms (Optional[List[Callable[[Dataset], Dataset]]): If
                specified, each entry of `transforms` returns a dataset. The
                entries will be applied sequentially to the components of this
                dataset.

        """
        if transforms is None:
            transforms = []
        self.transforms = transforms
        self.datasets = []

        pyfastx_cls = SafePyfastxFasta if distributed.is_initialized() else pyfastx.Fasta
        for f in files:
            new_dataset = PyfastxFastaDataset(
                pyfastx_cls(f, uppercase=uppercase),
                max_length,
                backend=backend,
            )
            self.datasets.append(self._apply_transforms(new_dataset))

        self._dataset = ConcatDataset(self.datasets)
        self.ids = {}
        self.build_id_index()

    def build_id_index(self):
        for ds in self.datasets:
            if hasattr(ds, "ids"):
                for key in ds.ids():
                    # TODO warn for overriding key?
                    self.ids[key] = ds

    def fetch(self, id_, start, end):
        ds = self.ids[id_]
        return ds.fetch(id_, start, end)

    def _apply_transforms(self, dataset):
        for transform in self.transforms:
            dataset = transform(dataset)
        return dataset

    def __len__(self):
        """Return the length of the dataset."""
        return len(self._dataset)

    def __getitem__(self, idx):
        """Return the subsequence starting at idx

        Returns:
            Dict: containing the sequence, the contig/description,
            and the position at the index.
        """
        return self._dataset[idx]


def pyfastx_constructor(builder, discretize, max_length):
    transforms = builder.make_transforms(discretize)
    return ConcatFastaDataset(
        builder.dataset_paths,
        max_length,
        backend=builder.options["dataset_backend"],
        transforms=transforms,
    )


def fasta_memmap_constructor(builder, discretize, max_length):
    if discretize:
        dataset = DiscretizeFastaDataset(InternallyIndexedFastaMemMapDataset(list(builder.dataset_paths), max_length))
    else:
        dataset = InternallyIndexedFastaMemMapDataset(
            builder.dataset_paths,
            max_length,
        )
    return dataset


tokenizers = {
    "kmer": KmerTokenizer,
}

adapters = {
    "kmer": KmerTokenizerAdapter,
}

dataset_constructors = {"pyfastx": pyfastx_constructor, "memmap": fasta_memmap_constructor}


class FastaDatasetBuilder(DatasetBuilderSpec):
    def format_dataset_paths(self):
        """
        Parses FASTA paths.

        """
        self.dataset_paths = expand_dataset_paths(self.options["filepath"], None)

    def check_path(self, filepath):
        """
        Checks whether a FASTA exists.

        Arguments:
            filepath (str): a string that can be used to identify the filepath

        Returns:
            Optional[str]: If the file exists, this returns None, otherwise
                it returns the on the filepath.

        """
        if not os.path.exists(filepath):
            return filepath

    def create_dataset(self):
        """
        Instantiates a FastaDataset.

        Returns:
            Dataset: Dataset instantiated from paths.
        """
        cfg = self.options["cfg"]
        discretize = self.options["discretize"]
        dataset_class = self.options["dataset_class"]
        max_length = cfg.seq_length - 1 + cfg.k
        constructor = dataset_constructors[dataset_class]
        self.dataset = constructor(self, discretize=discretize, max_length=max_length)
        return self.dataset

    def make_transforms(self, discretize):
        """
        Makes transformations to use for the a Dataset.

        Arguments:
            discretize (bool): whether the Discretize argument should be added
        Returns:
            List[Callable[[Dataset], Dataset]]: Dataset transformations

        """
        transforms = (
            [
                DiscretizeFastaDataset,
            ]
            if discretize
            else []
        )
        return transforms


class DNABERTDatasetFactory(FormattedDatasetFactory):
    def __init__(self):
        """
        Initializes a dataset factory for handling fasta formats.
        """
        self.formats = {
            "fasta": FastaDatasetBuilder,
        }


tokenizers = {
    "kmer": KmerTokenizer,
}

adapters = {
    "kmer": KmerTokenizerAdapter,
}


class DNABERTDataModule(BioNeMoDataModule):
    def __init__(self, cfg, trainer):
        """Initializes a DNABERTDataModule

        Arguments:
            cfg (OmegaConf): A config object for a model
            trainer (pytorch_lightning.Trainer): Trainer of the corresponding
                model.

        """
        super().__init__(cfg, trainer)
        self.pad_size_divisible_by_8 = True if self.model_cfg.masked_softmax_fusion else False

    # this only needs to be run once, but it shouldn't be called on initialization,
    #  and we are not entirely sure where it will be called first
    @lru_cache
    def onetime_init_num_samples(self):
        self.init_num_samples()

    # this can probably be shared by some BioNeMo models that already use the
    # dataset factory or should be refactored from `build_train_valid_test_datasets`
    def _configure_options(self, name, ds):
        cfg = deepcopy(self.cfg)
        with open_dict(cfg):
            dataset_path = cfg.get("dataset_path", "")
            dataset_format = cfg.get("dataset_format")
            dataset_class = cfg.get("dataset_class", "memmap")
            dataset_backend = cfg.get("dataset_backend")
            if dataset_class == "pyfastx":
                if dataset_backend not in BACKENDS.keys():
                    raise ValueError(
                        f"dataset_backend={dataset_backend} for dataset_class="
                        f"{dataset_class} must be configured to one of: "
                        f"{BACKENDS.keys()}"
                    )

        # Build individual datasets.
        filepath = os.path.join(dataset_path, name, ds)

        options = {
            "cfg": cfg,
            "filepath": filepath,
            "dataset_format": dataset_format,
            "batch_size": self.get_global_batch_size(),
            "dataset_class": dataset_class,
            "dataset_backend": dataset_backend,
        }
        return options

    @staticmethod
    def _custom_get_sentence(x):
        """Replaces lambda usage below so this can be properly pickled."""
        return x["seq"]

    @staticmethod
    def _get_random_length_truncator():
        sentence_transform = LengthTruncator()
        sentence_transform.get_sentence = DNABERTDataModule._custom_get_sentence
        return sentence_transform

    @staticmethod
    def _get_deterministic_length_truncator():
        sentence_transform = DeterministicLengthTruncator()
        sentence_transform.get_sentence = DNABERTDataModule._custom_get_sentence
        return sentence_transform

    def _setup_collate(self, model, dataloader, sentence_transform):
        dataloader.collate_fn = KmerBertCollate(
            model.tokenizer,
            seq_length=self.cfg.seq_length,
            pad_size_divisible_by_8=self.pad_size_divisible_by_8,
            masking_strategy=SpanMasking(
                tokenizer=model.tokenizer,
                seed_probability=0.15,
                span_length=model._cfg.tokenizer.k,
            ),
            transform_sentences=sentence_transform,
            dynamic_padding=self.cfg.dynamic_padding,
        ).collate_fn

    def _setup_eval_dataloader(self, model, dataloader):
        sentence_transform = self._get_deterministic_length_truncator()
        self._setup_collate(model, dataloader, sentence_transform)

    def train_dataset(self):
        """Creates a Training dataset for DNABERT

        Returns:
            Dataset: A Dataset.

        """
        ds = self.cfg.dataset.train
        name = "train"
        options = self._configure_options(name, ds)
        options["discretize"] = self.cfg.discretize_train

        dataset = DNABERTDatasetFactory().create_dataset(options)
        return dataset

    def sample_train_dataset(self, dataset):
        """Creates an upsampled version of the DNABERT training dataset"""
        self.onetime_init_num_samples()
        num_samples = self.train_num_samples
        dataset_dir = os.path.join(
            self.cfg.dataset_path,
            "train",
        )
        dataset = ResamplingMappedDataset(
            dataset,
            num_samples=num_samples,
            cfg=self.cfg,
            data_prefix=self.cfg.dataset.train,
            index_mapping_dir=dataset_dir,
            name="train",
        )
        return dataset

    def val_dataset(self):
        """
        Produces a discretized FASTA dataset to use for validation.
        """
        ds = self.cfg.dataset.val
        name = "val"
        options = self._configure_options(name, ds)
        options["discretize"] = True
        options["name"] = "val"
        dataset = DNABERTDatasetFactory().create_dataset(options)
        return dataset

    def sample_val_dataset(self, dataset):
        self.onetime_init_num_samples()
        num_samples = self.val_num_samples
        dataset_dir = os.path.join(
            self.cfg.dataset_path,
            "val",
        )
        dataset = ResamplingMappedDataset(
            dataset,
            num_samples=num_samples,
            cfg=self.cfg,
            data_prefix=self.cfg.dataset.val,
            index_mapping_dir=dataset_dir,
            name="val",
        )
        return dataset

    def test_dataset(self):
        """
        Produces a discretized FASTA dataset to use for testing.
        """
        ds = self.cfg.dataset.test
        name = "test"
        options = self._configure_options(name, ds)
        options["discretize"] = True
        return DNABERTDatasetFactory().create_dataset(options)

    def adjust_train_dataloader(self, model, dataloader):
        sentence_transform = self._get_random_length_truncator()
        return self._setup_collate(model, dataloader, sentence_transform)

    def adjust_val_dataloader(self, model, dataloader):
        return self._setup_eval_dataloader(model, dataloader)

    def adjust_test_dataloader(self, model, dataloader):
        return self._setup_eval_dataloader(model, dataloader)
