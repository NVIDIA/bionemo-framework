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


import gc
import os
from typing import List, Optional

import numpy as np
import torch
from nemo.utils import logging
from torch.utils.data import Dataset
from tqdm import tqdm, trange

from bionemo.core import BioNeMoDataModule
from bionemo.data.utils import expand_dataset_paths
from bionemo.model.utils import _reconfigure_inference_batch, get_from_encoder_or_model
from bionemo.tokenizer.label2id_tokenizer import Label2IDTokenizer


def columns_in_header(cols: List[str], header: List[str], allow_none: Optional[bool] = False):
    for col in cols:
        if col is None and not allow_none:
            raise ValueError("Column name {} is not in file header {}".format(col, header))
        elif col is not None and col not in header:
            raise ValueError("Column name {} is not in file header {}".format(col, header))
    return True


class PerTokenValueDataModule(BioNeMoDataModule):
    def __init__(self, cfg, trainer, model):
        super().__init__(cfg, trainer)
        self.model = model
        self.tokenizers = [Label2IDTokenizer() for _ in range(len(self.cfg.target_sizes))]
        if "shuffle" in self.cfg.keys():
            self.shuffle = self.cfg.shuffle
        else:
            self.shuffle = False

    def _update_tokenizers(self, tok, labels):
        tokenizers = [tok[i].build_vocab(np.array(labels)[:, i]) for i in range(len(tok))]
        return tokenizers

    def _create_dataset(self, split, files):
        datafiles = os.path.join(self.cfg.dataset_path, split, files)
        datafiles = expand_dataset_paths(datafiles, ".csv")
        dataset = get_data(
            datafiles=datafiles,
            model=self.model,
            max_seq_length=self.cfg.max_seq_length,
            emb_batch_size=self.cfg.emb_batch_size,
            labels_size=self.cfg.target_sizes,
            mask_col=self.cfg.mask_column,
            labels_col=self.cfg.target_column,
            sequence_col=self.cfg.sequence_column,
        )
        self.tokenizers = self._update_tokenizers(self.tokenizers, dataset.labels_str)
        dataset.labels = get_data._tokenize_labels(self.tokenizers, dataset.labels_str, dataset.labels_sizes)
        return dataset

    def train_dataset(self):
        """Creates a training dataset
        Returns:
            Dataset: dataset to use for training
        """
        self.train_ds = self._create_dataset("train", self.cfg.dataset.train)
        return PerTokenValueDataset(self.train_ds, shuffle=self.shuffle)

    def val_dataset(self):
        if "val" in self.cfg.dataset:
            self.val_ds = self._create_dataset("val", self.cfg.dataset.val)
            return PerTokenValueDataset(self.val_ds)
        else:
            pass

    def test_dataset(self):
        if "test" in self.cfg.dataset:
            self.test_ds = self._create_dataset("test", self.cfg.dataset.test)
            return PerTokenValueDataset(self.test_ds)
        else:
            pass


class PerTokenValueDataset(Dataset):
    def __init__(self, data, shuffle=False):
        self.data = data
        self.idxs = list(range(self.data.length()))
        self.shuffle = shuffle
        if shuffle:
            np.random.shuffle(self.idxs)

    def __len__(self):
        return self.data.length()

    def __getitem__(self, idx):
        if self.shuffle:
            idx = self.idxs[idx]
        embeddings = self.data.get_embeddings(idx)
        if isinstance(embeddings, str):
            seq_len = len(embeddings)
        else:
            seq_len, emb_dim = embeddings.size()
            embeddings = torch.cat([embeddings, torch.zeros((self.data.max_length - seq_len), emb_dim)])
        labels = self.data.get_labels(idx)
        item_dict = {"embeddings": embeddings, "seq_len": seq_len}
        for i in range(len(self.data.label_names)):
            name = self.data.label_names[i]
            padding = torch.zeros((self.data.max_length - seq_len), self.data.labels_sizes[i])
            item_dict[name] = torch.cat([labels[i], padding])
            mask = torch.tensor(self.data.label_masks[idx][i])
            mask_padding = torch.zeros((self.data.max_length - seq_len))
            item_dict["_".join(["mask", name])] = torch.cat([mask, mask_padding])
        return item_dict

    def free_memory(self):
        members = [
            attr for attr in dir(self.data) if not callable(getattr(self.data, attr)) and not attr.startswith("_")
        ]
        for m in members:
            self.data.__delattr__(m)
        gc.collect()

    @staticmethod
    def prepare_batch(batch, data, task=None):
        label_names = data.data.label_names
        num_labels = len(label_names)
        max_batch_seq_len = batch["seq_len"].max()
        embeddings = batch["embeddings"]
        if not isinstance(embeddings, list):
            embeddings = embeddings[:, :max_batch_seq_len, :].to("cuda")
        labels = []
        masks = []
        for i in range(num_labels):
            cur_label = batch[label_names[i]][:, :max_batch_seq_len, :]
            labels.append(cur_label.to("cuda"))
            mask = batch["_".join(["mask", label_names[i]])][:, :max_batch_seq_len].to("cuda")
            masks.append(mask)
        return embeddings, (labels, masks)


class get_data:
    def __init__(
        self,
        datafiles: List,
        model,
        max_seq_length: int,
        labels_size: List[int],
        sequence_col: str,
        labels_col: List[str],
        mask_col: Optional[List[str]] = None,
        delimiter=",",
        emb_batch_size=None,
    ):
        self.datafiles = datafiles
        self.model = model
        self.labels_sizes = labels_size
        self.mask_col = mask_col
        self.label_masks = []
        self.delimiter = delimiter
        self.hiddens = []
        self.emb_masks = []
        self.n_labels = len(labels_size)
        self.labels = []
        self.labels_str = []
        self.input_seq = []
        self.label_names = labels_col
        self.sequence_col = sequence_col
        self.max_length = max_seq_length
        if model is not None:
            self.prot_model_batch_size = model.cfg.model.micro_batch_size
            self.emb_batch_size = emb_batch_size
        self._get_data()
        if len(self.mask_col) != self.n_labels:
            raise ValueError("Size mask_col doesn't match size of labels_size")
        if len(self.label_names) != self.n_labels:
            raise ValueError("Size labels_col doesn't match size of labels_size")

    def _get_data(self):
        for fname in self.datafiles:
            with open(fname, "r") as f:
                for line_idx, line in tqdm(enumerate(f)):
                    # read header and check that all columns are present
                    if line_idx == 0:
                        header = line.split(self.delimiter)
                        header = [h.strip() for h in header]
                        if (
                            columns_in_header(self.label_names, header)
                            and columns_in_header(self.mask_col, header, allow_none=True)
                            and columns_in_header([self.sequence_col], header)
                        ):
                            logging.info("Reading file {}...".format(fname))
                        continue
                    line = line.split(self.delimiter)
                    sequence = line[header.index(self.sequence_col)].strip()
                    if len(sequence) > self.max_length - 2:
                        continue
                    self.input_seq.append(sequence)
                    cur_label = []
                    cur_mask = []
                    for i in range(self.n_labels):
                        l = line[header.index(self.label_names[i])].strip()
                        cur_label.append(l)
                        if self.mask_col[i] is not None:
                            m = line[header.index(self.mask_col[i])]
                            m = np.array(list(m.strip()), dtype="int")
                        else:
                            m = np.array([1] * len(l))
                        cur_mask.append(m)
                    self.labels_str.append(cur_label)
                    self.label_masks.append(cur_mask)

        if self.model is not None:
            with torch.no_grad():
                _reconfigure_inference_batch(self.emb_batch_size)

                batches = list(range(0, len(self.labels_str), self.emb_batch_size))
                if batches[-1] < len(self.labels_str):
                    batches.append(len(self.labels_str))
                for i in trange(len(batches) - 1):
                    hiddens, masks = self.compute_hiddens(self.input_seq[batches[i] : batches[i + 1]])
                    self.hiddens += list(hiddens.cpu().float())
                    self.emb_masks += list(masks.cpu().float())
                _reconfigure_inference_batch(
                    self.prot_model_batch_size, global_batch_size=self.model.cfg.model.global_batch_size
                )

    def compute_hiddens(self, seqs):
        if self.model is not None:
            hiddens, masks = self.model.seq_to_hiddens(seqs)
            return hiddens, masks
        else:
            return None

    def get_hidden_size(self):
        return get_from_encoder_or_model(self.model.cfg.model, "hidden_size")

    def get_embeddings(self, idx):
        if self.model is not None:
            return torch.squeeze(self.hiddens[idx][self.emb_masks[idx].bool()])
        else:
            return self.input_seq[idx]

    def get_labels(self, idx):
        return self.labels[idx]

    def length(self):
        return len(self.labels_str)

    @staticmethod
    def _tokenize_labels(tokenizers, labels_str, labels_sizes):
        labels = []
        one_hot = []
        for s in labels_sizes:
            one_hot.append(np.eye(s))
        for strings in labels_str:
            cur_labels = []
            for i in range(len(strings)):
                try:
                    ids = tokenizers[i].text_to_ids(strings[i])
                # TODO(trvachov): Understand except type
                except:  # noqa: E722
                    pass
                cur_one_hot = []
                for id_ in ids:
                    cur_one_hot.append(one_hot[i][id_])
                cur_labels.append(torch.tensor(np.array(cur_one_hot)))
            labels.append(cur_labels)
        return labels
