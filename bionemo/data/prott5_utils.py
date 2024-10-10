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


from nemo.collections.nlp.data.language_modeling.megatron.blendable_dataset import BlendableDataset
from nemo.collections.nlp.data.language_modeling.megatron.dataset_utils import get_indexed_dataset_
from nemo.utils import logging
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

from bionemo.data.utils import expand_dataset_paths


def _build_dataset(
    cfg,
    trainer,
    data_prefix,
    data_impl,
    num_samples,
    max_seq_length,
    masked_lm_prob,
    short_seq_prob,
    seed,
    skip_warmup,
    max_seq_length_dec,
    name,
    dataset_type="t5",
    tokenizer=None,
    max_ngram_size=1,
    mean_ngram_size=None,
    geometric_dist=True,
    permutation=False,
    whole_word_masking=True,
    favor_long_ngrams=False,
    data_impl_kwargs={},
):
    if dataset_type != "t5":
        raise ValueError("Invalid dataset_type: ", dataset_type)

    min_seq_length_dec = round(max_seq_length * masked_lm_prob * 2 + 1)

    if min_seq_length_dec > max_seq_length_dec:
        raise ValueError(
            f"Cannot have seq_length_dec ({max_seq_length_dec}) less than seq_length * masked_lm_prob * 2 + 1 ({min_seq_length_dec}). Increase seq_length_dec or decrease masked_lm_prob"
        )

    # Indexed dataset.
    indexed_dataset = get_indexed_dataset_(data_prefix, data_impl, skip_warmup, data_impl_kwargs=data_impl_kwargs)
    total_num_of_documents = indexed_dataset.doc_idx.shape[0] - 1
    # Checks.
    assert indexed_dataset.doc_idx[0] == 0
    assert indexed_dataset.doc_idx.shape[0] == (total_num_of_documents + 1)

    def build_t5_dataset(name):
        from nemo.collections.nlp.data.language_modeling.megatron.t5_dataset import T5Dataset

        dataset = None
        kwargs = {
            "name": name,
            "data_prefix": data_prefix,
            "num_epochs": None,
            "max_num_samples": int(num_samples),
            "max_seq_length": max_seq_length,
            "seed": seed,
        }
        if dataset_type == "t5":
            assert tokenizer is not None, "Tokenizer is required for T5 dataset"
            logging.info("Instatiating T5 Dataset ...")
            dataset = T5Dataset(
                cfg=cfg,
                trainer=trainer,
                tokenizer=tokenizer,
                indexed_dataset=indexed_dataset,
                masked_lm_prob=masked_lm_prob,
                max_seq_length_dec=max_seq_length_dec,
                short_seq_prob=short_seq_prob,
                max_ngram_size=max_ngram_size,
                mean_ngram_size=mean_ngram_size,
                geometric_dist=geometric_dist,
                permutation=permutation,
                whole_word_masking=whole_word_masking,
                favor_long_ngrams=favor_long_ngrams,
                **kwargs,
            )
        else:
            raise NotImplementedError("Dataset type must be t5.")

        return dataset

    dataset = build_t5_dataset(name)

    return dataset


def prott5_build_dataset(
    cfg,
    trainer,
    tokenizer,
    data_prefix,
    data_impl,
    num_samples,
    max_seq_length,
    max_seq_length_dec,
    masked_lm_prob,
    short_seq_prob,
    seed,
    skip_warmup,
    name,
    dataset_type,
    max_ngram_size,
    mean_ngram_size,
    geometric_dist,
    permutation,
    whole_word_masking,
    favor_long_ngrams,
    data_impl_kwargs,
):
    # do not load a dataset when num_samples is 0
    if num_samples == 0:
        return None

    if data_impl in ["text_mmap", "csv_mmap"]:
        if "tokenizer" not in data_impl_kwargs:
            if isinstance(data_impl_kwargs, DictConfig):
                data_impl_kwargs = OmegaConf.to_object(data_impl_kwargs)

            data_impl_kwargs["tokenizer"] = tokenizer

    # only for csv_mmap we enforce the extension to be .csv (for backward compatibility)
    if data_impl == "csv_mmap":
        ext = ".csv"
    else:
        ext = ""

    fnames = expand_dataset_paths(data_prefix, ext)
    weights = [1.0 / len(fnames)] * len(fnames)

    # Build individual datasets
    datasets = []
    for i in range(len(fnames)):
        ds = _build_dataset(
            cfg=cfg,
            trainer=trainer,
            data_prefix=fnames[i],
            data_impl=data_impl,
            num_samples=num_samples,
            max_seq_length=max_seq_length,
            masked_lm_prob=masked_lm_prob,
            short_seq_prob=short_seq_prob,
            seed=seed,
            skip_warmup=skip_warmup,
            max_seq_length_dec=max_seq_length_dec,
            name=name,
            dataset_type=dataset_type,
            tokenizer=tokenizer,
            max_ngram_size=max_ngram_size,
            mean_ngram_size=mean_ngram_size,
            geometric_dist=geometric_dist,
            permutation=permutation,
            whole_word_masking=whole_word_masking,
            favor_long_ngrams=favor_long_ngrams,
            data_impl_kwargs=data_impl_kwargs,
        )
        if ds:
            datasets.append(ds)

    # Blend.
    blending_dataset = None
    if datasets:
        blending_dataset = BlendableDataset(datasets, weights, num_samples)

    return blending_dataset
