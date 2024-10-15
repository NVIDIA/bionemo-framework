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

import argparse
import contextlib
import functools
import sys
from pathlib import Path

import torch
from megatron.core.transformer.module import Float16Module
from torch.utils.data import DataLoader
from torchmetrics.text import Perplexity
from tqdm import tqdm

from bionemo.core.utils.dtypes import get_autocast_dtype
from bionemo.esm2.api import ESM2Config
from bionemo.esm2.data import dataset
from bionemo.esm2.data.tokenizer import get_tokenizer
from bionemo.esm2.model.model import ESM2Model
from bionemo.llm.data import collate
from bionemo.testing import megatron_parallel_state_utils


def _compute_preplexity(model, dataloader, vocab_size=None, limit_batches: int | None = None):
    """Calculate perplexity over an entire dataloader using whatever MLM settings are in it.

    Args:
        model: model to apply to data loader
        dataloader: dataloader to iterate over
        vocab_size: vocab size to limit computation to. Useful when there are unused padded tokens.
        limit_batches: for testing, you can limit to some number of batches. Defaults to None.

    Returns:
        global perplexity defined as exponentiation of the mean per-token cross entropy.
    """
    # We use the mask, so having the correct ignore_index isn't as important, as long as it is not a real
    #  index.
    perplexity = Perplexity(ignore_index=-100).cuda()
    for i, batch in enumerate(tqdm(dataloader)):
        assert isinstance(batch, dict)
        result = model(input_ids=batch["text"].cuda(), attention_mask=batch["attention_mask"].cuda())

        # bionemo ESM2 vocab_size
        if vocab_size is not None:
            logits = result["token_logits"][..., :vocab_size]
        else:
            logits = result.logits

        loss_mask = batch["loss_mask"].cuda()
        target = batch["labels"].cuda()
        logits_masked = logits[loss_mask].float()
        target_masked = target[loss_mask]
        # Perplexity calc wants batch, seq, vocab, but we're masking which flattens, so instead add
        #  back dummy batch dim at 0
        perplexity.update(preds=logits_masked.unsqueeze(0), target=target_masked.unsqueeze(0))
        if limit_batches is not None and i + 1 >= limit_batches:
            break
    mean_perplexity = perplexity.compute()  # convert global mean loss to perplexity.
    return mean_perplexity


if __name__ == "__main__":
    # TODO migrate to hydra config
    # Parse the arguments and pull them out into local variables for ease of future refactor to a
    #   config management system.
    parser = argparse.ArgumentParser(description="Calculate perplexity on an ESM2 style dataset on a single GPU.")
    parser.add_argument("--restore-from-checkpoint-path", type=Path, help="Path to checkpoint to load", required=True)
    parser.add_argument("--cluster-path", type=Path, help="Cluster path", required=True)
    parser.add_argument("--database-path", type=Path, help="Database path", required=True)
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--limit-batches", type=int, help="Limit batches for testing this function.")
    args = parser.parse_args()

    limit_batches: int | None = args.limit_batches
    cluster_path: Path = args.cluster_path
    database_path: Path = args.database_path
    restore_from_checkpoint_path: Path = args.restore_from_checkpoint_path
    batch_size: int = args.batch_size
    # This is the strategy used by HF
    random_mask_strategy: dataset.RandomMaskStrategy = dataset.RandomMaskStrategy.ALL_TOKENS

    max_seq_length: int = 1024
    # min = max seems faster. Perhaps due to kernel recompilation when the seq length changes.
    min_seq_length: int | None = max_seq_length
    seed: int = args.seed
    precision: str = "bf16-mixed"

    esm2_config = ESM2Config(
        seq_length=max_seq_length,
        params_dtype=get_autocast_dtype(precision),
        pipeline_dtype=get_autocast_dtype(precision),
        autocast_dtype=get_autocast_dtype(precision),
        initial_ckpt_path=str(restore_from_checkpoint_path),
        variable_seq_lengths=True,
    )
    mask_prob: float = 0.15
    mask_token_prob: float = 0.8
    mask_random_prob: float = 0.1

    tokenizer = get_tokenizer()

    # Create validation dataset
    clusters = dataset.create_valid_clusters(cluster_path)

    ds = dataset.create_valid_dataset(
        clusters=cluster_path,
        db_path=database_path,
        total_samples=None,  # None shoudl be once through.
        seed=seed,
        max_seq_length=max_seq_length,
        mask_prob=mask_prob,
        mask_token_prob=mask_token_prob,
        mask_random_prob=mask_random_prob,
        random_mask_strategy=random_mask_strategy,
        tokenizer=tokenizer,
    )
    dl = DataLoader(
        ds,
        shuffle=False,
        batch_size=batch_size,
        drop_last=False,
        collate_fn=functools.partial(
            collate.bert_padding_collate_fn,
            padding_value=tokenizer.pad_token_id,
            min_length=min_seq_length,
            max_length=max_seq_length,
        ),
    )
    vocab_size: int = tokenizer.vocab_size
    with contextlib.redirect_stdout(sys.stderr):
        with (
            torch.inference_mode(),
            megatron_parallel_state_utils.distributed_model_parallel_state(seed),
        ):
            # Need to initialize in megatron context
            model_pre: ESM2Model = esm2_config.configure_model(tokenizer).eval().cuda()
            model = Float16Module(model_pre.config, model_pre)

            perplexity = _compute_preplexity(model, dl, vocab_size, limit_batches)
    print(f"Perplexity: {perplexity}")
