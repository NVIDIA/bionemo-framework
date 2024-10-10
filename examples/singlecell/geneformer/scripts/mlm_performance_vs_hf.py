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
import pickle
from copy import deepcopy
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.distributed
import torch.utils
import torch.utils.data
from omegaconf import open_dict
from torch.utils.data import DataLoader
from tqdm import trange
from transformers import AutoModelForMaskedLM

from bionemo.data.singlecell.dataset import SingleCellDataset
from bionemo.model.singlecell.geneformer.infer import GeneformerInference
from bionemo.tokenizer.gene_tokenizer import GeneTokenizer
from bionemo.utils.hydra import load_model_config


class GeneformerHFAdapter(torch.nn.Module):
    def __init__(self, hf_path: str, my_token_dict: Dict[str, int], nv_tokenizer: GeneTokenizer):
        super().__init__()
        self.model = AutoModelForMaskedLM.from_pretrained(hf_path)
        self.my_token_dict = deepcopy(my_token_dict)
        self.nv_tokenizer = deepcopy(nv_tokenizer)
        self.n_tokens_nv = len(self.nv_tokenizer.vocab)
        self.n_tokens_hf = len(my_token_dict)

        # nvidia tokenizer has [cls] and [pad] first along with some others that do not overlap. This mapper
        hf_ordered_nv_tokenizer = {
            self.nv_tokenizer.pad_token: my_token_dict["<pad>"],
            self.nv_tokenizer.mask_token: my_token_dict["<mask>"],
        }

        missing_nv_tokens = []
        for ens, idx in list(my_token_dict.items())[2:]:
            if ens in nv_tokenizer.vocab.keys():
                hf_ordered_nv_tokenizer[ens] = idx
            else:
                missing_nv_tokens.append(idx)
        self.hf_ordered_nv_tokenizer = hf_ordered_nv_tokenizer
        self.register_buffer("missing_nv_tokens", torch.tensor(missing_nv_tokens, dtype=int))

    @property
    def device(self) -> torch.device:
        return self.missing_nv_tokens.device

    def get_tokenizer(self) -> GeneTokenizer:
        nv_tok = deepcopy(self.nv_tokenizer)
        nv_tok.vocab = self.hf_ordered_nv_tokenizer
        nv_tok.decode_vocab = dict(zip(nv_tok.vocab.values(), nv_tok.vocab.keys()))
        return nv_tok

    def forward(self, *args, **kwargs):
        logits = self.model(*args, **kwargs).logits
        logits[:, :, self.missing_nv_tokens] = -torch.inf
        return logits


class OneBatchDataset(torch.utils.data.IterableDataset):
    def __init__(self, batch):
        super().__init__()
        self.batch = batch

    def __iter__(self):
        yield self.batch


# Custom collate function that returns the batch as-is
def custom_collate(batch):
    # Since the dataset yields a single batch which is a dictionary of tensors,
    # and the DataLoader will wrap this in a list, we just return the first element.
    return batch[0]


class OneBatchDataLoader(DataLoader):
    def __init__(self, batch, *args, dataset=None, collate_fn=None, **kwargs):
        self.batch = batch  # Storing the batch for reference
        dataset = OneBatchDataset(batch)
        # Ensure that the custom collate function is used, specifically for dict[str, tensor]
        super().__init__(dataset, collate_fn=custom_collate, *args, **kwargs)


def nested_getattr(obj, attr):
    for a in attr.split("."):
        obj = getattr(obj, a)
    return obj


def main(
    model_path: Path, hf_model_path: str, dataset_path: Path, token_dictionary_path: Path, mask_prob: float = 0.15
):
    with open(token_dictionary_path, "rb") as geneformer_hf_token_file:
        geneformer_hf_token_dict = pickle.load(geneformer_hf_token_file)
    cfg = load_model_config("infer.yaml", config_path="/workspace/bionemo/examples/singlecell/geneformer/conf")
    with open_dict(cfg):
        cfg.model.downstream_task.restore_from_path = model_path
        cfg.exp_manager.exp_dir = dataset_path
        cfg.model.megatron_amp_O2 = False
        cfg.trainer.precision = "bf16-mixed"
        cfg.model.data.num_workers = 0
        cfg.model.precision = "bf16-mixed"
        # cfg.model.virtual_pipeline_model_parallel_size=None
        cfg.model.data.batch_size = 4
        cfg.model.post_process = True  # Set to True so we get Logits out for the following test
    # with distributed_model_parallel_state():
    # model = GeneformerModel.restore_from(model_path).cuda(0)
    geneformer_nv_inferer = GeneformerInference(cfg=cfg, inference_batch_size_for_warmup=2).cuda(0)
    # with tarfile.open(model_path, 'r') as tar:
    #     weights = torch.load(tar.extractfile("./model_weights.ckpt"))
    # for name, tensor in weights.items():
    #     torch.testing.assert_close(nested_getattr(geneformer_nv_inferer.model, name), tensor)

    ds_nv = SingleCellDataset(
        dataset_path,
        geneformer_nv_inferer.model.tokenizer,
        geneformer_nv_inferer.model.median_dict,
        max_len=cfg.model.seq_length,
        mask_prob=mask_prob,
    )
    hf_model = GeneformerHFAdapter(hf_model_path, geneformer_hf_token_dict, geneformer_nv_inferer.tokenizer).cuda(1)
    tokenizer_hf_to_nv = hf_model.get_tokenizer()
    tokenizer_shared = deepcopy(geneformer_nv_inferer.model.tokenizer)

    set(tokenizer_hf_to_nv.vocab.keys())
    tokenizer_shared.vocab = dict(tokenizer_shared.vocab.items())
    tokenizer_shared.decode_vocab = dict(zip(tokenizer_shared.vocab.values(), tokenizer_shared.vocab.keys()))

    ds_hf_nvfilt = SingleCellDataset(
        dataset_path,
        hf_model.get_tokenizer(),
        geneformer_nv_inferer.model.median_dict,
        max_len=cfg.model.seq_length,
        mask_prob=mask_prob,
        prepend_cls_token=False,
    )
    print(f"Loaded dataset of length (NV): {len(ds_nv)}, (HF): {len(ds_hf_nvfilt)}")

    dl_hf = DataLoader(ds_hf_nvfilt, batch_size=8, shuffle=False, num_workers=0, drop_last=False)
    dl_nv = DataLoader(ds_nv, batch_size=8, shuffle=False, num_workers=0, drop_last=False)

    with torch.no_grad():
        dl_hf_iter = iter(dl_hf)
        dl_nv_iter = iter(dl_nv)
        losses_hf = []
        losses_nv = []
        for b_idx in trange(len(dl_hf)):
            np.random.seed(b_idx)
            batch_hf = {k: v.to(hf_model.device) for k, v in next(dl_hf_iter).items()}
            np.random.seed(b_idx)
            batch_nv = {k: v.to(geneformer_nv_inferer.device) for k, v in next(dl_nv_iter).items()}
            logits_hf = hf_model(batch_hf["text"], batch_hf["padding_mask"])
            loss_hf = torch.nn.functional.cross_entropy(
                logits_hf[batch_hf["loss_mask"]], batch_hf["labels"][batch_hf["loss_mask"]]
            ).cpu()

            loss_nv = geneformer_nv_inferer(batch_nv)["loss"].cpu()
            losses_hf.append(loss_hf)
            losses_nv.append(loss_nv)
    print(f"HF mean loss: {np.sum(losses_hf) / len(losses_hf)}, NV mean loss: {np.sum(losses_nv) / len(losses_nv)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLM Performance vs HF Script")
    parser.add_argument("--model-path", type=Path, help="Nvidia model path to .nemo file", required=True)
    parser.add_argument(
        "--token-dictionary-path",
        type=Path,
        help="Path to token dictionary file. "
        "Eg `wget https://huggingface.co/ctheodoris/Geneformer/resolve/main/geneformer/token_dictionary.pkl` "
        "then provide the path to the downloaded file.",
        required=True,
    )
    parser.add_argument("--hf-model-path", type=str, default="ctheodoris/Geneformer", help="HF model path")
    parser.add_argument("--dataset-path", type=Path, help="Path to dataset directory", required=True)

    args = parser.parse_args()
    main(args.model_path, args.hf_model_path, args.dataset_path, args.token_dictionary_path)
