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


import logging
from copy import deepcopy
from pathlib import Path
from typing import Generator, List

import pytest
from omegaconf import open_dict

from bionemo.model.singlecell.geneformer.infer import GeneformerInference
from bionemo.utils.hydra import load_model_config
from bionemo.utils.tests import (
    distributed_model_parallel_state,
)

from .inference_shared_test_code import (
    get_config_dir,
    get_expected_vals_file,
    run_seqs_to_embedding,
    run_seqs_to_hiddens_with_goldens,
)


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


CELLS_FOR_TEST = [
    [
        "ENSG00000288623",
        "ENSG00000288658",
        "ENSG00000288681",
        "ENSG00000288698",
        "ENSGR0000002586",
        "ENSGR0000124333",
        "ENSGR0000124334",
        "ENSGR0000167393",
        "ENSGR0000168939",
        "ENSGR0000169084",
    ],
    [
        "ENSG00000259900",
        "ENSG00000259916",
        "ENSG00000259956",
        "ENSG00000259958",
        "ENSG00000259991",
        "ENSG00000260001",
        "ENSG00000260007",
        "ENSG00000260027",
        "ENSG00000260040",
        "ENSG00000260045",
        "ENSG00000260092",
        "ENSG00000260099",
        "ENSG00000260119",
    ],
    [
        "ENSG00000269743",
        "ENSG00000269746",
        "ENSG00000269748",
        "ENSG00000269753",
        "ENSG00000269754",
        "ENSG00000269755",
        "ENSG00000269759",
        "ENSG00000269766",
        "ENSG00000269773",
        "ENSG00000269781",
        "ENSG00000269782",
        "ENSG00000269783",
        "ENSG00000269790",
        "ENSG00000269791",
        "ENSG00000269795",
    ],
]


@pytest.fixture()
def cells() -> List[List[str]]:
    return deepcopy(CELLS_FOR_TEST)


@pytest.fixture(scope="module")
def geneformer_inferer(bionemo_home: Path) -> Generator[GeneformerInference, None, None]:
    model_name = "geneformer"
    cfg_path = get_config_dir(bionemo_home, model_name)
    cfg = load_model_config(config_name="infer", config_path=cfg_path)
    with open_dict(cfg):
        # Override where the loaded inference object will look for training data for the inference.model object, which needs train/val/test dataloaders.
        base_dir_processed_files = (
            bionemo_home
            / "examples"
            / "tests"
            / "test_data"
            / "singlecell"
            / "cellxgene_2023-12-15_small"
            / "processed_data"
        )
        # TODO make it so that we do not need to load the datamodule into the Geneformer model when loaded by the inference object.
        #  as it is now, whenver a user loads an inference object they need to remember to set the train/val/test paths in the config to
        #  valid locations on disk. This is not ideal for usability.
        base_train_dir = base_dir_processed_files / "train"
        base_val_dir = base_dir_processed_files / "val"
        base_test_dir = base_dir_processed_files / "test"

        cfg.model.data.train_dataset_path = base_train_dir
        cfg.model.data.val_dataset_path = base_val_dir
        cfg.model.data.test_dataset_path = base_test_dir

    with distributed_model_parallel_state():
        inferer = GeneformerInference(cfg=cfg, inference_batch_size_for_warmup=2)  # Change to 1 to debug the failure
        yield inferer  # Yield so cleanup happens after the test


@pytest.fixture(scope="module")
def geneformer_expected_vals_path(bionemo_home: Path) -> Path:
    return get_expected_vals_file(bionemo_home, "geneformer")


def test_inferer_error_bad_token(geneformer_inferer: GeneformerInference):
    with pytest.raises(ValueError, match="Unknown token"):
        geneformer_inferer.seq_to_hiddens([["BAD_TOKEN"]])


def test_geneformer_inference_tokenization(geneformer_inferer: GeneformerInference, cells: List[List[str]]):
    cell_tokenization, genes_mask = geneformer_inferer.tokenize(cells)
    assert cell_tokenization.shape == (
        3,
        max(len(cell) for cell in cells) + 1,
    ), "Each gene should be 1 token plus one token for [CLS]"
    cls_token_id = geneformer_inferer.tokenizer.token_to_id(geneformer_inferer.tokenizer.cls_token)
    n_tokens = genes_mask.sum(-1)
    assert all([n_tokens[i] == len(cells[i]) + 1] for i in range(len(cells)))
    assert all(cell_tokenization[:, 0] == cls_token_id), "First column should be all [CLS] tokens."


@pytest.mark.needs_fork
@pytest.mark.needs_gpu
def test_cells_to_hiddens_with_goldens_geneformer(
    geneformer_inferer: GeneformerInference, cells: List[List[str]], geneformer_expected_vals_path: Path
):
    run_seqs_to_hiddens_with_goldens(
        geneformer_inferer,
        cells,
        geneformer_expected_vals_path,
        geneformer_inferer.model.cfg.hidden_size,
        "bert",
        geneformer_inferer._tokenize,
    )


@pytest.mark.needs_fork
@pytest.mark.needs_gpu
def test_cells_to_embedding_geneformer(geneformer_inferer: GeneformerInference, cells: List[List[str]]):
    run_seqs_to_embedding(geneformer_inferer, cells, geneformer_inferer.model.cfg.hidden_size)
