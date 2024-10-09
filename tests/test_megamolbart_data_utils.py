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


from typing import Any, Dict, Generator, List, Tuple

import pytest
import pytorch_lightning as pl
import torch
from nemo.collections.common.tokenizers.regex_tokenizer import RegExTokenizer
from nemo.core import Dataset
from omegaconf import DictConfig

from bionemo.data.molecule.augment import MoleculeEnumeration, MoleculeInputTargetEnumeration
from bionemo.data.molecule.megamolbart_utils import (
    megamolbart_build_train_valid_test_datasets,
    megamolbart_retro_build_train_valid_test_datasets,
)
from bionemo.model.utils import initialize_distributed_parallel_state
from bionemo.utils.hydra import load_model_config
from bionemo.utils.tests import teardown_apex_megatron_cuda


@pytest.fixture(scope="module")
def num_train_samples() -> int:
    return 5


@pytest.fixture(scope="module")
def tokenizer(bionemo_home) -> RegExTokenizer:
    tokenizer_path = bionemo_home / "tokenizers" / "molecule" / "megamolbart" / "vocab"
    model = str(tokenizer_path / "megamolbart.model")
    vocab = str(tokenizer_path / "megamolbart.vocab")
    tokenizer = RegExTokenizer().load_tokenizer(regex_file=model, vocab_file=vocab)
    return tokenizer


@pytest.fixture(scope="module")
def training_cfg(config_path_for_tests) -> DictConfig:
    cfg = load_model_config(config_name="megamolbart_test", config_path=config_path_for_tests)
    return cfg


@pytest.fixture(scope="module")
def training_retro_cfg(config_path_for_tests) -> DictConfig:
    cfg = load_model_config(config_name="megamolbart_downstream_retro_test", config_path=config_path_for_tests)
    return cfg


@pytest.fixture(scope="module")
def megamolbart_datasets(training_cfg, num_train_samples) -> Generator[Any, Any, Tuple[Dataset, Dataset, Dataset]]:
    initialize_distributed_parallel_state()
    train_ds, val_ds, test_ds = megamolbart_build_train_valid_test_datasets(
        cfg=training_cfg.model.data, train_n_samples=num_train_samples
    )
    yield train_ds, val_ds, test_ds
    teardown_apex_megatron_cuda()


@pytest.fixture(scope="module")
def megamolbart_retro_datasets(
    training_retro_cfg, num_train_samples
) -> Generator[Any, Any, Tuple[Dataset, Dataset, Dataset]]:
    initialize_distributed_parallel_state()
    train_ds, val_ds, test_ds = megamolbart_retro_build_train_valid_test_datasets(
        cfg=training_retro_cfg.model.data, train_n_samples=num_train_samples
    )
    yield train_ds, val_ds, test_ds
    teardown_apex_megatron_cuda()


@pytest.fixture(scope="module")
def batch() -> List[str]:
    batch = ["CO[C@H](C[C@@H]1CCC[C@H]1O)c1ccccc1", "COC[C@H](O)CCNC(=O)CCC[C@@H](C)N", "CC[C@](C)(Nc1cnnn1C)C1CC1"]
    return batch


@pytest.fixture(scope="module")
def batch_retro() -> List[Dict[str, str]]:
    batch = [
        {"products": "CCc1cc(N)c(N)cc1Cl", "reactants": "CCc1cc([N+](=O)[O-])c(N)cc1Cl"},
        {
            "products": "CC(C)(C)c1ccc(CN(CCc2ccc(Cl)c(C(F)(F)F)c2)C(=O)c2cc(Cl)cc3cc[nH]c23)cc1",
            "reactants": "CC(C)(C)c1ccc(CNCCc2ccc(Cl)c(C(F)(F)F)c2)cc1.O=C(O)c1cc(Cl)cc2cc[nH]c12",
        },
    ]
    return batch


@pytest.mark.needs_gpu
def test_megamolbart_build_train_valid_test_datasets(megamolbart_datasets, num_train_samples):
    train_ds, val_ds, test_ds = megamolbart_datasets

    assert len(train_ds) == num_train_samples

    assert train_ds[2] == "Cc1ccccc1CCON"
    assert val_ds[0] == "O=C[C@H](O)[C@H](O)[C@@H](O)[C@H](S)CO"
    assert test_ds[4] == "Cn1cnc(NC(=O)[C@@H](CN)CO)n1"


def test_megamolbart_retro_build_train_valid_test_datasets(
    training_retro_cfg, megamolbart_retro_datasets, num_train_samples
):
    train_ds, val_ds, test_ds = megamolbart_retro_datasets
    len(train_ds) == num_train_samples

    input_name = training_retro_cfg.model.data.input_name
    target_name = training_retro_cfg.model.data.target_name
    assert input_name in train_ds[0].keys() and target_name in train_ds[0].keys()

    assert train_ds[3] == {
        "products": "Cc1c(C)n(-c2cccnc2)c2ccc(Br)cc12",
        "reactants": "Cc1[nH]c2ccc(Br)cc2c1C.Fc1cccnc1",
    }

    assert val_ds[2] == {
        "products": "CC(=O)Nc1cccc(C(O)CN(C)Cc2sc3c(=O)c(C(=O)NCc4ccc(Cl)cc4)cn(C)c3c2C)c1",
        "reactants": "CNCC(O)c1cccc(NC(C)=O)c1.Cc1c(CCl)sc2c(=O)c(C(=O)NCc3ccc(Cl)cc3)cn(C)c12",
    }


def test_molecule_enumeration_collate_fn_no_mask(tokenizer, training_cfg, batch):
    pl.seed_everything(training_cfg.seed)
    seq_len = training_cfg.model.data.max_seq_length
    collate_fn = MoleculeEnumeration(
        tokenizer=tokenizer,
        seq_length=seq_len,
        pad_size_divisible_by_8=True,
        encoder_mask=False,
        decoder_mask=False,
        encoder_augment=False,
        decoder_independent_augment=False,
        canonicalize_target_smile=False,
        canonicalize_decoder_output=False,
        canonicalize_encoder_input=False,
    ).collate_fn

    output = collate_fn(batch)
    expected_keys = ["text_enc", "enc_mask", "text_dec", "dec_mask", "labels", "loss_mask", "target_smiles"]
    assert all(k in expected_keys for k in output.keys())
    assert len(output["target_smiles"]) == output["text_enc"].shape[0] and output["text_enc"].shape[0] == len(batch)
    assert output["text_enc"].shape[1] % 8 == 0

    assert not (tokenizer.mask_id in output["text_enc"] and tokenizer.mask_id in output["text_dec"])
    assert all(output["text_dec"][i][0] == tokenizer.bos_id for i in range(len(batch)))
    assert all(target_smi == smi for target_smi, smi in zip(output["target_smiles"], batch))

    for i in range(len(batch)):
        mask_dec = output["dec_mask"][i]
        mask_dec[0] = 0
        dec_ids = output["text_dec"][i][mask_dec == 1].cpu().detach().numpy().tolist()

        mask_enc = output["enc_mask"][i]
        enc_ids = output["text_enc"][i][mask_enc == 1].cpu().detach().numpy().tolist()

        assert tokenizer.ids_to_text([dec_ids])[0] == batch[i] and tokenizer.ids_to_text([enc_ids])[0] == batch[i]


def test_molecule_enumeration_collate_fn_mask(tokenizer, training_cfg, batch):
    pl.seed_everything(training_cfg.seed)
    seq_len = training_cfg.model.data.max_seq_length

    # double-checking masking
    collate_fn = MoleculeEnumeration(
        tokenizer=tokenizer,
        seq_length=seq_len,
        pad_size_divisible_by_8=True,
        encoder_mask=True,
        decoder_mask=True,
        encoder_augment=False,
        decoder_independent_augment=False,
        canonicalize_target_smile=False,
        canonicalize_decoder_output=False,
        canonicalize_encoder_input=False,
        mask_prob=0.1,
        span_lambda=3.0,
    ).collate_fn
    output = collate_fn(batch)
    assert (tokenizer.mask_id in output["text_enc"]) and (tokenizer.mask_id in output["text_dec"])

    assert torch.equal(
        output["text_enc"],
        torch.tensor(
            [
                [272, 4, 287, 275, 272, 290, 274, 4, 272, 4, 281, 273, 274, 273, 273, 273, 4, 273, 274, 0, 0, 0, 0, 0],
                [272, 285, 272, 287, 4, 279, 272, 275, 280, 285, 281, 272, 272, 272, 290, 4, 4, 0, 0, 0, 0, 0, 0, 0],
                [
                    272,
                    272,
                    291,
                    275,
                    272,
                    281,
                    275,
                    279,
                    273,
                    274,
                    273,
                    282,
                    282,
                    282,
                    274,
                    272,
                    281,
                    272,
                    274,
                    272,
                    272,
                    274,
                    0,
                    0,
                ],
            ]
        ),
    )

    assert torch.equal(
        output["enc_mask"],
        torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            ]
        ),
    )

    assert torch.equal(
        output["text_dec"],
        torch.tensor(
            [
                [
                    2,
                    4,
                    4,
                    287,
                    275,
                    272,
                    290,
                    274,
                    272,
                    272,
                    272,
                    287,
                    274,
                    285,
                    281,
                    273,
                    274,
                    273,
                    273,
                    273,
                    273,
                    273,
                    4,
                    0,
                ],
                [2, 4, 272, 287, 275, 4, 272, 279, 272, 4, 281, 272, 272, 272, 290, 275, 272, 281, 279, 0, 0, 0, 0, 0],
                [
                    2,
                    272,
                    272,
                    291,
                    275,
                    272,
                    281,
                    275,
                    279,
                    273,
                    274,
                    273,
                    282,
                    4,
                    281,
                    272,
                    274,
                    4,
                    274,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
            ]
        ),
    )

    assert torch.equal(
        output["dec_mask"],
        torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            ]
        ),
    )

    assert torch.equal(
        output["labels"],
        torch.tensor(
            [
                [
                    4,
                    4,
                    287,
                    275,
                    272,
                    290,
                    274,
                    272,
                    272,
                    272,
                    287,
                    274,
                    285,
                    281,
                    273,
                    274,
                    273,
                    273,
                    273,
                    273,
                    273,
                    4,
                    3,
                    -1,
                ],
                [
                    4,
                    272,
                    287,
                    275,
                    4,
                    272,
                    279,
                    272,
                    4,
                    281,
                    272,
                    272,
                    272,
                    290,
                    275,
                    272,
                    281,
                    279,
                    3,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                ],
                [
                    272,
                    272,
                    291,
                    275,
                    272,
                    281,
                    275,
                    279,
                    273,
                    274,
                    273,
                    282,
                    4,
                    281,
                    272,
                    274,
                    4,
                    274,
                    3,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                ],
            ]
        ),
    )

    assert torch.equal(
        output["loss_mask"],
        torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            ]
        ),
    )

    assert output["target_smiles"] == batch


@pytest.mark.parametrize(
    "encoder_augment, decoder_independent_augment, canonicalize_decoder_output, canonicalize_encoder_input, test_desc",
    [
        (False, False, False, False, "as_is"),  # leave as is with input
        (True, False, False, False, "equivariant"),  # augmented encoder, matched decoder
        (True, False, True, False, "invariant"),  # augmented encoder canonical decoder
        (False, False, True, True, "canonical"),  # canonical encoder canonical decoder
    ],
)
def test_molecule_enumeration_collate_function_options(
    tokenizer,
    training_cfg,
    batch,
    encoder_augment: bool,
    decoder_independent_augment: bool,
    canonicalize_decoder_output: bool,
    canonicalize_encoder_input: bool,
    test_desc: str,
):
    pl.seed_everything(training_cfg.seed)
    seq_len = training_cfg.model.data.max_seq_length
    collate_fn = MoleculeEnumeration(
        tokenizer=tokenizer,
        seq_length=seq_len,
        pad_size_divisible_by_8=True,
        encoder_mask=False,
        decoder_mask=False,
        encoder_augment=encoder_augment,
        decoder_independent_augment=decoder_independent_augment,
        canonicalize_target_smile=True,
        canonicalize_decoder_output=canonicalize_decoder_output,
        canonicalize_encoder_input=canonicalize_encoder_input,
    ).collate_fn

    output = collate_fn(batch)
    expected_keys = ["text_enc", "enc_mask", "text_dec", "dec_mask", "labels", "loss_mask", "target_smiles"]
    assert all(k in expected_keys for k in output.keys())
    assert len(output["target_smiles"]) == output["text_enc"].shape[0] and output["text_enc"].shape[0] == len(batch)
    assert output["text_enc"].shape[1] % 8 == 0

    assert not (tokenizer.mask_id in output["text_enc"] and tokenizer.mask_id in output["text_dec"])
    assert all(output["text_dec"][i][0] == tokenizer.bos_id for i in range(len(batch)))
    assert all(target_smi == smi for target_smi, smi in zip(output["target_smiles"], batch))

    for i in range(len(batch)):
        mask_dec = output["dec_mask"][i]
        mask_dec[0] = 0
        dec_ids = output["text_dec"][i][mask_dec == 1].cpu().detach().numpy().tolist()

        mask_enc = output["enc_mask"][i]
        enc_ids = output["text_enc"][i][mask_enc == 1].cpu().detach().numpy().tolist()
        if test_desc == "as_is" or test_desc == "canonical":
            # # batch is canonical so we can use this.
            assert tokenizer.ids_to_text([dec_ids])[0] == batch[i] and tokenizer.ids_to_text([enc_ids])[0] == batch[i]
        elif test_desc == "equivariant":
            assert tokenizer.ids_to_text([dec_ids])[0] == tokenizer.ids_to_text([enc_ids])[0]
            assert tokenizer.ids_to_text([dec_ids])[0] != batch[i]
        elif test_desc == "invariant":
            assert tokenizer.ids_to_text([dec_ids])[0] != tokenizer.ids_to_text([enc_ids])[0]
            assert tokenizer.ids_to_text([dec_ids])[0] == batch[i]  # batch is canonical so we can use this.
        else:
            assert False, test_desc + " is not a valid test description"

    assert output["target_smiles"] == batch

    if test_desc != "invariant":
        assert torch.equal(
            output["text_enc"][:, :-1], output["text_dec"][:, 1:]
        )  # other than BOS token, we should have equality
    else:
        assert not torch.equal(
            output["text_enc"][:, :-1], output["text_dec"][:, 1:]
        )  # other than BOS token, we should not have equality


def test_molecule_input_target_enumeration_collate_fn(tokenizer, training_retro_cfg, batch_retro):
    pl.seed_everything(training_retro_cfg.seed)
    seq_len = training_retro_cfg.model.data.max_seq_length
    input_name = training_retro_cfg.model.data.input_name
    target_name = training_retro_cfg.model.data.target_name

    collate_fn = MoleculeInputTargetEnumeration(
        tokenizer=tokenizer,
        seq_length=seq_len,
        pad_size_divisible_by_8=True,
        encoder_mask=False,
        decoder_mask=False,
        encoder_augment=False,
        decoder_independent_augment=False,
        canonicalize_encoder_input=False,
        canonicalize_decoder_output=False,
        canonicalize_target_smile=False,
        input_name=input_name,
        target_name=target_name,
    ).collate_fn
    output = collate_fn(batch_retro)
    expected_keys = ["text_enc", "enc_mask", "text_dec", "dec_mask", "labels", "loss_mask", "target_smiles"]
    assert all(k in expected_keys for k in output.keys())
    assert len(output["target_smiles"]) == output["text_enc"].shape[0] and output["text_enc"].shape[0] == len(
        batch_retro
    )
    assert output["text_enc"].shape[1] % 8 == 0

    assert (tokenizer.mask_id not in output["text_enc"]) and (tokenizer.mask_id not in output["text_dec"])
    assert all(output["text_dec"][i][0] == tokenizer.bos_id for i in range(len(batch_retro)))

    assert all(target_smi == smi["reactants"] for target_smi, smi in zip(output["target_smiles"], batch_retro))

    for i in range(len(batch_retro)):
        mask_dec = output["dec_mask"][i].clone()
        mask_dec[0] = 0
        dec_ids = output["text_dec"][i][mask_dec == 1].cpu().detach().numpy().tolist()

        mask_enc = output["enc_mask"][i]
        enc_ids = output["text_enc"][i][mask_enc == 1].cpu().detach().numpy().tolist()

        assert (
            tokenizer.ids_to_text([dec_ids])[0] == batch_retro[i]["reactants"]
            and tokenizer.ids_to_text([enc_ids])[0] == batch_retro[i]["products"]
        )

    assert torch.equal(
        output["text_enc"],
        torch.tensor(
            [
                [
                    272,
                    272,
                    273,
                    274,
                    273,
                    273,
                    275,
                    279,
                    281,
                    273,
                    275,
                    279,
                    281,
                    273,
                    273,
                    274,
                    304,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    272,
                    272,
                    275,
                    272,
                    281,
                    275,
                    272,
                    281,
                    273,
                    274,
                    273,
                    273,
                    273,
                    275,
                    272,
                    279,
                    275,
                    272,
                    272,
                    273,
                    277,
                    273,
                    273,
                    273,
                    275,
                    304,
                    281,
                    273,
                    275,
                    272,
                    275,
                    303,
                    281,
                    275,
                    303,
                    281,
                    303,
                    281,
                    273,
                    277,
                    281,
                    272,
                    275,
                    280,
                    285,
                    281,
                    273,
                    277,
                    273,
                    273,
                    275,
                    304,
                    281,
                    273,
                    273,
                    284,
                    273,
                    273,
                    286,
                    273,
                    277,
                    284,
                    281,
                    273,
                    273,
                    274,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
            ]
        ),
    )
    assert torch.equal(
        output["enc_mask"],
        torch.tensor(
            [
                [
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
            ]
        ),
    )

    assert torch.equal(
        output["text_dec"],
        torch.tensor(
            [
                [
                    2,
                    272,
                    272,
                    273,
                    274,
                    273,
                    273,
                    275,
                    309,
                    275,
                    280,
                    285,
                    281,
                    305,
                    281,
                    273,
                    275,
                    279,
                    281,
                    273,
                    273,
                    274,
                    304,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    2,
                    272,
                    272,
                    275,
                    272,
                    281,
                    275,
                    272,
                    281,
                    273,
                    274,
                    273,
                    273,
                    273,
                    275,
                    272,
                    279,
                    272,
                    272,
                    273,
                    277,
                    273,
                    273,
                    273,
                    275,
                    304,
                    281,
                    273,
                    275,
                    272,
                    275,
                    303,
                    281,
                    275,
                    303,
                    281,
                    303,
                    281,
                    273,
                    277,
                    281,
                    273,
                    273,
                    274,
                    299,
                    285,
                    280,
                    272,
                    275,
                    285,
                    281,
                    273,
                    274,
                    273,
                    273,
                    275,
                    304,
                    281,
                    273,
                    273,
                    277,
                    273,
                    273,
                    286,
                    273,
                    274,
                    277,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
            ]
        ),
    )

    assert torch.equal(
        output["dec_mask"],
        torch.tensor(
            [
                [
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
            ]
        ),
    )

    assert torch.equal(
        output["labels"],
        torch.tensor(
            [
                [
                    272,
                    272,
                    273,
                    274,
                    273,
                    273,
                    275,
                    309,
                    275,
                    280,
                    285,
                    281,
                    305,
                    281,
                    273,
                    275,
                    279,
                    281,
                    273,
                    273,
                    274,
                    304,
                    3,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                ],
                [
                    272,
                    272,
                    275,
                    272,
                    281,
                    275,
                    272,
                    281,
                    273,
                    274,
                    273,
                    273,
                    273,
                    275,
                    272,
                    279,
                    272,
                    272,
                    273,
                    277,
                    273,
                    273,
                    273,
                    275,
                    304,
                    281,
                    273,
                    275,
                    272,
                    275,
                    303,
                    281,
                    275,
                    303,
                    281,
                    303,
                    281,
                    273,
                    277,
                    281,
                    273,
                    273,
                    274,
                    299,
                    285,
                    280,
                    272,
                    275,
                    285,
                    281,
                    273,
                    274,
                    273,
                    273,
                    275,
                    304,
                    281,
                    273,
                    273,
                    277,
                    273,
                    273,
                    286,
                    273,
                    274,
                    277,
                    3,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                ],
            ]
        ),
    )

    assert torch.equal(
        output["loss_mask"],
        torch.tensor(
            [
                [
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
            ]
        ),
    )

    assert output["target_smiles"] == [b[target_name] for b in batch_retro]


def test_megamolbart_retro_collate_fn_masking_config(tokenizer, training_retro_cfg, batch_retro):
    pl.seed_everything(training_retro_cfg.seed)
    seq_len = training_retro_cfg.model.data.max_seq_length
    collate_fn = MoleculeInputTargetEnumeration(
        tokenizer=tokenizer,
        seq_length=seq_len,
        pad_size_divisible_by_8=True,
        **training_retro_cfg.model.data,
    ).collate_fn
    batch = [batch_retro[0]]

    output = collate_fn(batch)
    assert (tokenizer.mask_id not in output["text_enc"]) and (tokenizer.mask_id not in output["text_dec"])
