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
import pickle
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from bionemo.data.dsmbind.dataset import DSMBindDataset
from bionemo.data.dsmbind.preprocess import preprocess
from bionemo.utils.hydra import load_model_config

from .inference_shared_test_code import get_config_dir


def dsmbind_test_raw_data_dir(bionemo_home) -> str:
    path = bionemo_home / "examples" / "tests" / "test_data" / "molecule" / "dsmbind" / "inference"
    return str(path)


def test_dsmbind_data_preprocess(bionemo_home: Path):
    raw_data_dir = dsmbind_test_raw_data_dir(bionemo_home)
    preprocess(raw_data_dir=raw_data_dir)
    with open(os.path.join(raw_data_dir, "processed.pkl"), "rb") as f:
        processed_data = pickle.load(f)
    os.remove(os.path.join(raw_data_dir, "processed.pkl"))
    for data in processed_data:
        if data["pdb"] == "5cpr":
            assert (
                data["target_seq"]
                == "MSAKELCENDDLATSLVLDPYLGFQTHKMNRQEELKEVIERFKKDEHLEKAFKCLTSGEWARHYFLNKNKMQEKLFKEHVFIYLRMFATDSGFEILPCNRYSSEQNGAKIVATKEWKRNDKIELLVGCIAELSEIEENMLLRHGENDFSVMYSTRKNCAQLWLGPAAFINHDCRPNCKFVSTGRDTACVKALRDIEPGEEISCYYGDGFFGENNEFCECYTCERRGTGAFKS"
            )
            assert np.allclose(
                data["target_coords"][0],
                np.array(
                    [
                        [11.038, -10.959, 16.546],
                        [10.365, -9.857, 17.242],
                        [10.655, -9.821, 18.753],
                        [10.295, -10.757, 19.504],
                        [8.822, -9.844, 16.986],
                        [8.381, -8.921, 15.833],
                        [6.625, -8.416, 15.836],
                        [5.837, -9.901, 15.122],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ],
                    dtype=np.float32,
                ),
                atol=1.0e-3,
            )
        elif data["pdb"] == "4r0i":
            assert (
                data["target_seq"]
                == "VVGGTDADEGEWPWQVSLHALGQGHICGASLISPNWLVSAAHCYIDDRGFRYSDPTQWTAFLGLHDQSQRSAPGVQERRLKRIISHPFFNDFTFDYDIALLELEKPAEYSSMVRPICLPDASHVFPAGKAIWVTGWGHTQYGGTGALILQKGEIRVINQTTCENLLPQQITPRMMCVGFLSGGVDSCQGDSGGPLSSVEADGRIFQAGVVSWGDGCAQRNKPGVYTRLPLFRDWIKENTGVCGLR"
            )
            assert np.allclose(
                data["target_coords"][0],
                np.array(
                    [
                        [6.858, 11.616, 20.381],
                        [6.608, 12.884, 21.104],
                        [5.454, 12.6, 22.101],
                        [4.383, 12.094, 21.655],
                        [6.199, 13.994, 20.092],
                        [5.8, 15.274, 20.833],
                        [7.337, 14.296, 19.093],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ],
                    dtype=np.float32,
                ),
                atol=1.0e-3,
            )


def test_dsmbind_dataset(bionemo_home: Path):
    config_path = get_config_dir(bionemo_home, "dsmbind")
    cfg = load_model_config(config_name="pretrain", config_path=config_path)

    # process the raw data for testing
    raw_data_dir = dsmbind_test_raw_data_dir(bionemo_home)
    preprocess(raw_data_dir=raw_data_dir)

    dataset = DSMBindDataset(
        processed_data_path=os.path.join(raw_data_dir, "processed.pkl"),
        aa_size=cfg.model.aa_size,
        max_residue_atoms=cfg.model.max_residue_atoms,
    )
    os.remove(os.path.join(raw_data_dir, "processed.pkl"))
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=dataset.pl_collate_fn, shuffle=False)
    data_iterator = iter(dataloader)
    batched_binder, batched_target = next(
        data_iterator
    )  # first batch, where the maximum number of ligand atoms is 25 and the number of residues to be considered as in pocket is 50.
    assert batched_binder[0].shape == (2, 37, cfg.model.max_residue_atoms, 3)
    assert len(batched_binder[1]) == 2
    assert batched_binder[2].shape == (2, 37, cfg.model.max_residue_atoms)
    assert batched_target[0].shape == (2, 50, cfg.model.max_residue_atoms, 3)
    assert batched_target[1].shape == (2, 50, cfg.model.aa_size)
    assert batched_target[2].shape == (2, 50, cfg.model.max_residue_atoms)
