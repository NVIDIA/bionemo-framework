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


import pytest

from bionemo.data.preprocess.protein.preprocess import ESM2Preprocess
from bionemo.model.protein.esm1nv import esm1nv_model
from bionemo.model.utils import setup_trainer


# TODO setup test data
@pytest.mark.skip
def test_uf90_resampling_dataset_integration():
    """
    This test requires:

    cluster_mapping_tsv: $BIONEMO_HOME/data/uniref2022_05/fake-uniref/mapping.tsv
    uf50_datapath: $BIONEMO_HOME/data/uniref2022_05/fake-uniref/Fake50.fasta
    uf90_datapath: $BIONEMO_HOME/data/uniref2022_05/fake-uniref/Fake90.fasta
    dataset_path: $BIONEMO_HOME/data/uniref2022_05/fake_uf_map

    you can generate Fake50.fasta, Fake90.fasta, and mapping.tsv by running this script with a python interpreter instead of pytest:
        python tests/_test_esm2_uf90clustermapping_dataset.py


    files are deposited into the current working directory, move them to $BIONEMO_HOME/data/uniref2022_05 before preprocessing.
    """

    # TODO (SKH): automate data generation
    from omegaconf import OmegaConf

    cfg = OmegaConf.load("conf/uf90-resampling.yaml")
    trainer = setup_trainer(cfg, callbacks=[])

    # Required for the dataloaders
    from bionemo.model.utils import initialize_distributed_parallel_state

    initialize_distributed_parallel_state(
        local_rank=0,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        pipeline_model_parallel_split_rank=0,
    )

    # Preprocess the testing dataset
    preprocessor = ESM2Preprocess()
    preprocessor.prepare_dataset(
        uf50_datapath=cfg.model.data.uf50_datapath,
        uf90_datapath=cfg.model.data.uf90_datapath,
        cluster_mapping_tsv=cfg.model.data.cluster_mapping_tsv,
        uf50_output_dir=cfg.model.data.dataset_path,
        uf90_output_dir=cfg.model.data.uf90.uniref90_path,
        val_size=cfg.model.data.val_size,
        test_size=cfg.model.data.test_size,
        force=False,
    )

    # Then test that it does what we think it should do.
    train_ds, val_ds, test_ds = esm1nv_model.ESM2nvModel._build_train_valid_test_datasets(
        trainer, cfg.model, keep_uf50=True
    )
    for ds in [train_ds, val_ds, test_ds]:
        for uf50, uf90 in zip(ds.uniref50_dataset, ds):
            # Compares the IDs, where we manually constructed them to be nearly-equal.
            assert uf50.split("UniRef50_")[1] in uf90["sequence_id"].split("UniRef90_")[1]


def make_data():
    alphabet = "ARNDCQEGHILKMFPSTWYV"
    cluster_mapping = {}
    with open("Fake50.fasta", "w") as fd:
        for length in range(50, 1000, 50):
            for base in alphabet:
                fd.write(f">UniRef50_{base}_{length}\n")
                fd.write(base * length)
                fd.write("\n")
                cluster_mapping[f"UniRef50_{base}_{length}"] = []

    with open("Fake90.fasta", "w") as fd:
        num_clusters = 10
        for length in range(50, 1000, 50):
            for base in alphabet:
                for i in range(num_clusters):
                    fd.write(f">UniRef90_{base}_{length}-{i}\n")
                    fd.write(base * (length + i))
                    fd.write("\n")
                    cluster_mapping[f"UniRef50_{base}_{length}"].append(f"UniRef90_{base}_{length}-{i}")
                num_clusters += 1

    with open("mapping.tsv", "w") as fd:
        fd.write("dumbheader\n")
        for key, values in cluster_mapping.items():
            str_ = key + "\t" + ",".join(values)
            fd.write(f"{str_}\n")


if __name__ == "__main__":
    make_data()
