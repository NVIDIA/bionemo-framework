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


import glob
import os
import pickle
import tarfile

from bionemo.data.diffdock.webdataset_utils import pickles_to_tars


BIONEMO_HOME = os.environ.get("BIONEMO_HOME", "/workspace/bionemo")
SOURCE_DATA = os.path.join(
    BIONEMO_HOME,
    "examples/tests/test_data/molecule/diffdock/data_cache/",
    "torsion_limit0_maxLigSizeNone_H0_recRad15_recMax24_esmEmbeddings_INDEXsplit_train",
)
REF_TARFILE_SIZE = [27801600, 27463680, 27166720, 25139200]


def test_pickles_to_wds_tars(tmp_path):
    pickle_file_path = os.path.join(tmp_path, "pickle_files")
    os.makedirs(pickle_file_path)

    for file in glob.glob(os.path.join(SOURCE_DATA, "heterographs-*.tar")):
        tar = tarfile.open(file)
        tar.extractall(path=pickle_file_path)

    complex_names = [file.split(".")[0] for file in sorted(os.listdir(pickle_file_path))]
    assert len(complex_names) == 40

    output_path = os.path.join(tmp_path, "webdataset_tarfiles")
    os.makedirs(output_path)

    pickles_to_tars(
        pickle_file_path,
        "HeteroData.pyd",
        complex_names,
        output_path,
        "heterographs",
        lambda complex_graph: {"__key__": complex_graph.name, "HeteroData.pyd": pickle.dumps(complex_graph)},
        4,
    )
    assert len(os.listdir(output_path)) == 4

    for idx, file in enumerate(sorted(glob.glob(os.path.join(output_path, "heterographs-*.tar")))):
        assert os.stat(file).st_size == REF_TARFILE_SIZE[idx]
