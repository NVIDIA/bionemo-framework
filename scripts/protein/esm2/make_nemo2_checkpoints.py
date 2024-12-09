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
import tempfile
from pathlib import Path

from bionemo.esm2.model.model import ESM2NeMo1LightningModuleConnector
from bionemo.testing.data.load import _s3_download


if __name__ == "__main__":
    """Usage:
    # ESM2 3B
    ## ESM2 3b checkpoint conversion:
    python scripts/protein/esm2/make_nemo2_checkpoints.py --s3-path s3://bionemo-ci/models/esm2nv_3B_converted.nemo --output-path ~/.cache/bionemo/checkpoints/esm2_3B_nemo2
    ## ESM2 3b checkpoint upload (recursive since it is a directory)
    aws s3 cp --recursive ~/.cache/bionemo/checkpoints/esm2_3B_nemo2 s3://bionemo-ci/models/esm2_3B_nemo2
    # ESM2 650M
    ## ESM2 650M checkpoint conversion
    python scripts/protein/esm2/make_nemo2_checkpoints.py --s3-path s3://bionemo-ci/models/esm2nv_650M_converted.nemo --output-path ~/.cache/bionemo/checkpoints/esm2_650M_nemo2
    ## ESM2 650M checkpoint upload
    aws s3 cp --recursive ~/.cache/bionemo/checkpoints/esm2_650M_nemo2 s3://bionemo-ci/models/esm2_650M_nemo2
    """
    parser = argparse.ArgumentParser(description="Download and convert an S3 nemo1 checkpoint into a nemo2 checkpoint")

    # s3-path as a string
    parser.add_argument(
        "--s3-path",
        type=str,
        required=True,
        help="S3 path as a string. See https://github.com/NVIDIA/bionemo-fw-ea/blob/bionemo1/artifact_paths.yaml",
    )

    # output-path as a pathlib.Path
    parser.add_argument("--output-path", type=Path, required=True, help="Output path as a pathlib.Path")
    args = parser.parse_args()
    output_path = args.output_path
    assert not output_path.exists(), "Please specify an output path that does not exist."
    with tempfile.TemporaryDirectory() as temp_dir:
        fname = args.s3_path.split("/")[-1]
        nemo1_ckpt_path = Path(temp_dir) / fname
        _s3_download(args.s3_path, nemo1_ckpt_path, None)
        connector = ESM2NeMo1LightningModuleConnector(nemo1_ckpt_path)
        connector.apply(output_path)
    print(output_path)
