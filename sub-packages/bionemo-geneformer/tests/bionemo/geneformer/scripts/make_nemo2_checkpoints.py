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

from bionemo.core.data.load import _s3_download
from bionemo.geneformer.api import GeneformerNeMo1LightningModuleConnector


if __name__ == "__main__":
    """Usage:
    # Geneformer 10M
    ## checkpoint conversion:
    python scripts/singlecell/geneformer/make_nemo2_checkpoints.py --s3-path s3://bionemo-ci/models/geneformer-10M-240530-step-115430-wandb-4ij9ghox.nemo --output-path ~/.cache/bionemo/checkpoints/geneformer_10M_240530_nemo2
    ## checkpoint upload (recursive since it is a directory)
    aws s3 cp --recursive ~/.cache/bionemo/checkpoints/geneformer_10M_240530_nemo2 s3://bionemo-ci/models/geneformer_10M_240530_nemo2
    # Geneformer 106M
    ## checkpoint conversion
    python scripts/protein/esm2/make_nemo2_checkpoints.py --s3-path s3://bionemo-ci/models/geneformer-106M-240530-step-115430-wandb-KZxWJ0I5.nemo--output-path ~/.cache/bionemo/checkpoints/geneformer_106M_240530_nemo2
    ## checkpoint upload
    aws s3 cp --recursive ~/.cache/bionemo/checkpoints/geneformer_106M_240530_nemo2 s3://bionemo-ci/models/geneformer_106M_240530_nemo2
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
        connector = GeneformerNeMo1LightningModuleConnector(nemo1_ckpt_path)
        connector.apply(output_path)
    print(output_path)
