# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import hydra
import torch
from omegaconf import DictConfig
from transformers import AutoTokenizer, EsmForProteinFolding
from utils import convert_outputs_to_pdb, load_fasta


@hydra.main(config_path="hydra_config", config_name="L0_sanity_infer", version_base="1.2")
def main(args: DictConfig):
    """Infer the protein structure using ESM-Fold."""
    records = load_fasta(args.input_fasta_file)
    test_protein = records[0]["sequence"]

    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)

    model = model.to("cuda").eval()

    # Uncomment to switch the stem to float16
    model.esm = model.esm.half()

    tokenized_input = tokenizer([test_protein], return_tensors="pt", add_special_tokens=False)["input_ids"]

    tokenized_input = tokenized_input.to("cuda")

    with torch.no_grad():
        output = model(tokenized_input)

    pdbs = convert_outputs_to_pdb(output)

    with open(args.output_pdb_file, "w") as f:
        f.write("".join(pdbs))


if __name__ == "__main__":
    main()
