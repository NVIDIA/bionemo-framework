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
from omegaconf import DictConfig
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

from peft import PeftModel


@hydra.main(config_path="hydra_config", config_name="L0_sanity_infer", version_base="1.2")
def main(args: DictConfig):
    """Infer using a PEFT ESM-2 model.

    This script can be run once ESM2 has been PEFT fine-tuned and adapaters have
    been checkpointed. For reference, an example has been provided in the './checkpoints' directory.
    """
    # Ideally we would like to load the PEFT model directly by doing:
    # >>> model = AutoPeftModelForTokenClassification.from_pretrained("<save_directory>", trust_remote_code=True)
    #
    # However, the from_pretrained() function has a positional argument named 'config' which prevent us from passing a
    # a different model config to the base_model. Thus, we first build the base model and then we load the PEFT adapters.

    # Load the custom config
    config = AutoConfig.from_pretrained(args.base_model_config_dir, trust_remote_code=True)

    # Load base model with the custom config
    base_model = AutoModelForTokenClassification.from_pretrained(
        args.model_tag,  # original model tag
        config=config,
        trust_remote_code=True,
    )

    # Load PEFT adapters on top
    peft_model = PeftModel.from_pretrained(base_model, args.peft_model_config_dir)

    tokenizer = AutoTokenizer.from_pretrained("nvidia/esm2_t48_15B_UR50D")

    peft_model = peft_model.to("cuda")
    peft_model.eval()
    inputs = tokenizer("QQLFSYAILGFALSEAMGLFCLMVAFLILFA", return_tensors="pt")

    outputs = peft_model(input_ids=inputs["input_ids"].to("cuda"))

    preds = outputs.logits.argmax(dim=-1)

    id2label = peft_model.config.id2label
    labels = ["".join(id2label[i.item()] for i in seq) for seq in preds]

    print(labels)


if __name__ == "__main__":
    main()
