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

import peft
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
)


ss_dataset = load_dataset("lamm-mit/protein_secondary_structure_from_PDB", split="train", streaming=True)
tokenizer = AutoTokenizer.from_pretrained("example_8m_checkpoint")

tokenizer_args = {
    "max_length": 128,
    "truncation": True,
    "stride": 16,  # TODO: figure this out later
    "return_overflowing_tokens": True,
    "return_offsets_mapping": True,
}


def tokenize(example):
    """Tokenize both the input protein sequence and the secondary structure labels."""
    result = tokenizer(example["Sequence"], **tokenizer_args)
    breakpoint()
    # result["labels"] = [[ii if ii != 8 else -100 for ii in item] for item in tokenized_labels]
    return result


tokenized_dataset = ss_dataset.map(
    tokenize, batched=True, remove_columns=[col for col in ss_dataset.features if col not in ["input_ids", "labels"]]
)

# TODO: use THD / sequence packing.
collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding="max_length", max_length=1024)
dataloader = torch.utils.data.DataLoader(tokenized_dataset, batch_size=16, collate_fn=collator)


model = AutoModelForTokenClassification.from_pretrained(
    "example_8m_checkpoint", num_labels=8, trust_remote_code=True, dtype="bfloat16"
)


peft_config = peft.LoraConfig(
    task_type=peft.TaskType.TOKEN_CLS,
    inference_mode=False,
    r=16,
    lora_alpha=16,
    # target_modules=["layernorm_qkv"],  # TODO: figure out if this could work?
    target_parameters=["layernorm_qkv.weight"],
    bias="none",
)

peft_model = peft.get_peft_model(model, peft_config)

model.to("cuda")

# Create optimizer.
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

with tqdm(dataloader, desc="Training") as progress_bar:
    for batch in progress_bar:
        batch = {k: v.to("cuda") for k, v in batch.items()}  # noqa PLW2901
        output = model(**batch)
        loss = output.loss
        loss.backward()
        progress_bar.set_postfix({"loss": loss.item()})

        # Step optimizer.
        optimizer.step()
        optimizer.zero_grad()
