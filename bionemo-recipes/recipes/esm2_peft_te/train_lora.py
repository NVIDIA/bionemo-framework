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

ss_token_map = {"H": 0, "E": 1, "I": 2, "S": 3, "T": 4, "C": 5, "B": 6, "G": 7, "~": -100}


def tokenize(example):
    """Tokenize both the input protein sequence and the secondary structure labels."""
    result = tokenizer(example["Sequence"], **tokenizer_args)

    # While we can use the rust-based tokenizer for the protein sequence, we manually encode the secondary structure
    # labels. Our goal is to return a list of integer labels with the same shape as the input_ids.
    labels = []
    for batch_idx in range(len(result["input_ids"])):
        sequence_labels = []

        # This array maps the possibly-chunked result["input_ids"] to the original sequence. Because of
        # `return_overflowing_tokens`, each input sequence may be split into multiple input rows.
        offsets = result["offset_mapping"][batch_idx]

        # This gets the original secondary structure sequence for the current chunk.
        ss_sequence = example["Secondary_structure"][result["overflow_to_sample_mapping"][batch_idx]]

        for offset_start, offset_end in offsets:
            if offset_start == offset_end:
                sequence_labels.append(-100)  # Start and end of the sequence tokens can be ignored.
            elif offset_end == offset_start + 1:  # All tokens are single-character.
                ss_char = ss_sequence[offset_start]
                ss_label_value = ss_token_map[ss_char]  # Encode the secondary structure character
                sequence_labels.append(ss_label_value)
            else:
                raise ValueError(f"Invalid offset: {offset_start} {offset_end}")

        labels.append(sequence_labels)

    return {"input_ids": result["input_ids"], "labels": labels}


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

peft_model.to("cuda")

# Create optimizer.
optimizer = torch.optim.AdamW(peft_model.parameters(), lr=1e-4, weight_decay=0.01)

with tqdm(dataloader, desc="Training") as progress_bar:
    for batch in progress_bar:
        batch = {k: v.to("cuda") for k, v in batch.items()}  # noqa PLW2901
        output = peft_model(**batch)
        loss = output.loss
        loss.backward()
        progress_bar.set_postfix({"loss": loss.item()})

        # Step optimizer.
        optimizer.step()
        optimizer.zero_grad()
