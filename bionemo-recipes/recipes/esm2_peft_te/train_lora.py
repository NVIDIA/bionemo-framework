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


"""Demonstration of LoRA fine-tuning of ESM-2 with Transformer Engine and PEFT.

Still needs:
 - [ ] Hydra config management.
 - [ ] THD / sequence packing.
 - [ ] DDP / Multi-node training.
 - [ ] FP8 tests.
 - [ ] Perf / wandb logging.
"""

import hydra
import peft
import torch
from datasets import load_dataset
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
)

from distributed_config import DistributedConfig
from perf_logger import PerfLogger


# Print fully without truncation
torch.set_printoptions(threshold=int(1e6))  # Set a very high threshold


def create_dataloader(
    use_sanity_dataset: bool = False,
    micro_batch_size: int = 2,
    max_seq_length: int = 1024,
    stride: int = 16,
    perform_validation: bool = False,
    validation_samples: int = 1024,
    ss3_classification: bool = True,
    dataset_file_name: str = "peft_sanity_dataset.parquet",
    **kwargs,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader | None]:
    """Create a dataloader for the secondary structure dataset."""
    if use_sanity_dataset:
        # 5000 row sanity dataset.
        train_dataset = load_dataset("parquet", data_files=dataset_file_name, split="train", streaming=True)
    else:
        # Full-scale source dataset.
        train_dataset = load_dataset("lamm-mit/protein_secondary_structure_from_PDB", split="train", streaming=True)

    if perform_validation:
        val_dataset = train_dataset.take(validation_samples)
        train_dataset = train_dataset.skip(validation_samples)

    ss8_token_map = {"H": 0, "I": 1, "G": 2, "E": 3, "B": 4, "S": 5, "T": 6, "~": 7}  # '~' denotes coil / unstructured
    ss3_token_map = {"H": 0, "I": 0, "G": 0, "E": 1, "B": 1, "S": 2, "T": 2, "~": 2}  # '~' denotes coil / unstructured

    if ss3_classification:
        ss_token_map = ss3_token_map
    else:
        ss_token_map = ss8_token_map

    tokenizer = AutoTokenizer.from_pretrained("example_8m_checkpoint")
    tokenize_args = {
        "max_length": max_seq_length,
        "truncation": True,
        "stride": stride,
        "return_overflowing_tokens": True,
        "return_offsets_mapping": True,
    }

    def tokenize(example):
        """Tokenize both the input protein sequence and the secondary structure labels."""
        result = tokenizer(example["Sequence"], **tokenize_args)

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

    train_tokenized_dataset = train_dataset.map(
        tokenize,
        batched=True,
        remove_columns=[col for col in train_dataset.features if col not in ["input_ids", "labels"]],
    )

    collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding="max_length", max_length=1024)
    train_dataloader = torch.utils.data.DataLoader(
        train_tokenized_dataset, batch_size=micro_batch_size, collate_fn=collator
    )

    if perform_validation:
        val_tokenized_dataset = val_dataset.map(
            tokenize,
            batched=True,
            remove_columns=[col for col in val_dataset.features if col not in ["input_ids", "labels"]],
        )

        collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding="max_length", max_length=1024)
        val_dataloader = torch.utils.data.DataLoader(
            val_tokenized_dataset, batch_size=micro_batch_size, collate_fn=collator
        )
    else:
        val_dataloader = None

    return train_dataloader, val_dataloader


def compute_accuracy(preds, labels, ignore_index=-100) -> float:
    """Calculate the accuracy."""
    preds_labels = torch.argmax(preds, dim=-1)
    mask = labels != ignore_index
    correct = (preds_labels == labels) & mask
    total = mask.sum()
    if total == 0:
        return 0.0
    acc = correct.sum().item() / total.item()

    # print("-------")
    # print(labels.shape)
    # print("---------")
    # print(preds_labels)
    # print("----------")
    # print(labels)
    # print("---------")
    # print(correct)
    # print("----------")
    # print(correct.sum())
    return acc


@hydra.main(config_path="hydra_config", config_name="L0_sanity", version_base="1.2")
def main(args: DictConfig) -> float:
    """Training loop for LoRA fine-tuning of ESM-2 with Transformer Engine and PEFT.

    Args:
        args: Configuration arguments from hydra.

    Returns:
        Final loss value.
    """
    train_dataloader, val_dataloader = create_dataloader(perform_validation=args.perform_validation, **args.dataset)

    # Initialize the distributed configuration, including creating the distributed process group.
    dist_config = DistributedConfig()

    # For testing, we don't want to depend on loading pre-trained weights.
    config = AutoConfig.from_pretrained(args.model_tag, trust_remote_code=True)
    if args.dataset["ss3_classification"]:
        config.num_labels = 3
    else:
        config.num_labels = 8

    model = AutoModelForTokenClassification.from_config(config, trust_remote_code=True)
    print("----- Model --------")
    print(model)
    print("-------------------------")
    # Alternatively, we'd want to load an actual pre-trained checkpoint.
    # model = AutoModelForTokenClassification.from_pretrained(
    #     "example_8m_checkpoint", num_labels=8, trust_remote_code=True, dtype="bfloat16"
    # )

    peft_config = peft.LoraConfig(
        task_type=peft.TaskType.TOKEN_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        target_modules=["layernorm_qkv"],  # TODO: figure out if this could work?
        # target_parameters=["layernorm_qkv.weight"],
        bias="none",
    )

    peft_model = peft.get_peft_model(model, peft_config)
    peft_model.to("cuda", dtype=torch.bfloat16)
    print("----- PEFT Model --------")
    print(peft_model)
    print("-------------------------")
    peft_model.print_trainable_parameters()

    # Create optimizer.
    optimizer = torch.optim.AdamW(peft_model.parameters(), **args.adamw_kwargs)

    perf_logger = PerfLogger(dist_config, args)

    # Training loop.
    step = 0
    while step < args.num_train_steps:
        with tqdm(train_dataloader, desc="Training") as progress_bar:
            for batch in progress_bar:
                batch = {k: v.to("cuda") for k, v in batch.items()}  # noqa PLW2901
                # print(batch["input_ids"].shape)
                output = peft_model(**batch)
                loss = output.loss
                loss.backward()
                progress_bar.set_postfix({"loss": loss.item()})

                # Compute and clip gradient norms.
                total_norm = torch.nn.utils.clip_grad_norm_(peft_model.parameters(), max_norm=1.0).item()

                # Step optimizer.
                optimizer.step()
                optimizer.zero_grad()

                step += 1

                # Validation
                avg_val_loss = None
                avg_val_acc = None
                if args.perform_validation and step % args.validation_interval == 0:
                    peft_model.eval()
                    val_loss_total = 0.0
                    val_acc_total = 0.0
                    val_steps = 1
                    with torch.no_grad():
                        for val_batch in val_dataloader:
                            val_batch = {k: v.to("cuda") for k, v in val_batch.items()}  # noqa PLW2901
                            val_output = peft_model(**val_batch)

                            # Loss
                            val_loss_total += val_output.loss.item()

                            # Accuracy
                            logits = val_output.logits
                            labels = val_batch["labels"]
                            val_acc_total += compute_accuracy(logits, labels)

                            val_steps += 1

                    avg_val_loss = val_loss_total / val_steps
                    avg_val_acc = val_acc_total / val_steps

                    print(f"\nStep: {step}: Validation Loss = {avg_val_loss:.4f}, Accuracy: {avg_val_acc:.4f}\n")
                    peft_model.train()

                perf_logger.log_step(
                    step=step,
                    batch=batch,
                    outputs=output,
                    grad_norm=total_norm,
                    lr=optimizer.param_groups[0]["lr"],
                    val_loss=avg_val_loss,
                    val_acc=avg_val_acc,
                )

                if step >= args.num_train_steps:
                    break

    perf_logger.finish()

    return perf_logger.min_loss


if __name__ == "__main__":
    main()
