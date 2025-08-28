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

import logging
from pathlib import Path

import hydra
import torch
import transformers
from omegaconf import DictConfig
from transformers import AutoConfig, AutoModelForMaskedLM
from transformers.trainer import Trainer
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

from dataset import create_datasets_and_collator
from metrics import compute_metrics


logger = logging.getLogger(__name__)


@hydra.main(config_path="hydra_config", config_name="L0_sanity", version_base="1.2")
def main(args: DictConfig):
    """Entrypoint."""
    config = AutoConfig.from_pretrained(args.model_tag, trust_remote_code=True)
    config.max_length = args.max_seq_length

    model = AutoModelForMaskedLM.from_config(
        config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    train_dataset, eval_dataset, data_collator = create_datasets_and_collator(
        pretained_model=args.model_tag,
        max_length=config.max_length,
        data_size=args.data_size,
    )

    training_args = TrainingArguments(**args.trainer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=[StopAfterNStepsCallback(args.stop_after_n_steps)],
    )

    logger.info("ACCELERATE STATE:\n%s\n", trainer.accelerator.state)

    if training_args.do_train:
        Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)
        last_checkpoint = transformers.trainer_utils.get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None:
            logger.info("Resuming from checkpoint: %s", last_checkpoint)
        else:
            logger.info("No checkpoint found, starting from scratch")
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
        logger.info("Training complete. Metrics: %s", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_model(str(Path(training_args.output_dir) / "checkpoint-last"))

    if training_args.do_eval:
        trainer.evaluate()

    torch.distributed.destroy_process_group()


class StopAfterNStepsCallback(TrainerCallback):
    """Callback to interrupt training after a specified number of steps.

    This allows us to use a learning rate scheduler consistent with the full training run while
    stopping after a pre-determined number of steps.
    """

    def __init__(self, max_steps: int):
        """Initialize the callback."""
        self.max_steps = max_steps

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Interrupt training after a specified number of steps."""
        if state.global_step >= self.max_steps:
            control.should_training_stop = True
        return control


if __name__ == "__main__":
    main()
