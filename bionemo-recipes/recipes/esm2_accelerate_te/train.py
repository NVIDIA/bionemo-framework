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
import time
from pathlib import Path

import hydra
import torch
import transformers
import wandb
from accelerate import PartialState
from omegaconf import DictConfig, OmegaConf
from transformers import AutoConfig, AutoModelForMaskedLM
from transformers.trainer import Trainer
from transformers.trainer_callback import TrainerCallback
from transformers.training_args import TrainingArguments

from callbacks import StepTimingCallback, StopAfterNStepsCallback
from dataset import create_datasets_and_collator
from metrics import compute_metrics


logger = logging.getLogger(__name__)


class StepTimeCallback(TrainerCallback):
    """Callback to log performance metrics for an end-to-end training step."""

    def __init__(self, trainer):
        """Initialize the HuggingFace TrainerCallback.

        Args:
            trainer (Trainer): The HuggingFace Trainer instance.
        """
        super().__init__()
        self.start_time = None
        self.end_time = None
        self.trainer = trainer

    def on_step_begin(self, args, state, control, **kwargs):
        """Start the timer for the end-to-end training step."""
        self.start_time = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        """Stop the timer for the end-to-end training step and log relevant performance metrics."""
        self.end_time = time.perf_counter()
        self.step_time = self.end_time - self.start_time
        self.trainer.total_step_time_buffer.append(self.step_time)
        self.trainer.samples_per_second_buffer.append(self.trainer.microbatch_size / self.step_time)
        self.trainer.num_tokens_buffer.append(self.trainer.seq_length * self.trainer.microbatch_size)
        self.trainer.tps_buffer.append(self.trainer.seq_length * self.trainer.microbatch_size / self.step_time)
        self.trainer.sig_tps_buffer.append(self.trainer.num_sig_tokens_buffer[-1] / self.step_time)
        self.trainer.alloc_memory_gb_buffer.append(torch.cuda.memory.memory_allocated() / 1024**3)
        self.trainer.reserv_memory_gb_buffer.append(torch.cuda.memory.memory_reserved() / 1024**3)
        self.trainer.log(
            {
                "loss": self.trainer.loss,
                "perf/total_step_time_sec": self.trainer.total_step_time_buffer[-1],
                "perf/model_step_time_sec": self.trainer.model_step_time_buffer[-1],
                "perf/microbatch_size": self.trainer.microbatch_size,
                "perf/samples_per_second": self.trainer.samples_per_second_buffer[-1],
                "perf/num_tokens": self.trainer.num_tokens_buffer[-1],
                "perf/num_sig_tokens": self.trainer.num_sig_tokens_buffer[-1],
                "perf/tps": self.trainer.tps_buffer[-1],
                "perf/sig_tps": self.trainer.sig_tps_buffer[-1],
                "perf/alloc_memory_gb": self.trainer.alloc_memory_gb_buffer[-1],
                "perf/reserv_memory_gb": self.trainer.reserv_memory_gb_buffer[-1],
            }
        )


class PerfTrainer(Trainer):
    """Trainer class to log performance metrics for an end-to-end training step.

    Adds the StepTimeCallback to the Trainer, and stores historical performance metrics.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the PerfTrainer."""
        super().__init__(*args, **kwargs)
        self.total_step_time_buffer = []
        self.model_step_time_buffer = []
        self.samples_per_second_buffer = []
        self.num_tokens_buffer = []
        self.num_sig_tokens_buffer = []
        self.tps_buffer = []
        self.sig_tps_buffer = []
        self.alloc_memory_gb_buffer = []
        self.reserv_memory_gb_buffer = []

        # HACK(@cspades): Add total step time callback, and pass the
        # Trainer to the callback for communicating information.
        self.add_callback(StepTimeCallback(self))

    def training_step(self, model, inputs, num_items_in_batch):
        """Run the training step and capture some metrics pertaining to the model computation and input."""
        start_time = time.perf_counter()
        self.loss = super().training_step(model, inputs, num_items_in_batch)
        self.model_step_time_buffer.append(time.perf_counter() - start_time)
        self.microbatch_size, self.seq_length = inputs["input_ids"].shape
        self.num_sig_tokens_buffer.append(inputs["input_ids"][inputs["input_ids"] != 1].shape[0])
        return self.loss

    def evaluate(self, *args, **kwargs):
        """Evaluate the model and log average performance metrics across the entire training session."""
        super().evaluate(*args, **kwargs)
        # Take the average of all steps.
        avg_total_step_time_sec = sum(self.total_step_time_buffer) / len(self.total_step_time_buffer)
        avg_model_step_time_sec = sum(self.model_step_time_buffer) / len(self.model_step_time_buffer)
        avg_samples_per_second = sum(self.samples_per_second_buffer) / len(self.samples_per_second_buffer)
        avg_num_tokens = sum(self.num_tokens_buffer) / len(self.num_tokens_buffer)
        avg_num_sig_tokens = sum(self.num_sig_tokens_buffer) / len(self.num_sig_tokens_buffer)
        avg_tps = sum(self.tps_buffer) / len(self.tps_buffer)
        avg_sig_tps = sum(self.sig_tps_buffer) / len(self.sig_tps_buffer)
        avg_alloc_memory_gb = sum(self.alloc_memory_gb_buffer) / len(self.alloc_memory_gb_buffer)
        avg_reserv_memory_gb = sum(self.reserv_memory_gb_buffer) / len(self.reserv_memory_gb_buffer)
        self.log(
            {
                "perf/avg_model_step_time_sec": avg_model_step_time_sec,
                "perf/avg_total_step_time_sec": avg_total_step_time_sec,
                "perf/avg_samples_per_second": avg_samples_per_second,
                "perf/avg_num_tokens": avg_num_tokens,
                "perf/avg_num_sig_tokens": avg_num_sig_tokens,
                "perf/avg_tps": avg_tps,
                "perf/avg_sig_tps": avg_sig_tps,
                "perf/avg_alloc_memory_gb": avg_alloc_memory_gb,
                "perf/avg_reserv_memory_gb": avg_reserv_memory_gb,
            }
        )


@hydra.main(config_path="hydra_config", config_name="L0_sanity", version_base="1.2")
def main(args: DictConfig):
    """Entrypoint."""
    # add wandb logging on main process
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        wandb.init(config=OmegaConf.to_container(args, resolve=True, throw_on_missing=True), **args.wandb_init_args)
    # Initialize Accelerate's distributed state early so torch device is set per process
    state = PartialState()
    logger.info(
        "Accelerate initialized (local_process_index=%s, num_processes=%s, device=%s)",
        state.local_process_index,
        state.num_processes,
        state.device,
    )

    config = AutoConfig.from_pretrained(args.model_tag, trust_remote_code=True)
    if args.attn_backend is not None:
        config._attn_implementation = args.attn_backend
    model = AutoModelForMaskedLM.from_config(config, trust_remote_code=True, dtype=torch.bfloat16)

    train_dataset, eval_dataset, data_collator = create_datasets_and_collator(**args.dataset)

    training_args = TrainingArguments(**args.trainer)

    trainer = PerfTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=[
            StopAfterNStepsCallback(args.stop_after_n_steps),
            StepTimingCallback(),
        ],
    )

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

    # Report Torch memory profile.
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        torch_memory_profiler_snapshot = torch.cuda.memory._snapshot()
        from pickle import dump

        from hydra.core.hydra_config import HydraConfig

        with open(
            # Path will only exist when using @hydra.main()!
            Path(HydraConfig.get().runtime.output_dir)
            / f"{args.trainer.run_name.replace('/', '_')}_torch_memory_profiler_snapshot.pickle",
            "wb",
        ) as f:
            dump(torch_memory_profiler_snapshot, f)

    if wandb.run is not None:
        wandb.finish()

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    torch.cuda.memory._record_memory_history(max_entries=250000)
    main()
