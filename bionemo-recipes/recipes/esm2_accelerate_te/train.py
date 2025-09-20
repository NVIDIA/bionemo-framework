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
from accelerate import PartialState
from omegaconf import DictConfig
from transformers import AutoConfig, AutoModelForMaskedLM
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

from callbacks import StopAfterNStepsCallback
from dataset import create_datasets_and_collator
from metrics import compute_metrics


logger = logging.getLogger(__name__)


class PerfTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_time_sec = []
        self.samples_per_second = []
        self.num_tokens = []
        self.num_sig_tokens = []
        self.tps = []
        self.sig_tps = []
        self.alloc_memory_gb = []
        self.reserv_memory_gb = []

    def training_step(self, model, inputs, num_items_in_batch):
        """
        model, input dictionary, and number of accumulated samples in this optimization cycle
        """
        import time
        start_time = time.time()
        loss = super().training_step(model, inputs, num_items_in_batch)
        step_time = time.time() - start_time
        microbatch_size, seq_length = inputs["input_ids"].shape
        num_sig_tokens = inputs["input_ids"][inputs["input_ids"] != 1].shape[0]
        self.step_time_sec.append(step_time)
        self.samples_per_second.append(microbatch_size / step_time)
        self.num_tokens.append(seq_length * microbatch_size)
        self.num_sig_tokens.append(num_sig_tokens)
        self.tps.append(seq_length * microbatch_size / step_time)
        self.sig_tps.append(num_sig_tokens / step_time)
        self.alloc_memory_gb.append(torch.cuda.memory.memory_allocated() / 1024**3)
        self.reserv_memory_gb.append(torch.cuda.memory.memory_reserved() / 1024**3)
        self.log({
            "perf/step_time_in_seconds": step_time,
            "perf/microbatch_size": microbatch_size,
            "perf/samples_per_second": self.samples_per_second[-1],
            "perf/num_tokens": seq_length * microbatch_size,
            "perf/num_sig_tokens": num_sig_tokens,
            "perf/tps": self.tps[-1],
            "perf/sig_tps": self.sig_tps[-1],
            "perf/alloc_memory_gb": self.alloc_memory_gb[-1],
            "perf/reserv_memory_gb": self.reserv_memory_gb[-1],
        })
        return loss
    
    def evaluate(self, *args, **kwargs):
        super().evaluate(*args, **kwargs)
        # Take the average of all steps.
        avg_step_time_sec = sum(self.step_time_sec) / len(self.step_time_sec)
        avg_samples_per_second = sum(self.samples_per_second) / len(self.samples_per_second)
        avg_num_tokens = sum(self.num_tokens) / len(self.num_tokens)
        avg_num_sig_tokens = sum(self.num_sig_tokens) / len(self.num_sig_tokens)
        avg_tps = sum(self.tps) / len(self.tps)
        avg_sig_tps = sum(self.sig_tps) / len(self.sig_tps)
        avg_alloc_memory_gb = sum(self.alloc_memory_gb) / len(self.alloc_memory_gb)
        avg_reserv_memory_gb = sum(self.reserv_memory_gb) / len(self.reserv_memory_gb)
        self.log({
            "perf/avg_step_time_in_seconds": avg_step_time_sec,
            "perf/avg_samples_per_second": avg_samples_per_second,
            "perf/avg_num_tokens": avg_num_tokens,
            "perf/avg_num_sig_tokens": avg_num_sig_tokens,
            "perf/avg_tps": avg_tps,
            "perf/avg_sig_tps": avg_sig_tps,
            "perf/avg_alloc_memory_gb": avg_alloc_memory_gb,
            "perf/avg_reserv_memory_gb": avg_reserv_memory_gb,
        })


@hydra.main(config_path="hydra_config", config_name="L0_sanity", version_base="1.2")
def main(args: DictConfig):
    """Entrypoint."""
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
        callbacks=[StopAfterNStepsCallback(args.stop_after_n_steps)],
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
    torch_memory_profiler_snapshot = torch.cuda.memory._snapshot()
    from pickle import dump
    from hydra.core.hydra_config import HydraConfig
    with open(
        # Path will only exist when using @hydra.main()!
        Path(HydraConfig.get().runtime.output_dir) / f"{args.trainer.run_name.replace('/', '_')}_torch_memory_profiler_snapshot.pickle",
        "wb",
    ) as f:
        dump(torch_memory_profiler_snapshot, f)


if __name__ == "__main__":
    torch.cuda.memory._record_memory_history(max_entries=250000)
    main()
