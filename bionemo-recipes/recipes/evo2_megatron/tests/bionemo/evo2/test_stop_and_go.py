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


# from typing import Literal

# # import lightning.pytorch as pl
# import pytest
# import torch

# from bionemo.evo2.models.mamba import HybridMambaConfig8BEvo2Loss, MambaModel
# from bionemo.testing import testing_callbacks
# from bionemo.testing.harnesses import stop_and_go
# from bionemo.testing.harnesses.mode import Mode
# from megatron.core.distributed import DistributedDataParallelConfig
# from megatron.core.optimizer import OptimizerConfig

# from nemo import lightning as nl
# from nemo.collections.llm import HyenaModel
# from nemo.collections.llm.gpt.data import MockDataModule
# from nemo.collections.llm.gpt.model.hyena import HyenaNVTestConfig
# from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
# from nemo.lightning.pytorch.optim import CosineAnnealingScheduler, MegatronOptimizerModule
# from nemo.lightning.pytorch.strategies import MegatronStrategy
# from typing_extensions import override


# FIXME update these tests so they work with megatron bridge
# class TestEvo2StopAndGo(stop_and_go.StopAndGoHarness):
#     """Most of these parameters are copied from test_evo2.py which runs train.py."""

#     num_steps: int = 3
#     val_check_interval: int = 1
#     limit_val_batches: int = 1
#     lr: float = 3e-4
#     wd: float = 0.01
#     clip_grad: float = 1.0
#     micro_batch_size: int = 1
#     global_batch_size: int = 1
#     num_layers: int = 4
#     precision: Literal["16-mixed", "bf16-mixed", "32"] = "bf16-mixed"
#     workers: int = 8
#     seq_length: int = 8
#     hybrid_override_pattern: str = "SDH*"
#     use_megatron_comm_overlap_llama3_8k: bool = False
#     hidden_dropout: float = 0.1
#     attention_dropout: float = 0.1
#     no_renormalize_loss: bool = False
#     sequence_parallel: bool = False
#     cross_entropy_loss_fusion: bool = False
#     no_fp32_residual_connection: bool = False
#     add_bias_output: bool = True

#     @classmethod
#     def setup_trainer(
#         cls,
#         mode: Mode,
#     ) -> nl.Trainer:
#         """Setup trainer by passing stop, resume, or continuous callbacks according to mode."""
#         ddp = DistributedDataParallelConfig(
#             check_for_nan_in_grad=True,
#             overlap_grad_reduce=False,
#             overlap_param_gather=False,
#             grad_reduce_in_fp32=False,
#             align_param_gather=False,
#             average_in_collective=True,
#         )
#         strategy = MegatronStrategy(
#             ddp=ddp,
#             tensor_model_parallel_size=1,
#             pipeline_model_parallel_size=1,
#             context_parallel_size=1,
#             sequence_parallel=cls.sequence_parallel,
#             pipeline_dtype=torch.bfloat16,
#             ckpt_async_save=False,
#             ckpt_load_optimizer=True,
#             ckpt_save_optimizer=True,
#             save_ckpt_format="torch_dist",
#             ckpt_load_strictness="log_all",
#         )

#         trainer = nl.Trainer(
#             devices=1,
#             max_steps=cls.num_steps,
#             num_nodes=1,
#             accelerator="gpu",
#             strategy=strategy,
#             limit_val_batches=cls.limit_val_batches,
#             num_sanity_val_steps=0,
#             val_check_interval=cls.val_check_interval,
#             log_every_n_steps=cls.val_check_interval,
#             enable_checkpointing=True,
#             use_distributed_sampler=False,
#             callbacks=list(cls.callbacks[mode].values()),
#             plugins=nl.MegatronMixedPrecision(
#                 precision=cls.precision, params_dtype=torch.bfloat16, grad_reduce_in_fp32=False, fp8_wgrad=False
#             ),
#         )
#         return trainer

#     @override
#     @classmethod
#     def setup_class(cls):
#         super().setup_class()

#         # setup data
#         cls.tokenizer = get_nmt_tokenizer("byte-level")
#         # run stop and go
#         cls.run_stop_and_go()

#     @classmethod
#     def setup_model(cls, mode: Mode) -> tuple[pl.LightningModule, pl.LightningDataModule, nl.MegatronOptimizerModule]:
#         # build data module
#         data = MockDataModule(
#             seq_length=cls.seq_length,
#             micro_batch_size=cls.micro_batch_size,
#             global_batch_size=cls.global_batch_size,
#             num_workers=cls.workers,
#             tokenizer=cls.tokenizer,
#         )

#         data.init_global_step = 0
#         # config
#         config = HyenaNVTestConfig(
#             **{
#                 "tp_comm_overlap": cls.use_megatron_comm_overlap_llama3_8k,
#                 "seq_length": cls.seq_length,
#                 "use_te": True,
#                 "params_dtype": torch.bfloat16,
#                 "bf16": True,
#                 "recompute_granularity": None,
#                 "recompute_method": None,
#                 "recompute_num_layers": None,
#                 "num_layers": cls.num_layers,
#                 "hidden_size": 1920,
#                 "hybrid_override_pattern": cls.hybrid_override_pattern,
#                 "num_attention_heads": 15,
#                 "num_query_groups": 15,
#                 "ffn_hidden_size": 5120,
#                 "hidden_dropout": cls.hidden_dropout,
#                 "num_groups_hyena": 1920,
#                 "num_groups_hyena_medium": 128,
#                 "num_groups_hyena_short": 128,
#                 "attention_dropout": cls.attention_dropout,
#                 "to_upper": "weighted" if cls.no_renormalize_loss else "normalized_weighted",
#                 "distribute_saved_activations": False if cls.sequence_parallel else True,
#                 "cross_entropy_loss_fusion": cls.cross_entropy_loss_fusion,
#                 "fp32_residual_connection": not cls.no_fp32_residual_connection,
#                 "add_bias_output": cls.add_bias_output,
#             }
#         )

#         optimizer_config = OptimizerConfig(
#             optimizer="adam",
#             lr=cls.lr,
#             adam_beta1=0.9,
#             adam_beta2=0.95,
#             weight_decay=cls.wd,
#             clip_grad=cls.clip_grad,
#             params_dtype=torch.float32,
#             use_distributed_optimizer=True,
#             bf16=True,
#         )
#         # build optimizer
#         optimizer = MegatronOptimizerModule(
#             config=optimizer_config,
#             lr_scheduler=CosineAnnealingScheduler(warmup_steps=1, max_steps=cls.num_steps, min_lr=3e-5),
#             no_weight_decay_cond=config.hyena_no_weight_decay_cond_fn,
#         )

#         # # Build model
#         module = HyenaModel(config, tokenizer=data.tokenizer)
#         optimizer.connect(module)
#         return module, data, optimizer

#     @pytest.mark.parametrize(
#         "callback_type",
#         [
#             testing_callbacks.LearningRateCallback,
#             testing_callbacks.GlobalStepStateCallback,
#             testing_callbacks.ConsumedSamplesCallback,
#             testing_callbacks.OptimizerStateCallback,
#             testing_callbacks.TrainInputCallback,
#             testing_callbacks.TrainOutputCallback,
#             testing_callbacks.TrainLossCallback,
#             testing_callbacks.ValidInputCallback,
#             testing_callbacks.ValidOutputCallback,
#             testing_callbacks.ValidLossCallback,
#         ],
#     )
#     def test_stop_and_go_consistency(self, callback_type):
#         if callback_type in [
#             testing_callbacks.ValidLossCallback,
#             testing_callbacks.ValidOutputCallback,
#             testing_callbacks.TrainInputCallback,
#             testing_callbacks.TrainOutputCallback,
#             testing_callbacks.TrainLossCallback,
#             testing_callbacks.OptimizerStateCallback,
#         ]:
#             pytest.xfail(reason="Tensors not close")
#         super().test_stop_and_go_consistency(callback_type)

#     @pytest.mark.skip(reason="TODO: assert train_consumed_go > 0 fails.")
#     def test_train_val_init_consumed_samples(self):
#         pass

import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from bionemo.evo2.data.dataset_tokenizer import DEFAULT_HF_TOKENIZER_MODEL_PATH


def find_free_network_port() -> int:
    """Finds a free port on localhost.

    It is useful in single-node training when we don't want to connect to a real master node but
    have to set the `MASTER_PORT` environment variable.
    """
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.mark.parametrize(
    "tp_size,cp_size,dp_size,pp_size,dp_rank_check",
    [
        (1, 1, 1, 1, False),
        (1, 1, 2, 1, True),
        (1, 1, 2, 1, False),
        (1, 2, 1, 1, True),
        (2, 1, 1, 1, False),
    ],
)
def test_stop_and_go(
    tmp_path: Path, tp_size: int, cp_size: int, dp_size: int, pp_size: int = 1, dp_rank_check: bool = False
):
    """Test stop and go functionality."""
    world_size = tp_size * pp_size * cp_size * dp_size
    mbs = 32
    gbs = mbs * dp_size
    num_gpus = torch.cuda.device_count()
    if world_size > num_gpus:
        pytest.skip(f"World size {world_size} is greater than the number of GPUs {num_gpus}")
    run_dir = tmp_path / f"run_tp{tp_size}_pp{pp_size}_cp{cp_size}_dp{dp_size}"
    run_dir.mkdir(parents=True, exist_ok=True)
    dp_rank_check_str = "--debug-ddp-parity-freq 5" if dp_rank_check else ""
    cmd1 = f"""torchrun --nproc-per-node {world_size} --no-python \
    train_evo2 \
        --hf-tokenizer-model-path {DEFAULT_HF_TOKENIZER_MODEL_PATH} \
        --model-size striped_hyena_1b_nv_parallel --num-layers 4 --hybrid-override-pattern SDH* \
        --max-steps 5 --eval-interval 5 \
        --eval-iters 3 --mock-data --result-dir {run_dir} \
        --micro-batch-size {mbs} --global-batch-size {gbs} --seq-length 512 \
        --tensor-model-parallel {tp_size} \
        --pipeline-model-parallel {pp_size} \
        --context-parallel {cp_size} \
        {dp_rank_check_str} \
        --use-precision-aware-optimizer --dataset-seed 33 \
        --seed 41 --spike-no-more-embedding-init \
        --no-weight-decay-embeddings --cross-entropy-loss-fusion \
        --grad-reduce-in-fp32 \
        --decay-steps 1000 --warmup-steps 10 \
        --eod-pad-in-loss-mask \
        --log-interval 1
    """

    # Split the command and run it
    cmd_parts = shlex.split(cmd1)
    env = dict(**os.environ)
    env["MASTER_PORT"] = str(find_free_network_port())
    env["NCCL_P2P_DISABLE"] = "1"
    result = subprocess.run(cmd_parts, check=False, capture_output=True, text=True, cwd=run_dir, env=env)

    stdout = result.stdout
    stderr = result.stderr
    returncode = result.returncode

    # For debugging, print the output
    print(f"Return code: {returncode}")
    print(f"STDOUT:\n{stdout}")
    print(f"STDERR:\n{stderr}")

    # Assert the command succeeded
    assert returncode == 0, f"Command failed with return code {returncode}\nSTDERR:\n{stderr}"
    result_dir = run_dir / "evo2"
    ckpt_dir = result_dir / "checkpoints"
    tb_log_dir = result_dir / "tb_logs"
    assert ckpt_dir.exists() and ckpt_dir.is_dir(), "Checkpoints directory not found"
    assert tb_log_dir.exists() and tb_log_dir.is_dir(), "TensorBoard logs directory not found"
    iter_5_dir = ckpt_dir / "iter_0000005"
    assert iter_5_dir.exists() and iter_5_dir.is_dir(), f"No iterations 5 checkpoint found in {ckpt_dir}"
    assert len(list(ckpt_dir.glob("iter_*"))) == 1, f"Expected 1 iterations, found {list(ckpt_dir.glob('iter_*'))}"
    # Load tensorboard logs to verify they were written correctly

    # Find the events file(s) in tb_log_dir
    event_files = list(tb_log_dir.rglob("events.out.*"))
    assert len(event_files) > 0, f"No tensorboard event files found in {tb_log_dir}"

    # Load events from the event files
    event_acc = EventAccumulator(str(tb_log_dir))
    event_acc.Reload()

    # 1. collect the last loss, as well as the average of the last step validation losses, as well as the last step
    # Note: EventAccumulator.Scalars returns a list of ScalarEvent(wall_time, step, value)
    lm_loss_events = event_acc.Scalars("lm loss")
    val_loss_events = event_acc.Scalars("lm loss validation")

    assert len(lm_loss_events) > 0, "No 'lm loss' events found in run 1"
    last_lm_loss_step = lm_loss_events[-1].step
    last_lm_loss_val = lm_loss_events[-1].value

    print(f"Run 1: Last training step: {last_lm_loss_step}, Loss: {last_lm_loss_val}")

    # Check if we have validation events (might depend on eval-interval)
    if val_loss_events:
        print(f"Run 1: Last validation loss: {val_loss_events[-1].value}")

    assert last_lm_loss_step == 5, f"Expected run 1 to end at step 5, but got {last_lm_loss_step}"

    # 2. run the above training command a second time, this time set max_steps to 10. Verify that the run resumes from the last step.
    #   Do this by moving the tb_logs to a different directory from the first part so the second run makes fresh logs.
    tb_log_dir_run1 = result_dir / "tb_logs_run1"
    if tb_log_dir.exists():
        shutil.move(str(tb_log_dir), str(tb_log_dir_run1))

    # Modify the command to increase max steps to 10
    # We reuse the same result_dir so it picks up the checkpoint
    cmd2 = cmd1.replace("--max-steps 5", "--max-steps 10")
    cmd_parts_2 = shlex.split(cmd2)

    print("Starting Run 2 (resuming to step 10)...")
    env["MASTER_PORT"] = str(find_free_network_port())
    result_2 = subprocess.run(cmd_parts_2, check=False, capture_output=True, text=True, cwd=run_dir, env=env)

    print(f"Run 2 Return code: {result_2.returncode}")
    if result_2.returncode != 0:
        print(f"Run 2 STDERR:\n{result_2.stderr}")

    assert result_2.returncode == 0, f"Run 2 failed with return code {result_2.returncode}"

    # 3. Load the new tb logs as before, and sanity check my recommendations as well as any others that make sense.
    assert tb_log_dir.exists(), "TensorBoard logs directory not found after Run 2"

    event_acc_2 = EventAccumulator(str(tb_log_dir))
    event_acc_2.Reload()

    lm_loss_events_2 = event_acc_2.Scalars("lm loss")
    assert len(lm_loss_events_2) > 0, "No 'lm loss' events found in run 2"

    first_step_run2 = lm_loss_events_2[0].step
    last_step_run2 = lm_loss_events_2[-1].step

    print(f"Run 2: First step: {first_step_run2}, Last step: {last_step_run2}")

    # Sanity checks:
    # 1. Resumption: Should start after step 5 (e.g., step 6)
    assert first_step_run2 > 5, f"Run 2 should resume after step 5, but started at {first_step_run2}"

    # 2. Completion: Should reach step 10
    assert last_step_run2 == 10, f"Run 2 should reach step 10, but ended at {last_step_run2}"

    # 3. Loss Continuity check (basic): The first loss of run 2 should be reasonably close to the last loss of run 1,
    #    or at least not exploding, though optimization steps might cause fluctuations.
    first_loss_run2 = lm_loss_events_2[0].value
    print(f"Run 1 Last Loss: {last_lm_loss_val}, Run 2 First Loss: {first_loss_run2}")
    assert abs(last_lm_loss_val - first_loss_run2) < 0.1, (
        f"Run 2 first loss {first_loss_run2} is not close to run 1 last loss {last_lm_loss_val}"
    )
