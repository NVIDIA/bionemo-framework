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


from typing import Literal

import lightning.pytorch as pl
import torch
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig
from nemo import lightning as nl
from nemo.collections.llm import HyenaModel
from nemo.collections.llm.gpt.data import MockDataModule
from nemo.collections.llm.gpt.model.hyena import HyenaNV1bConfig
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.lightning.pytorch.optim import CosineAnnealingScheduler, MegatronOptimizerModule
from nemo.lightning.pytorch.strategies import MegatronStrategy
from typing_extensions import override

from bionemo.core.utils.dtypes import get_autocast_dtype
from bionemo.testing.harnesses import stop_and_go
from bionemo.testing.harnesses.mode import Mode


MODEL_PRECISION: Literal["bf16-mixed"] = "bf16-mixed"


class TestEvo2StopAndGo(stop_and_go.StopAndGoHarness):
    num_steps: int = 2
    val_check_interval: int = 2
    limit_val_batches: int = 1
    lr: float = 3e-4
    wd: float = 0.01
    clip_grad: float = 1.0
    micro_batch_size: int = 1
    global_batch_size: int = 1

    precision: Literal["16-mixed", "bf16-mixed", "32"] = get_autocast_dtype(MODEL_PRECISION)
    workers: int = 8
    seq_length: int = 8
    hybrid_override_pattern: str = "SDH*"
    use_megatron_comm_overlap_llama3_8k: bool = False
    hidden_dropout: float = 0.1
    attention_dropout: float = 0.1
    no_renormalize_loss: bool = False
    sequence_parallel: bool = False
    cross_entropy_loss_fusion: bool = False
    no_fp32_residual_connection: bool = False
    add_bias_output: bool = True

    @classmethod
    def setup_trainer(
        cls,
        mode: Mode,
    ) -> nl.Trainer:
        """Setup trainer by passing stop, resume, or continuous callbacks according to mode."""
        ddp = DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            overlap_grad_reduce=False,
            overlap_param_gather=False,  # Verify that this works using
            grad_reduce_in_fp32=True,
            align_param_gather=False,
            average_in_collective=True,
        )
        strategy = MegatronStrategy(
            ddp=ddp,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            pipeline_dtype=torch.float32,
            sequence_parallel=False,
            ckpt_load_optimizer=True,
            ckpt_save_optimizer=True,
            ckpt_async_save=False,
            save_ckpt_format="torch_dist",
            ckpt_load_strictness="log_all",
        )

        trainer = nl.Trainer(
            devices=1,
            max_steps=cls.num_steps,
            num_nodes=1,
            accelerator="gpu",
            strategy=strategy,
            limit_val_batches=cls.limit_val_batches,
            val_check_interval=cls.val_check_interval,
            log_every_n_steps=cls.val_check_interval,
            callbacks=list(cls.callbacks[mode].values()),
            plugins=nl.MegatronMixedPrecision(precision=cls.precision),
        )
        return trainer

    @override
    @classmethod
    def setup_class(cls):
        super().setup_class()

        # setup data
        cls.tokenizer = get_nmt_tokenizer("byte-level")
        # run stop and go
        cls.run_stop_and_go()

    @classmethod
    def setup_model(cls, mode: Mode) -> tuple[pl.LightningModule, pl.LightningDataModule, nl.MegatronOptimizerModule]:
        # build data module
        data = MockDataModule(
            seq_length=cls.seq_length,
            micro_batch_size=cls.micro_batch_size,
            global_batch_size=cls.global_batch_size,
            num_workers=cls.workers,
            tokenizer=cls.tokenizer,
        )

        data.init_global_step = 0
        # config
        config = HyenaNV1bConfig(
            **{
                "tp_comm_overlap": cls.use_megatron_comm_overlap_llama3_8k,
                "seq_length": cls.seq_length,
                "use_te": False,  # TODO: stop and go harness doesn't work with TE, since somehow query.dtype is torch.float32 instead of bfloat16
                "params_dtype": torch.bfloat16,
                "bf16": True,
                "recompute_granularity": None,
                "recompute_method": None,
                "recompute_num_layers": None,
                "hidden_size": 1920,
                "num_attention_heads": 15,
                "num_query_groups": 15,
                "ffn_hidden_size": 5120,
                "hidden_dropout": cls.hidden_dropout,
                "num_groups_hyena": 1920,
                "num_groups_hyena_medium": 128,
                "num_groups_hyena_short": 128,
                "attention_dropout": cls.attention_dropout,
                "to_upper": "weighted" if cls.no_renormalize_loss else "normalized_weighted",
                "distribute_saved_activations": False if cls.sequence_parallel else True,
                "cross_entropy_loss_fusion": cls.cross_entropy_loss_fusion,
                "fp32_residual_connection": not cls.no_fp32_residual_connection,
                "add_bias_output": cls.add_bias_output,
            }
        )

        optimizer_config = OptimizerConfig(
            optimizer="adam",
            lr=cls.lr,
            adam_beta1=0.9,
            adam_beta2=0.95,
            weight_decay=cls.wd,
            clip_grad=cls.clip_grad,
            params_dtype=torch.bfloat16,
            use_distributed_optimizer=True,
            bf16=True,
        )
        # build optimizer
        optimizer = MegatronOptimizerModule(
            config=optimizer_config,
            lr_scheduler=CosineAnnealingScheduler(warmup_steps=1, max_steps=cls.num_steps, min_lr=3e-5),
            no_weight_decay_cond=config.hyena_no_weight_decay_cond_fn,
        )

        # # Build model
        module = HyenaModel(config, tokenizer=data.tokenizer)
        # import pdb; pdb.set_trace()
        return module, data, optimizer


# class TestEvo2StopAndGoCheckpointNotAtValidation(TestEvo2StopAndGo):
#     @override
#     @classmethod
#     def get_default_callbacks(cls):
#         callbacks = super().get_default_callbacks()
#         callbacks[Mode.STOP][nl_callbacks.PreemptionCallback] = nl_callbacks.PreemptionCallback(sig=signal.SIGUSR2)
#         callbacks[Mode.STOP][testing_callbacks.SignalAfterGivenStepCallback] = (
#             testing_callbacks.SignalAfterGivenStepCallback(stop_step=2, signal_=signal.SIGUSR2)
#         )

#         return callbacks

#     @override
#     @classmethod
#     def stop(cls) -> None:
#         # The PreemptionCallback exits the process with sys.exit(0) after the checkpoint is saved. We obviously don't
#         # want that here, so we catch the SystemExit exception and make sure it was called appropriately.
#         with pytest.raises(SystemExit) as pytest_wrapped_e:
#             super().stop()

#         assert pytest_wrapped_e.type is SystemExit
#         assert pytest_wrapped_e.value.code == 0

# @pytest.mark.parametrize(
#     "callback_type",
#     [
#         testing_callbacks.LearningRateCallback,
#         testing_callbacks.GlobalStepStateCallback,
#         testing_callbacks.ConsumedSamplesCallback,
#         testing_callbacks.OptimizerStateCallback,
#         testing_callbacks.TrainInputCallback,
#         testing_callbacks.TrainOutputCallback,
#         testing_callbacks.TrainLossCallback,
#         testing_callbacks.ValidInputCallback,
#         testing_callbacks.ValidOutputCallback,
#         testing_callbacks.ValidLossCallback,
#     ],
# )
# def test_stop_and_go_consistency(self, callback_type):
#     if callback_type in [
#         testing_callbacks.ValidInputCallback,
#         testing_callbacks.ValidLossCallback,
#         testing_callbacks.ValidOutputCallback,
#     ]:
#         # On resumption from a checkpoint that wasn't created at the end of validation, the validation interval is
#         # shifted in the subsequent training jobs. See this slack thread for more details:
#         # https://nvidia.slack.com/archives/C074Z808N05/p1733171223813409
#         pytest.xfail(
#             reason="Currently seeing issues in validation timing with PreemptionCallback. "
#             "See https://nvbugspro.nvidia.com/bug/4994415F."
#         )
#     super().test_stop_and_go_consistency(callback_type)

# @pytest.mark.skip(reason="We don't expect the STOP variant to hit on_valid_epoch_end before stopping.")
# def test_train_val_init_consumed_samples(self):
#     pass

# def test_all_valid_batch_inputs_are_identical(self):
#     """A watered-down version of test_stop_and_go_consistency's ValidInputCallback that only checks whether the
#     first batches are the same, not the over length."""

#     valid_inputs_interrupted = stop_and_go.get_callback(
#         self.callbacks, Mode.RESUME, testing_callbacks.ValidInputCallback
#     ).data
#     valid_inputs_continuous = stop_and_go.get_callback(
#         self.callbacks, Mode.CONTINUOUS, testing_callbacks.ValidInputCallback
#     ).data

#     min_len = min(len(valid_inputs_interrupted), len(valid_inputs_continuous))
#     assert min_len
#     recursive_assert_approx_equal(valid_inputs_interrupted[:min_len], valid_inputs_continuous[:min_len])

# def test_train_val_init_consumed_samples(self):
#     """Tests the initial consumed samples in stop-and-go scenario."""
#     train_consumed_stop, val_consumed_stop = stop_and_go.get_callback(
#         self.callbacks, Mode.STOP, testing_callbacks.TrainValInitConsumedSamplesStopAndGoCallback
#     ).data
#     train_consumed_go, val_consumed_go = stop_and_go.get_callback(
#         self.callbacks, Mode.RESUME, testing_callbacks.TrainValInitConsumedSamplesStopAndGoCallback
#     ).data

#     assert val_consumed_stop == 0
#     assert val_consumed_go == 0
#     assert train_consumed_stop == 0
#     assert train_consumed_go > 0
