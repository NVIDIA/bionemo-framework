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


from dataclasses import fields

import numpy as np
import pytorch_lightning as pl
import torch
from megatron.core import ModelParallelConfig
from megatron.core.enums import ModelType
from nemo.core.config import hydra_runner
from nemo.utils import logging
from omegaconf.omegaconf import OmegaConf

from bionemo.model.singlecell.geneformer.model import GeneformerModel
from bionemo.model.utils import setup_trainer


class DummyBaseline(GeneformerModel):
    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        lm_labels=None,
        checkpoint_activations_all_layers=None,
        model=None,
    ):
        if model is None:
            model = self.model
        positions = torch.arange(input_ids.size(1), device=input_ids.device, dtype=input_ids.dtype).repeat(
            input_ids.size(0), 1
        )
        lm_logits = model(positions, input_ids)
        if lm_labels is not None:
            # lm_labels[lm_labels==-1] = 0
            lm_loss = torch.nn.functional.cross_entropy(
                lm_logits.view(-1, lm_logits.shape[-1]), lm_labels.view(-1), reduction="none"
            ).view(lm_labels.shape)
            output_tensor = (lm_loss, None)
        else:
            output_tensor = lm_logits

        return output_tensor


class ConfigBackedModel(torch.nn.Module):
    def __init__(self, cfg, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.torch_dtype = torch.float32
        self.params_dtype = torch.float32
        self.autocast_dtype = torch.float32
        self.model_type = ModelType.encoder_or_decoder
        self.config = self.build_model_parallel_config()

    def set_input_tensor(self, input_tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor

    def build_model_parallel_config(self) -> ModelParallelConfig:
        """For attributes in the nemo model config that are the same as the
        megatron core ModelParallelConfig we will use the value from the nemo config.
        For attributes in ModelParallelConfig that are not in the nemo model config, we add custom logic.
        """
        cfg = OmegaConf.to_container(self.cfg, resolve=True)

        # map precision related configs
        cfg.get("precision", 32)  # PTL trainer precision
        megatron_amp_O2 = cfg.get("megatron_amp_O2", False)

        # dtype used in p2p communication
        pipeline_dtype = self.torch_dtype

        # maps NeMo model configs to ModelParallelConfig from megatron core
        config_mapping = {
            "perform_initialization": True,  # initailize weights when constructing the module
            "fp16": self.torch_dtype == torch.float16
            and megatron_amp_O2,  # NeMo does not currently support fp16 training with megatron amp O2, eval and inference is supported
            "bf16": self.torch_dtype == torch.bfloat16 and megatron_amp_O2,
            "params_dtype": self.params_dtype,
            "timers": None,  # NeMo does not currently support megatron core timers
            "async_tensor_model_parallel_allreduce": self.cfg.get("tensor_model_parallel_world_size", 1) > 1
            and not self.cfg.get("sequence_parallel", False),
            "pipeline_dtype": pipeline_dtype,
            "grad_scale_func": self.trainer.precision_plugin.scaler.scale
            if self.torch_dtype == torch.float16
            else None,
            "enable_autocast": not megatron_amp_O2 and self.torch_dtype in [torch.bfloat16, torch.float16],
            "autocast_dtype": self.autocast_dtype,
            "variable_seq_lengths": False,  # set dynamically during training
            "num_microbatches_with_partial_activation_checkpoints": self.cfg.get(
                "num_micro_batches_with_partial_activation_checkpoints", None
            ),
            "batch_p2p_sync": True,  # call torch.cuda.synchronize() after batch isend/rcv
            "use_ring_exchange_p2p": False,  # not supported in NeMo
            "deallocate_pipeline_outputs": False,  # not supported in NeMo
            "no_sync_func": None,  # set dynamically during training
            "grad_sync_func": None,  # set dynamically during training
            "param_sync_func": None,  # set dynamically during training
        }

        # instantitate ModelParallelConfig from this dict
        mp_config_dict = {}

        for field in fields(ModelParallelConfig):
            # model config has priority
            if field.name in cfg:
                mp_config_dict[field.name] = cfg[field.name]
            # then config_mapping
            elif field.name in config_mapping:
                mp_config_dict[field.name] = config_mapping[field.name]
            else:
                logging.warning(
                    f"The model: {self} does not have field.name: {field.name} in its cfg. "
                    f"Add this key to cfg or config_mapping to make to make it configurable."
                )

        model_parallel_config = ModelParallelConfig(**mp_config_dict)

        try:
            # hidden size is needed for pipeline schedules but is not currently in ModelParallelConfig
            setattr(model_parallel_config, "hidden_size", self.cfg.hidden_size)
        except AttributeError:
            logging.warning(
                f"hidden_size not found in {self.cfg}. Set this in model_parallel_config if using pipeline parallelism."
            )

        return model_parallel_config


class PosModel(ConfigBackedModel):
    def __init__(self, cfg, seq_length, vocab_size, position_group_size: int = 1):
        super().__init__(cfg)
        self.position_group_size = position_group_size
        self.pos_model = torch.nn.Embedding(seq_length // position_group_size, vocab_size)
        torch.nn.init.normal_(self.pos_model.weight, mean=0, std=0.02)

    def forward(self, positions, tokens):
        return self.pos_model(positions // self.position_group_size)


class PosTokenModel(ConfigBackedModel):
    def __init__(self, cfg, seq_length, vocab_size, position_group_size: int = 1):
        super().__init__(cfg)
        self.position_group_size = position_group_size
        self.pos_model = torch.nn.Embedding(seq_length // position_group_size, vocab_size)
        self.token_pos_add = torch.nn.Parameter(
            torch.zeros(1, seq_length // position_group_size, 1, dtype=torch.float32, requires_grad=True)
        )
        self.token_add = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32, requires_grad=True))
        torch.nn.init.normal_(self.pos_model.weight, mean=0, std=0.02)
        torch.nn.init.normal_(self.token_pos_add, mean=0, std=0.005)
        torch.nn.init.constant_(self.token_add, 0.02)

    def forward(self, positions, tokens):
        not_mask = tokens != 0
        pos_pred = self.pos_model(positions // self.position_group_size)
        short_token_bias = torch.nn.functional.leaky_relu(self.token_add) + self.token_pos_add
        full_token_bias = torch.repeat_interleave(short_token_bias, self.position_group_size, dim=1)[
            :, : tokens.shape[1]
        ]
        return pos_pred + full_token_bias * torch.nn.functional.one_hot(tokens, num_classes=pos_pred.shape[-1]).to(
            dtype=pos_pred.dtype, device=pos_pred.device
        ) * not_mask.unsqueeze(-1)


@hydra_runner(config_path="../conf", config_name="geneformer_config")
def main(cfg) -> None:
    """
    Main function for pretraining the Geneformer model.

    Args:
        cfg (OmegaConf): Configuration object containing the experiment settings.

    Returns:
        None
    """
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")
    if cfg.get("seed_everything", True):
        np.random.seed(cfg.model.seed)
        pl.seed_everything(cfg.model.seed)

    trainer = setup_trainer(cfg)
    baseline_method = cfg.get("baseline_method", None)
    _model = GeneformerModel(cfg.model, trainer)
    assert baseline_method is not None, "baseline_method must be set"
    position_group_size: int = cfg.model.get("position_group_size", 1)
    if baseline_method == "pos":
        inner_model = PosModel(
            cfg.model, cfg.model.data.seq_length, _model.tokenizer.vocab_size, position_group_size=position_group_size
        )
    elif baseline_method == "pos_token":
        inner_model = PosTokenModel(
            cfg.model, cfg.model.data.seq_length, _model.tokenizer.vocab_size, position_group_size=position_group_size
        )
    else:
        raise ValueError(
            f"Unknown baseline method: {baseline_method}, currently we support either 'pos' or 'pos_token'"
        )
    logging.info(f"************** Using DummyBaseline with {baseline_method} ***********")
    model = DummyBaseline(inner_model, cfg.model, trainer)

    logging.info("************** Starting Training ***********")
    trainer.fit(model)
    logging.info("*************** Finish Training ************")


if __name__ == "__main__":
    main()
