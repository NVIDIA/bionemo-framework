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


from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence

from nemo.utils.model_utils import import_class_by_path
from omegaconf import DictConfig
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ProgressBar

from bionemo.callbacks.bnmo_progress_bar import BnmoTQDMProgressBar
from bionemo.callbacks.logging_callbacks import PerfLoggingCallback, SaveTrainerFinalMetricCallback
from bionemo.callbacks.metric_collector import MetricCollector
from bionemo.callbacks.testing_callbacks import (
    KillAfterSignalCallback,
    MetadataSaveCallback,
    TestCheckpointIntegrityCallback,
)
from bionemo.utils.dllogger import DLLogger


CREATE_METRIC_COLLECTOR = "create_metric_collector"
METRIC_COLLECTOR_KWARGS = "metrics_collector_kwargs"


__all__: Sequence[str] = (
    "add_test_callbacks",
    "add_training_callbacks",
    "add_progress_bar_callback",
    "setup_dwnstr_task_validation_callbacks",
)


def add_test_callbacks(cfg: DictConfig, callbacks: List[Callback], mode: str = "train"):
    """
    Appends testing-related callbacks to the list with PL Trainer's callbacks
    Args:
        cfg: config
        mode: str attached to the performance logs
    """
    if cfg.get("create_dllogger_callbacks", False):
        dllogger_callbacks_kwargs = cfg.get("dllogger_callbacks_kwargs", {})
        json_file = dllogger_callbacks_kwargs.get("json_file", "dllogger.json")
        append_to_json = dllogger_callbacks_kwargs.get("append_to_json", True)
        use_existing_dllogger = dllogger_callbacks_kwargs.get("use_existing_dllogger", False)
        warmup = dllogger_callbacks_kwargs.get("warmup", 0)

        dllogger = DLLogger(
            json_file=json_file, append_to_json=append_to_json, use_existing_dllogger=use_existing_dllogger
        )
        callbacks.append(
            PerfLoggingCallback(
                dllogger=dllogger, global_batch_size=cfg.model.global_batch_size, mode=mode, warmup=warmup
            )
        )

    if cfg.get("create_trainer_metric_callback", False):
        trainer_metric_callback_kwargs = cfg.get("trainer_metric_callback_kwargs", {})
        callbacks.append(SaveTrainerFinalMetricCallback(**trainer_metric_callback_kwargs))

    if cfg.get("create_kill_after_signal_callback", False):
        kill_after_signal_callback_kwargs = cfg.get("kill_after_signal_callback_kwargs", {})
        callbacks.append(KillAfterSignalCallback(**kill_after_signal_callback_kwargs))

    if cfg.get("create_metadata_save_callback", False):
        metadata_save_callback_kwargs = cfg.get("metadata_save_callback_kwargs", {})
        callbacks.append(MetadataSaveCallback(**metadata_save_callback_kwargs))

    if cfg.get("create_checkpoint_integrity_callback", False):
        checkpoint_integrity_callback_kwargs = cfg.get("checkpoint_integrity_callback_kwargs", {})
        callbacks.append(TestCheckpointIntegrityCallback(**checkpoint_integrity_callback_kwargs))


def _select_dwnstr_task_validation_callbacks(cfg: DictConfig) -> List[DictConfig]:
    """
    Selects configuration of validation callbacks from a main training config

    Params
        cfg: a main configuration of training used in training scripts

    Returns: List of selected validation callback config dicts
    """
    assert "model" in cfg, " The 'model' key is not present in the supplied cfg"
    valid_cbs = []
    if "dwnstr_task_validation" in cfg.model and cfg.model.dwnstr_task_validation["enabled"]:
        valid_cbs = [cfg.model.dwnstr_task_validation.dataset]

    return valid_cbs


def setup_dwnstr_task_validation_callbacks(cfg: DictConfig, plugins: Optional[List[Any]] = None) -> List[Callback]:
    """
    Sets up callbacks for short downstream tasks fine-tunings at the end of the main training validation loop.
    The configuration of callbacks is taken from the main training config.

    Params
        cfg: Dict
        plugins: Optional plugins to be passed to callbacks

    Returns
        List of callbacks to be passed into plt.Trainer
    """
    callbacks_cfg = []
    callbacks_cfg.extend(_select_dwnstr_task_validation_callbacks(cfg))

    callbacks = [
        import_class_by_path(callback_cfg["class"])(callback_cfg, cfg, plugins) for callback_cfg in callbacks_cfg
    ]
    return callbacks


def add_training_callbacks(cfg: DictConfig, callbacks: List[Callback]) -> List[Callback]:
    """
    Sets up callbacks for training loop.
    The configuration of callbacks is taken from the main training config.

    Params
        cfg: Dict
        plugins: Optional plugins to be passed to callbacks

    """
    if "model" in cfg and "training_callbacks" in cfg.model:
        callbacks_cfg: List[Dict[str, Any]] = cfg.model.training_callbacks
        for callback_cfg in callbacks_cfg:
            callback_kwargs = dict(deepcopy(callback_cfg))
            callback_cls = callback_kwargs.pop("class")
            callbacks.append(import_class_by_path(callback_cls)(**callback_kwargs))

    if CREATE_METRIC_COLLECTOR in cfg:
        if METRIC_COLLECTOR_KWARGS in cfg:
            kwargs = dict(deepcopy(cfg.get(METRIC_COLLECTOR_KWARGS)))
        else:
            kwargs = {}
        callbacks.append(MetricCollector(**kwargs))


def add_progress_bar_callback(cfg: DictConfig, callbacks: List[Callback]) -> List[Callback]:
    """Add a customized progressbar object to the list of callbacks passed
    to the trainer initialization method.

    The object added to the callbacks list must be an instance of a class
    that inherits from the pytorch_lightning.callbacks.progress.ProgressBar.
    Trainer has internal logic, such that if the callbacks list contains an
    instance of ProgressBar, then it is used as the progress bar; otherwise,
    an instance of TQDMProgressBar is used.

    Args:
        cfg: the top-level cfg passed to main
        callbacks: a list of pytorch_lightning.callbacks.Callback instances

    Returns:
        The updated callbacks list

    """
    if "progress_bar_kwargs" in cfg and cfg.progress_bar_kwargs.get("name", None) == "BnmoTQDMProgressBar":
        # if there is a ProgressBar instance in callbacks, remove it
        num_callbacks = len(callbacks)
        for i in range(num_callbacks - 1, -1, -1):
            if isinstance(callbacks[i], ProgressBar):
                callbacks.pop(i)

        # add BnmoTQDMProgressBar
        callbacks.append(BnmoTQDMProgressBar(warmup_n=cfg.progress_bar_kwargs.get("warmup", 1)))

    return callbacks
