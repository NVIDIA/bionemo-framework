from typing import List

from omegaconf import OmegaConf
from pytorch_lightning import Callback

from bionemo.tests.callbacks.callbacks import PerfLoggingCallback, SaveTrainerFinalMetricCallback
from bionemo.tests.callbacks.dllogger import DLLogger


def add_test_callbacks(cfg: OmegaConf, callbacks: List[Callback], mode: str = "train"):
    """
    Appends testing-related callbacks to the list with PL Trainer's callbacks
    Args:
        cfg: config
        mode: str attached to the performance logs
    """

    if cfg.get("create_dllogger_callbacks", False):
        if 'batch_size' in cfg.model.data and cfg.model.data.batch_size != cfg.model.micro_batch_size:
            raise ValueError("cfg.model.data.batch_size should be equal to cfg.model.micro_batch_size")
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
