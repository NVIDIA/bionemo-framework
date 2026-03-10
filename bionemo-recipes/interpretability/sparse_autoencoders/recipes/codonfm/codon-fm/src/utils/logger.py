import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict


from lightning.pytorch.loggers import WandbLogger
import nemo_run as run
import yaml
from nemo_run.core.serialization.yaml import YamlSerializer

import logging

log = logging.getLogger(__name__)

def wandb_logger(project: str,
                 name: str,
                 output_dir:str,
                 entity: Optional[str] = None,
                 offline: bool = False,
                 log_model: bool = False,
                 tags: List[str] = [],
                 config: Optional[Dict] = None) -> run.Config[WandbLogger]:
    cfg = run.Config(
        WandbLogger,
        project=project,
        name=name,
        save_dir=output_dir,
        offline=offline,
        log_model=log_model,
        tags=tags,
        config=config,
    )
    
    if entity:
        cfg.entity = entity
    return cfg


@dataclass(kw_only=True)
class WandbPlugin(run.Plugin):
    """
    ## Adapted from nemo-lightning
    A plugin for setting up Weights & Biases.

    This plugin sets a  the Pytorch Lightning `WandbLogger <https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.WandbLogger.html>`_.

    This plugin is only activated if the ``WANDB_API_KEY`` environment variable is set.
    The ``WANDB_API_KEY`` environment variables will also be set in the executor's environment variables.
    Follow https://docs.wandb.ai/quickstart to retrieve your ``WANDB_API_KEY``.

    If `log_task_config` is True, the plugin will log the task configuration as a config dictionary
    to the Weights and Biases logger. This is useful for tracking the task configuration and experiments reproducibility.

    Args:
        name (str): The name for the Weights & Biases run.
        logger_fn (Callable[..., run.Config[WandbLogger]]): A callable that returns a Config of ``WandbLogger``
        log_task_config (bool, optional): Whether to log the task configuration to the logger.
            Defaults to True.

    Raises:
        logging.warning: If the task is an instance of `run.Script`, as the plugin has no effect on such tasks.
    """

    name: str
    logger_fn: run.Config[WandbLogger]
    log_task_config: bool = True

    def setup(self, task: run.Partial | run.Script, executor: run.Executor):
        if isinstance(task, run.Script):
            log.info(
                f"The {self.__class__.__name__} will have no effect on the task as it's an instance of run.Script"
            )
            return

        if "WANDB_API_KEY" in os.environ:
            executor.env_vars["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]

            if hasattr(task, "log"):
                task.log = self.logger_fn
                if self.log_task_config:
                    partial_config = yaml.safe_load(YamlSerializer().serialize(task))
                    partial_config["experiment"] = {
                        "task_name": self.name,
                        "executor": executor.info(),
                        "remote_directory": (
                            os.path.join(executor.tunnel.job_dir, Path(executor.job_dir).name)
                            if isinstance(executor, run.SlurmExecutor)
                            else None
                        ),
                        "local_directory": executor.job_dir,
                    }
                    task.log.config = partial_config
        else:
            log.info(
                f"The {self.__class__.__name__} will have no effect as WANDB_API_KEY environment variable is not set."
            )
