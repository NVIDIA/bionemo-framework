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
import pathlib
from typing import Any, Dict, Optional, Sequence, TypedDict

from nemo.lightning.nemo_logger import NeMoLogger
from nemo.utils import logging
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger


__all__: Sequence[str] = (
    "WandbLoggerOptions",
    "setup_nemo_lightning_logger",
)


class WandbLoggerOptions(TypedDict):
    """Note: `name` controls the exp name is handled by the NeMoLogger so it is ommitted here.
    `directory` is also omitted since it is set by the NeMoLogger.
    """

    offline: bool  # offline mode
    project: str  # project name
    entity: str  # group name or user name
    # name: str # experiment name, this is handled by NeMoLogger
    # the directory is also set by NeMoLogger
    log_model: bool  # log model


def setup_nemo_lightning_logger(
    name: str = "default-name",
    root_dir: str = "./results",
    initialize_tensorboard_logger: bool = False,
    wandb_kwargs: Optional[WandbLoggerOptions] = None,
    **kwargs: Dict[str, Any],
) -> NeMoLogger:
    """Setup the logger for the experiment.

    Args:
        name: The name of the experiment. Results go into `root_dir`/`name`
        root_dir: The root directory to create the `name` directory in for saving run results.
        initialize_tensorboard_logger: Whether to initialize the tensorboard logger.
        wandb_kwargs: The kwargs for the wandb logger.
        **kwargs: The kwargs for the NeMoLogger.

    Returns:
        NeMoLogger: NeMo logger instance.
    """
    # The directory that the logger will save to
    save_dir = pathlib.Path(root_dir) / name
    if wandb_kwargs is not None:
        wandb_logger = WandbLogger(save_dir=save_dir, name=name, **wandb_kwargs)
    else:
        wandb_logger = None
        logging.warning("WandB is currently turned off.")
    if initialize_tensorboard_logger:
        tb_logger = TensorBoardLogger(save_dir=save_dir, name=name)
    else:
        tb_logger = None
        logging.warning("User-set tensorboard is currently turned off. Internally one may still be set by NeMo2.")
    logger: NeMoLogger = NeMoLogger(
        dir=root_dir,
        name=name,
        tensorboard=tb_logger,
        wandb=wandb_logger,
        **kwargs,
    )
    # Needed so that the trainer can find an output directory for the profiler
    logger.save_dir = save_dir
    return logger
