# Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

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

from typing import List, Optional, Union

from pytorch_lightning import Callback

import bionemo.utils.callbacks


__all__ = ["setup_callbacks"]


def _instantiate_callback(callback_cfg, parent_cfg, plugins: Optional[List] = None) -> Callback:
    """
    Instantiates callback

    Params
        callback_cfg: Config for callback to be created with 'class' attribute
        parent_cfg: Global cfg dict
        plugins: Optional plugins to be passed to callback

    Returns
        Instantiated class given in callback_cfg
    """
    callback_class = callback_cfg['class']
    cb = lookup_or_use(bionemo.utils.callbacks, callback_class, callback_cfg, parent_cfg, plugins)
    return cb


def _select_validation_callbacks(cfg) -> List:
    """
    Selects validation callbacks from config

    Params
        cfg: Dict

    Returns:
        List of selected validation callback config dicts
    """
    valid_cbs = []
    if 'dwnstr_task_validation' in cfg.model and cfg.model.dwnstr_task_validation['enabled']:
        valid_cbs = [cfg.model.dwnstr_task_validation.dataset]
    return valid_cbs


def setup_callbacks(cfg, plugins: Optional[List] = None) -> List:
    """
    Sets up callbacks from cfg

    Params
        cfg: Dict
        plugins: Optional plugins to be passed to callbacks

    Returns
        List of Callbacks to be passed into Trainer
    """
    callbacks = []
    callbacks.extend(_select_validation_callbacks(cfg))

    callbacks = [_instantiate_callback(cb, cfg, plugins) for cb in callbacks]
    return callbacks


def lookup_or_use(obj: object, attr: Union[str, object], *args, **kwargs):
    """Looks up an object from a module or returns the attribute if it is not a string"""
    return getattr(obj, attr)(*args, **kwargs) if isinstance(attr, str) else attr
