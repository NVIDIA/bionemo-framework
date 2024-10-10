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


from typing import Any, Tuple

from pytorch_lightning import Callback, LightningModule, Trainer


class ParameterMultiplicativeScheduler(Callback):
    """
    A callback that multiplies the parameter by a factor at each step.
    """

    @property
    def state_key(self):
        return f"ParameterMultiplicativeScheduler[module_parameter_path={self.module_parameter_path}]"

    def __init__(
        self,
        module_parameter_path: str,
        factor: float,
        min_multiplier=0.0,
        max_multiplier=1.0,
        enable_progress_bar: bool = False,
    ):
        """
        Args:
            parameter: the parameter to be multiplied
            factor: the factor to multiply the parameter by
        """
        super().__init__()
        self.module_parameter_path = module_parameter_path
        self.factor = factor
        self.min_multiplier = min_multiplier
        self.max_multiplier = max_multiplier
        self.enable_progress_bar = enable_progress_bar
        self.state = {"multiplier": min_multiplier, "parameter": None}

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        super().setup(trainer, pl_module, stage)
        if self.state["parameter"] is None:
            # Initialize the parameter if it is None (eg not loaded from a checkpoint)
            self.state["parameter"] = self.get_attr(pl_module, self.module_parameter_path)
        self.update_parameter(pl_module, log=False)  # Don't log in setup.

    def get_attr(self, pl_module: LightningModule, path: str) -> Any:
        """Get the attribute at the end of the path within pl_module.

        Args:
            pl_module (LightningModule): LightningModule to get the attribute from
            path (str): Path to follow within the module to get the attribute

        Returns:
            Any: desired attribute
        """
        attr, last_ptr = self.get_attr_ptr_prefix(pl_module, path)
        return getattr(attr, last_ptr)

    def get_attr_ptr_prefix(self, pl_module: LightningModule, path: str) -> Tuple[Any, str]:
        """Get a pointer to the owner of the attribute at the end of the path within pl_module, and the last attribute name.

        Args:
            pl_module (LightningModule): LightningModule to get the last owning object of the attribute from
            path (_type_): Path to follow within the module to get the attribute, and the final part of the path

        Returns:
            Tuple[Any, str]: _description_
        """
        attr = pl_module  # Start at the module, and follow the path to the second to last attribute
        path_parts = path.split(".")
        last_ptr = path_parts[-1]
        for ptr in path_parts[:-1]:
            attr = getattr(attr, ptr)
        return attr, last_ptr

    def set_attr(self, pl_module: LightningModule, path: str, value: Any) -> None:
        """Set the attribute at the end of the path within pl_module.

        Args:
            pl_module (LightningModule): LightningModule to set the attribute in
            path (str): Path to follow within the module to set the attribute
            value (Any): Value to set the attribute to
        """
        attr, last_ptr = self.get_attr_ptr_prefix(pl_module, path)
        setattr(attr, last_ptr, value)

    def step_multiplier(self):
        """
        Updates the state of the callback
        """
        self.state["multiplier"] = min(self.max_multiplier, self.state["multiplier"] + self.factor)

    def update_parameter(self, pl_module: LightningModule, log=True):
        """
        Updates the parameter and logs the new value
        """
        # This update may happen independently on every device and the state will still be correct. The same computation
        #  will happen on each device as a function of the number of steps on that device, which is the same as the global
        #  behavior we desire.
        new_value = self.state["multiplier"] * self.state["parameter"]
        self.set_attr(pl_module, self.module_parameter_path, new_value)
        if log:
            pl_module.log(
                self.module_parameter_path,
                new_value,
                sync_dist=False,  # We do not need to sync these calls, there is no state that needs to be reduced.
                batch_size=pl_module.cfg.micro_batch_size,
                prog_bar=self.enable_progress_bar,
            )

    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
        """
        Multiplies the parameter by the factor at each step.
        """
        self.update_parameter(pl_module)
        self.step_multiplier()

    def load_state_dict(self, state_dict):
        self.state.update(state_dict)

    def state_dict(self):
        return self.state.copy()
