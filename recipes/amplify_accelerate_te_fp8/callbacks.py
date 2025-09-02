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

from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments


class StopAfterNStepsCallback(TrainerCallback):
    """Callback to interrupt training after a specified number of steps.

    This allows us to use a learning rate scheduler consistent with the full training run while
    stopping after a pre-determined number of steps.
    """

    def __init__(self, max_steps: int):
        """Initialize the callback."""
        self.max_steps = max_steps

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Interrupt training after a specified number of steps."""
        if state.global_step >= self.max_steps:
            control.should_training_stop = True
