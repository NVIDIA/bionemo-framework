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


import os
import sys

from nemo.core import ModelPT
from nemo.utils import logging


DEFAULT_TAG = "******** bnmo_debug ********"


def log_with_nemo_at_level(
    input_string: str, model_pt: ModelPT = None, tag: str = DEFAULT_TAG, level=logging.DEBUG
) -> str:
    frame_of_caller = sys._getframe(1)
    output_string = f"""
        {input_string}
        frame_of_caller.f_code.co_name={frame_of_caller.f_code.co_name}
        frame_of_caller.f_code.co_filename={frame_of_caller.f_code.co_filename}
        frame_of_caller.f_lineno={frame_of_caller.f_lineno}
    """

    if model_pt:
        # Neither model_pt or model_pt.trainer contains step
        output_string = f"""
        {output_string}
        model_pt.trainer.global_step={model_pt.trainer.global_step}
        model_pt.trainer.max_steps={model_pt.trainer.max_steps}
        model_pt.trainer.current_epoch={model_pt.trainer.current_epoch}
        model_pt.trainer.max_epochs={model_pt.trainer.max_epochs}
        model_pt.trainer.global_rank={model_pt.trainer.global_rank}
        """

    if level == logging.WARNING:
        logging.warning(prefix_string_with_tag(output_string, tag))

    elif level == logging.DEBUG:
        logging.debug(prefix_string_with_tag(output_string, tag))

    elif level == logging.INFO:
        logging.info(prefix_string_with_tag(output_string, tag))

    else:
        raise NotImplementedError


def prefix_string_with_tag(input_string: str, tag: str = DEFAULT_TAG) -> str:
    return f"{tag}, {input_string}"


def environ_as_multiline_str() -> str:
    return "\n".join([str(pair[0]) + ":" + str(pair[1]) for pair in os.environ.items()])
