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


#
# title: nemo_utils.py
# description:
#   - NeMo supports experiment management, which includes checkpoint management.
#   Separate checkpoint files for model and EMA are saved at each call to
#   validation_step.
#   - Expected behavior, is that the checkpoint files writte at validation_step()
#   are tagged with 'last' in the filename, and previous filenames with 'last'
#   are deleted.
#   - The expected behavior can fail, somehow, and multiple checkpoint filenames
#   with 'last' appear in the 'artifacts/checkpoints' directory.  This failure
#   causes the application to crash
#       - Excecption handling is challenged as there is related logic in
#       exp_manager from NeMo.
#   - To address, the function isolate_last_checkpoint(.) is called before
#   the Lightning Trainer object is configured, to make sure there is at moste
#   one checkpoint of each type (model and optimizer) with tag 'last'.
#
#

import os
from time import sleep
from typing import Iterable, List, Tuple

from omegaconf import DictConfig

from bionemo.utils.logging_utils import log_with_nemo_at_level


def isolate_last_checkpoint(entire_cfg: DictConfig) -> Tuple[List[str], List[str], int]:
    """
        - NeMo supports experiment management, which includes checkpoint management.
        Separate checkpoint files for model and EMA are saved at each call to
        validation_step.

        - Expected behavior, is that the checkpoint files writte at validation_step()
        are tagged with 'last' in the filename, and previous filenames with 'last'
        are deleted.

        - The expected behavior can fail, somehow, and multiple checkpoint filenames
        with 'last' appear in the 'artifacts/checkpoints' directory.  This failure
        causes the application to crash

        - To address, the function isolate_last_checkpoint(.) is called before
        the Lightning Trainer object is configured, to make sure there is at moste
        one checkpoint of each type (model and optimizer) with tag 'last'.

        - There have been reported issues associated with error messages observed,
        so we put a sleep statement after filesystem operations.
        https://stackoverflow.com/questions/71617570/pytorchstreamreader-failed-reading-zip-archive-failed-finding-central-directory

    Args:
        entire_cfg (DictConfig): _description_

    Returns:
        Tuple[Union[int, None], Union[int, None]]:
        ( number of 'last' checkpoints to to keep, number of 'last' checkpoints to remove)
    """
    # (0) corner cases
    assert "exp_manager" in entire_cfg
    assert "exp_dir" in entire_cfg.get("exp_manager")

    checkpoint_dir = os.path.join(entire_cfg.get("exp_manager").get("exp_dir"), "checkpoints")
    if not os.path.isdir(checkpoint_dir):
        return [], []

    listdir_checkpoint_dir = os.listdir(checkpoint_dir)
    log_with_nemo_at_level(
        f"""
        isolate_last_checkpoint(), begin
        checkpoint_dir={checkpoint_dir}
        listdir_checkpoint_dir={listdir_checkpoint_dir}
        """
    )

    # (1) view contents of dir, and determine which to keep
    base_filenames_before_remove = [
        x for x in listdir_checkpoint_dir if os.path.isfile(os.path.join(checkpoint_dir, x))
    ]
    filenames_to_keep, filenames_to_rename = isolate_filenames_with_last_to_keep(base_filenames_before_remove)

    # (2) remove unwanted checkpoint files with 'last' tag
    for base_filename_with_last in filenames_to_rename:
        full_path_with_last = os.path.join(checkpoint_dir, base_filename_with_last)
        full_path_renamed = os.path.join(checkpoint_dir, base_filename_with_last.replace("last", "bnmorecent"))

        log_with_nemo_at_level(
            f"""
            isolate_last_checkpoint(), begin
            Rename checkpoint: {full_path_with_last}
            To: {full_path_renamed}
            """
        )
        os.rename(full_path_with_last, full_path_renamed)
    sleep(2)  # wait for possibly slow file io
    assert len(filenames_to_keep) <= 2

    return filenames_to_keep, filenames_to_rename


def isolate_filenames_with_last_to_keep(base_filenames: Iterable[str]) -> Tuple[List[str]]:
    """
    Args:
        base_filenames (Iterable[str]): _description_

    Returns:
        Tuple[List[str]]: _description_
    """

    log_with_nemo_at_level(
        f"""
        isolate_filenames_with_last_to_keep(), begin
        base_filenames={base_filenames}
        """
    )
    filenames_with_last_and_ckpt = [a for a in base_filenames if "last" in a and ".ckpt" in a]
    filenames_with_last_ckpt = [a for a in base_filenames if "last.ckpt" in a]
    filenames_with_last_and_ckpt_but_not_last_ckpt = [
        a for a in filenames_with_last_and_ckpt if a not in filenames_with_last_ckpt
    ]

    filenames_with_last_ckpt = sorted(filenames_with_last_ckpt, reverse=True)
    filenames_with_last_and_ckpt_but_not_last_ckpt = sorted(
        filenames_with_last_and_ckpt_but_not_last_ckpt, reverse=True
    )
    log_with_nemo_at_level(
        f"""
        isolate_filenames_with_last_to_keep(), after isolating
        filenames_with_last_ckpt={filenames_with_last_ckpt}
        filenames_with_last_and_ckpt_but_not_last_ckpt={filenames_with_last_and_ckpt_but_not_last_ckpt}
        """
    )
    filenames_with_last_and_ckpt_to_keep = []
    if len(filenames_with_last_ckpt) > 0:
        filenames_with_last_and_ckpt_to_keep.append(filenames_with_last_ckpt[0])
    if len(filenames_with_last_and_ckpt_but_not_last_ckpt) > 0:
        filenames_with_last_and_ckpt_to_keep.append(filenames_with_last_and_ckpt_but_not_last_ckpt[0])

    filenames_with_last_and_ckpt_to_remove = [
        x for x in filenames_with_last_and_ckpt if x not in filenames_with_last_and_ckpt_to_keep
    ]

    return filenames_with_last_and_ckpt_to_keep, filenames_with_last_and_ckpt_to_remove


def most_recent_run_dir_from_subdir_names(subdir_names):
    run_dirs_found = sorted(
        [x for x in subdir_names if "run_" == x[0:4]],
        reverse=True,
    )
    return run_dirs_found[0] if len(run_dirs_found) > 0 else None
