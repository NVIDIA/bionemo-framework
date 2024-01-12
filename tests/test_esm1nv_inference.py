# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging
import os
import pathlib
from contextlib import contextmanager

import pytest
from hydra import compose, initialize

from bionemo.model.protein.esm1nv import ESM1nvInference
from bionemo.utils.tests import (
    BioNemoSearchPathConfig,
    check_model_exists,
    register_searchpath_config_plugin,
    update_relative_config_dir,
)


log = logging.getLogger(__name__)

BIONEMO_HOME = os.getenv("BIONEMO_HOME")
CONFIG_PATH = "../examples/protein/esm1nv/conf"  # Hydra config paths must be relative
PREPEND_CONFIG_DIR = os.path.join(BIONEMO_HOME, "examples/conf")
MODEL_CLASS = ESM1nvInference
CHECKPOINT_PATH = os.path.join(BIONEMO_HOME, "models/protein/esm1nv/esm1nv.nemo")

####

_INFERER = None
THIS_FILE_DIR = pathlib.Path(os.path.abspath(__file__)).parent


def get_cfg(prepend_config_path, config_name, config_path='conf'):
    prepend_config_path = pathlib.Path(prepend_config_path)

    class TestSearchPathConfig(BioNemoSearchPathConfig):
        def __init__(self) -> None:
            super().__init__()
            self.prepend_config_dir = update_relative_config_dir(prepend_config_path, THIS_FILE_DIR)

    register_searchpath_config_plugin(TestSearchPathConfig)
    with initialize(config_path=config_path):
        cfg = compose(config_name=config_name)

    return cfg


@contextmanager
def load_model(inf_cfg):
    global _INFERER
    if _INFERER is None:
        _INFERER = MODEL_CLASS(inf_cfg)
    yield _INFERER


@pytest.mark.needs_checkpoint
def test_model_exists():
    check_model_exists(CHECKPOINT_PATH)


@pytest.mark.needs_gpu
@pytest.mark.needs_checkpoint
@pytest.mark.skip_if_no_file(CHECKPOINT_PATH)
def test_seq_to_embedding():
    cfg = get_cfg(PREPEND_CONFIG_DIR, config_name='infer', config_path=CONFIG_PATH)
    with load_model(cfg) as inferer:
        seqs = [
            'MSLKRKNIALIPAAGIGVRFGADKPKQYVEIGSKTVLEHVLGIFERHEAVDLTVVVVSPEDTFADKVQTAFPQVRVWKNGGQTRAETVRNGVAKLLETGLAAETDNILVHDAARCCLPSEALARLIEQAGNAAEGGILAVPVADTLKRAESGQISATVDRSGLWQAQTPQLFQAGLLHRALAAENLGGITDEASAVEKLGVRPLLIQGDARNLKLTQPQDAYIVRLLLDAV',
            'MIQSQINRNIRLDLADAILLSKAKKDLSFAEIADGTGLAEAFVTAALLGQQALPADAARLVGAKLDLDEDSILLLQMIPLRGCIDDRIPTDPTMYRFYEMLQVYGTTLKALVHEKFGDGIISAINFKLDVKKVADPEGGERAVITLDGKYLPTKPF',
        ]
        embedding, mask = inferer.seq_to_hiddens(seqs)
        assert embedding is not None
        assert embedding.shape[0] == len(seqs)
        assert len(embedding.shape) == 3


@pytest.mark.needs_gpu
@pytest.mark.needs_checkpoint
@pytest.mark.skip_if_no_file(CHECKPOINT_PATH)
def test_long_seq_to_embedding():
    long_seq = 'MIQSQINRNIRLDLADAILLSKAKKDLSFAEIADGTGLAEAFVTAALLGQQALPADAARLVGAKLDLDEDSILLLQMIPLRGCIDDRIPTDPTMYRFYEMLQVYGTTLKALVHEKFGDGIISAINFKLDVKKVADPEGGERAVITLDGKYLPTKPF'
    long_seq = long_seq * 10

    cfg = get_cfg(PREPEND_CONFIG_DIR, config_name='infer', config_path=CONFIG_PATH)
    with load_model(cfg) as inferer:
        seqs = [
            'MSLKRKNIALIPAAGIGVRFGADKPKQYVEIGSKTVLEHVLGIFERHEAVDLTVVVVSPEDTFADKVQTAFPQVRVWKNGGQTRAETVRNGVAKLLETGLAAETDNILVHDAARCCLPSEALARLIEQAGNAAEGGILAVPVADTLKRAESGQISATVDRSGLWQAQTPQLFQAGLLHRALAAENLGGITDEASAVEKLGVRPLLIQGDARNLKLTQPQDAYIVRLLLDAV',
            long_seq,
        ]
        try:
            inferer.seq_to_hiddens(seqs)
            assert False
        except Exception:
            pass
