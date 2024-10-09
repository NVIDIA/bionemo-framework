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


import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import List

import pytest

from bionemo.model.protein.prott5nv import ProtT5nvInference
from bionemo.model.utils import initialize_distributed_parallel_state
from bionemo.utils.hydra import load_model_config
from bionemo.utils.tests import (
    check_model_exists,
    distributed_model_parallel_state,
    teardown_apex_megatron_cuda,
)

from .inference_shared_test_code import (
    get_config_dir,
    get_expected_vals_file,
    get_inference_class,
    run_seqs_to_embedding,
    run_seqs_to_hiddens_with_goldens,
)


log = logging.getLogger(__name__)

CHECKPOINT_PATH = os.path.join(os.environ["BIONEMO_HOME"], "models/protein/prott5nv/prott5nv.nemo")
SEQS_FOR_TEST = [
    "MSLKRKNIALIPAAGIGVRFGADKPKQYVEIGSKTVLEHVLGIFERHEAVDLTVVVVSPEDTFADKVQTAFPQVRVWKNGGQTRAETVRNGVAKLLETGLAAETDNILVHDAARCCLPSEALARLIEQAGNAAEGGILAVPVADTLKRAESGQISATVDRSGLWQAQTPQLFQAGLLHRALAAENLGGITDEASAVEKLGVRPLLIQGDARNLKLTQPQDAYIVRLLLDAV",
    "MIQSQINRNIRLDLADAILLSKAKKDLSFAEIADGTGLAEAFVTAALLGQQALPADAARLVGAKLDLDEDSILLLQMIPLRGCIDDRIPTDPTMYRFYEMLQVYGTTLKALVHEKFGDGIISAINFKLDVKKVADPEGGERAVITLDGKYLPTKPF",
]


@pytest.fixture()
def _seqs() -> List[str]:
    return deepcopy(SEQS_FOR_TEST)


@pytest.fixture(scope="module")
def config_path(bionemo_home) -> str:
    path = bionemo_home / "examples" / "protein" / "prott5nv" / "conf"
    return str(path)


@pytest.fixture(scope="module")
def inference_model(config_path) -> ProtT5nvInference:
    cfg = load_model_config(config_name="infer", config_path=config_path)
    initialize_distributed_parallel_state()
    # load score model
    model = ProtT5nvInference(cfg, inference_batch_size_for_warmup=2)
    model.eval()
    yield model
    teardown_apex_megatron_cuda()


@pytest.mark.needs_checkpoint
def test_model_exists():
    check_model_exists(CHECKPOINT_PATH)


@pytest.mark.needs_gpu
@pytest.mark.needs_checkpoint
@pytest.mark.skip_if_no_file(CHECKPOINT_PATH)
def test_seq_to_embedding(inference_model: ProtT5nvInference):
    seqs = [
        "MSLKRKNIALIPAAGIGVRFGADKPKQYVEIGSKTVLEHVLGIFERHEAVDLTVVVVSPEDTFADKVQTAFPQVRVWKNGGQTRAETVRNGVAKLLETGLAAETDNILVHDAARCCLPSEALARLIEQAGNAAEGGILAVPVADTLKRAESGQISATVDRSGLWQAQTPQLFQAGLLHRALAAENLGGITDEASAVEKLGVRPLLIQGDARNLKLTQPQDAYIVRLLLDAV",
        "MIQSQINRNIRLDLADAILLSKAKKDLSFAEIADGTGLAEAFVTAALLGQQALPADAARLVGAKLDLDEDSILLLQMIPLRGCIDDRIPTDPTMYRFYEMLQVYGTTLKALVHEKFGDGIISAINFKLDVKKVADPEGGERAVITLDGKYLPTKPF",
    ]
    embedding = inference_model.seq_to_embeddings(seqs)
    assert embedding is not None

    assert embedding.shape[0] == len(seqs)
    assert len(embedding.shape) == 2


@pytest.mark.needs_gpu
@pytest.mark.needs_checkpoint
@pytest.mark.skip_if_no_file(CHECKPOINT_PATH)
def test_long_seq_to_embedding(inference_model):
    long_seq = "MIQSQINRNIRLDLADAILLSKAKKDLSFAEIADGTGLAEAFVTAALLGQQALPADAARLVGAKLDLDEDSILLLQMIPLRGCIDDRIPTDPTMYRFYEMLQVYGTTLKALVHEKFGDGIISAINFKLDVKKVADPEGGERAVITLDGKYLPTKPF"
    long_seq = long_seq * 10
    seqs = [
        "MSLKRKNIALIPAAGIGVRFGADKPKQYVEIGSKTVLEHVLGIFERHEAVDLTVVVVSPEDTFADKVQTAFPQVRVWKNGGQTRAETVRNGVAKLLETGLAAETDNILVHDAARCCLPSEALARLIEQAGNAAEGGILAVPVADTLKRAESGQISATVDRSGLWQAQTPQLFQAGLLHRALAAENLGGITDEASAVEKLGVRPLLIQGDARNLKLTQPQDAYIVRLLLDAV",
        long_seq,
    ]
    try:
        inference_model.seq_to_hiddens(seqs)
        assert False
    except Exception:
        pass


@pytest.fixture(scope="module")
def prott5nv_inferer(bionemo_home: Path) -> ProtT5nvInference:
    model_name = "prott5nv"
    cfg_path = get_config_dir(bionemo_home, model_name)
    cfg = load_model_config(config_name="infer", config_path=cfg_path)
    with distributed_model_parallel_state():
        inferer = get_inference_class(model_name)(
            cfg=cfg, inference_batch_size_for_warmup=2
        )  # Change to 1 to debug the failure
        yield inferer  # Yield so cleanup happens after the test


@pytest.fixture(scope="module")
def prott5nv_expected_vals_path(bionemo_home: Path) -> Path:
    return get_expected_vals_file(bionemo_home, "prott5nv")


@pytest.mark.needs_fork
@pytest.mark.needs_gpu
def test_seq_to_hiddens_with_goldens_prott5nv(
    prott5nv_inferer: ProtT5nvInference, _seqs: List[str], prott5nv_expected_vals_path: Path
):
    run_seqs_to_hiddens_with_goldens(
        prott5nv_inferer,
        _seqs,
        prott5nv_expected_vals_path,
        prott5nv_inferer.cfg.model.encoder.hidden_size,
        prott5nv_inferer.model.cfg.encoder.arch,
        tokenize_fn=prott5nv_inferer._tokenize,
    )


@pytest.mark.needs_fork
@pytest.mark.needs_gpu
def test_seqs_to_embedding_prott5nv(prott5nv_inferer: ProtT5nvInference, _seqs: List[str]):
    run_seqs_to_embedding(prott5nv_inferer, _seqs, prott5nv_inferer.cfg.model.encoder.hidden_size)
