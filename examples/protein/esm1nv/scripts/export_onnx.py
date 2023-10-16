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

import logging
import os

from hydra import compose, initialize

from bionemo.model.protein.esm1nv import ESM1nvInference


log = logging.getLogger(__name__)
SAVE_TO = '/model'


with initialize(config_path="../conf"):
    cfg = compose(config_name="infer")
    inferer = ESM1nvInference(cfg)
    model = inferer.model.cuda()
    if model.enable_autocast:
        model = model.half()

    # Convert to torchscript
    seqs = [
        'MSLKRKNIALIPAAGIGVRFGADKPKQYVEIGSKTVLEHVLGIFERHEAVDLTVVVVSPEDTFADKVQTAFPQVRVWKNGGQTRAETVRNGVAKLLETGLAAETDNILVHDAARCCLPSEALARLIEQAGNAAEGGILAVPVADTLKRAESGQISATVDRSGLWQAQTPQLFQAGLLHRALAAENLGGITDEASAVEKLGVRPLLIQGDARNLKLTQPQDAYIVRLLLDAV',
        'MIQSQINRNIRLDLADAILLSKAKKDLSFAEIADGTGLAEAFVTAALLGQQALPADAARLVGAKLDLDEDSILLLQMIPLRGCIDDRIPTDPTMYRFYEMLQVYGTTLKALVHEKFGDGIISAINFKLDVKKVADPEGGERAVITLDGKYLPTKPF',
    ]

    tokens_enc, enc_mask = inferer.tokenize(seqs)

    # print("simple inference: ", inferer.seq_to_embedding(seqs))

    dynamix_axes = model.input_types
    dynamix_axes.update(model.output_types)

    model.export(
        os.path.join(SAVE_TO, 'model.onnx'), input_example=(tokens_enc, enc_mask, None), dynamic_axes=dynamix_axes
    )

    print('Exported to', SAVE_TO)
