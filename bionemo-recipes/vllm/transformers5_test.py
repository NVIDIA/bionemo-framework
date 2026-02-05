# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from transformers import AutoModel, AutoTokenizer


model = AutoModel.from_pretrained("nvidia/esm2_t6_8M_UR50D", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("nvidia/esm2_t6_8M_UR50D", trust_remote_code=True)

print(model)

gfp_P42212 = (
    "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTL"
    "VTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLV"
    "NRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLAD"
    "HYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
)

inputs = tokenizer(gfp_P42212, return_tensors="pt")
inputs = inputs.to("cuda")
model = model.to("cuda")
model.eval()

output = model(**inputs)
# print(output)
print(output.last_hidden_state.shape)
print(output.pooler_output.shape)
if output.hidden_states:
    for hs in output.hidden_states:
        print(hs.shape)
