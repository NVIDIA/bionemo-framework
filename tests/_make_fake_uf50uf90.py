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


alphabet = "ARNDCQEGHILKMFPSTWYV"

cluster_mapping = {}
with open("Fake50.fasta", "w") as fd:
    for length in range(50, 1000, 50):
        for base in alphabet:
            fd.write(f">UniRef50_{base}_{length}\n")
            fd.write(base * length)
            fd.write("\n")
            cluster_mapping[f"UniRef50_{base}_{length}"] = []


with open("Fake90.fasta", "w") as fd:
    num_clusters = 10
    for length in range(50, 1000, 50):
        for base in alphabet:
            for i in range(num_clusters):
                fd.write(f">UniRef90_{base}_{length}-{i}\n")
                fd.write(base * (length + i))
                fd.write("\n")
                cluster_mapping[f"UniRef50_{base}_{length}"].append(f"UniRef90_{base}_{length}-{i}")
            num_clusters += 1


with open("mapping.tsv", "w") as fd:
    fd.write("dumbheader\n")
    for key, values in cluster_mapping.items():
        str_ = key + "\t" + ",".join(values)
        fd.write(f"{str_}\n")
