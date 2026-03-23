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

"""Summarize full rank_0_metrics CSVs into small files for git transfer."""

import os

import pandas as pd


os.chdir(os.path.dirname(os.path.abspath(__file__)))

for tag in ["fl2", "fl4"]:
    infile = f"analysis_output/csv_data/rank_0_metrics_{tag}.csv"
    if not os.path.exists(infile):
        print(f"Skipping {infile} (not found)")
        continue

    df = pd.read_csv(infile)
    print(f"\n{tag}: {len(df)} rows, iterations {df['iteration'].min()}-{df['iteration'].max()}")

    # Summary: one row per metric with mean/max/std
    summary = df.groupby("metric_name")["value"].agg(["mean", "max", "std"]).reset_index()
    summary.to_csv(f"analysis_output/csv_data/summary_{tag}.csv", index=False)
    print(f"  summary: {len(summary)} rows -> summary_{tag}.csv")

    # Sampled: every 100th iteration (keeps time trends, much smaller)
    sampled = df[df["iteration"] % 100 == 0]
    sampled.to_csv(f"analysis_output/csv_data/sampled_{tag}.csv", index=False)
    print(f"  sampled: {len(sampled)} rows -> sampled_{tag}.csv")
