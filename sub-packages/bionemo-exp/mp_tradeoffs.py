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


import multiprocessing as mp
import os
import time

import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader, Dataset


# ---------- Memory Tracking ----------
def get_rss(pid):
    try:
        return psutil.Process(pid).memory_info().rss
    except Exception:
        return 0


def get_pss(pid):
    try:
        return psutil.Process(pid).memory_full_info().pss
    except Exception:
        return get_rss(pid)


def monitor_memory(pid, stop_event, queue, use_pss=True):
    peak = 0
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        queue.put((0, 0))
        return

    start = get_pss(pid) if use_pss else get_rss(pid)

    while not stop_event.is_set():
        try:
            children = parent.children(recursive=True)
            all_pids = [pid] + [c.pid for c in children if c.is_running()]
        except Exception:
            all_pids = [pid]

        usage = sum((get_pss(p) if use_pss else get_rss(p)) for p in all_pids)
        peak = max(peak, usage)
        time.sleep(0.05)

    queue.put((start, peak))


# ---------- Dataset ----------
class MemmapDataset(Dataset):
    def __init__(self, path, dtype=np.uint32, chunk_size=1000):
        self.chunk_size = chunk_size
        self.data = np.memmap(path, dtype=dtype, mode="r")
        self.num_chunks = len(self.data) // chunk_size

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, idx):
        start = idx * self.chunk_size
        end = start + self.chunk_size
        return torch.from_numpy(self.data[start:end].copy())  # copy to ensure safe across workers


# ---------- Benchmark ----------
def run_dataloader_with_memory_tracking(dataloader, num_samples=10, madvise_interval=None, use_pss=False):
    stop_event = mp.Event()
    queue = mp.Queue()
    monitor = mp.Process(target=monitor_memory, args=(os.getpid(), stop_event, queue, use_pss))
    monitor.start()

    start_time = time.time()

    for i, batch in enumerate(dataloader):
        if madvise_interval and i % madvise_interval == 0:
            try:
                t0 = time.time()
                mm.madvise(mmap.MADV_DONTNEED)
                madvise_time += time.time() - t0
            except AttributeError:
                print("madvise not supported on this platform or Python version")
        if i >= num_samples:
            break
        del batch

    end_time = time.time()

    stop_event.set()
    monitor.join()

    start_mem, peak_mem = queue.get()

    delta_rss_mb = (get_memory(os.getpid(), use_pss) - start_mem) / 1024 / 1024
    peak_rss_delta_mb = (peak_mem - start_mem) / 1024 / 1024
    duration_sec = end_time - start_time

    return duration_sec, delta_rss_mb, peak_rss_delta_mb


# ---------- Main ----------
if __name__ == "__main__":
    import subprocess

    parser = argparse.ArgumentParser(description="Memmap tradeoff experiment")
    parser.add_argument(
        "--path",
        type=str,
        default="/home/pbinder/bionemo-framework/tahoe_memmap/col_ptr.npy",
        help="Path to .npy file",
    )
    parser.add_argument("--dtype", type=str, default="uint32", help="Numpy dtype (e.g. uint32)")
    parser.add_argument("--log_file", type=str, default="madvise_mp_results.tsv", help="File to save results")
    args = parser.parse_args()

    chunk_sizes = [1, 10, 100, 1000]
    madvise_intervals = [None, 1, 10, 100, 1_000]
    shuffle_options = [False, True]
    num_samples_list = [10_000, 1_000_000]

    header_written = False

    with open(args.log_file, "a") as f:
        for num_samples in num_samples_list:
            for chunk_size in chunk_sizes:
                for madvise_interval in madvise_intervals:
                    for shuffle in shuffle_options:
                        label = (
                            f"chunk={chunk_size}, madvise={madvise_interval}, "
                            f"shuffle={shuffle}, num_samples={num_samples}"
                        )
                        print(f"\n=== Running: {label} ===")

                        # Drop caches
                        try:
                            subprocess.run(["sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"], check=True)
                        except subprocess.CalledProcessError:
                            print("⚠️ Warning: failed to drop caches — are you running with sudo?")

                        # Run benchmark and capture outputs
                        try:
                            dataset = MemmapDataset(args.path, chunk_size=chunk_size)
                            dataloader = DataLoader(
                                dataset, batch_size=8, num_workers=4, prefetch_factor=2, shuffle=shuffle
                            )

                            duration_sec, delta_rss_mb, peak_rss_delta_mb = run_dataloader_with_memory_tracking(
                                dataloader=dataloader,
                                num_samples=num_samples,
                                madvise_interval=madvise_interval,
                            )
                        except Exception as e:
                            print(f"❌ Error during run: {e}")
                            continue

                        # Write result immediately
                        if not header_written:
                            f.write(
                                "chunk_size\tmadvise_interval\tshuffle\tnum_samples\tduration_sec\tdelta_rss_mb\tpeak_rss_delta_mb\n"
                            )
                            header_written = True

                        f.write(
                            f"{chunk_size}\t{madvise_interval}\t{shuffle}\t{num_samples}\t{duration_sec:.2f}\t{delta_rss_mb:.2f}\t{peak_rss_delta_mb:.2f}\n"
                        )
                        f.flush()

    (dataloader)
