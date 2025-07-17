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


import argparse
import gc
import mmap
import multiprocessing as mp
import os
import time

import numpy as np
import psutil


def get_rss(pid):
    try:
        return psutil.Process(pid).memory_info().rss
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0


def monitor_peak_memory(parent_pid, stop_event, result_queue):
    peak = get_rss(parent_pid)
    try:
        parent = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        result_queue.put((0, 0))
        return

    start_rss = peak

    while not stop_event.is_set():
        try:
            children = parent.children(recursive=True)
            all_pids = [parent_pid] + [c.pid for c in children if c.is_running()]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            all_pids = [parent_pid]

        mem = sum(get_rss(pid) for pid in all_pids if psutil.pid_exists(pid))
        peak = max(peak, mem)
        time.sleep(0.05)

    result_queue.put((start_rss, peak))


def main(path, num_samples=100_000, chunk_size=1_000, dtype_str="uint32", madvise_interval=None, shuffle=False):
    dtype = getattr(np, dtype_str)
    dtype_size = np.dtype(dtype).itemsize

    file_size = os.path.getsize(path)
    num_elements = file_size // dtype_size

    with open(path, "r+b") as f:
        mm = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)
        arr = np.ndarray(shape=(num_elements,), dtype=dtype, buffer=mm)

        # Start peak memory monitor
        parent_pid = os.getpid()
        stop_event = mp.Event()
        result_queue = mp.Queue()
        monitor = mp.Process(target=monitor_peak_memory, args=(parent_pid, stop_event, result_queue))

        gc.collect()
        process = psutil.Process(parent_pid)
        start_rss = process.memory_info().rss
        start_time = time.time()

        monitor.start()

        rng = np.random.default_rng()
        if shuffle:
            random_indices = rng.integers(0, num_elements - chunk_size, size=num_samples)
        elif chunk_size == num_elements:
            random_indices = np.arange(0, num_elements, chunk_size)
        else:
            offset = rng.integers(0, chunk_size)
            num_chunks = (num_elements - chunk_size - offset) // chunk_size

            if num_chunks < num_samples:
                raise ValueError(f"Not enough possible chunks ({num_chunks}) for num_samples={num_samples}")

            chunk_indices = rng.choice(num_chunks, size=num_samples, replace=False)
            random_indices = offset + chunk_indices * chunk_size

        madvise_time = 0

        for i, k in enumerate(random_indices):
            x = arr[k : k + chunk_size] + 1
            del x
            if madvise_interval is not None and i % madvise_interval == 0:
                try:
                    t0 = time.time()
                    mm.madvise(mmap.MADV_DONTNEED)
                    madvise_time += time.time() - t0
                except AttributeError:
                    print("madvise not supported on this platform or Python version")

        end_time = time.time()
        end_rss = process.memory_info().rss
        stop_event.set()
        monitor.join(timeout=5)

        try:
            start_rss_from_monitor, peak_rss = result_queue.get(timeout=2)
        except Exception:
            start_rss_from_monitor = start_rss
            peak_rss = end_rss

        duration_sec = end_time - start_time
        delta_rss_mb = (end_rss - start_rss) / 1024 / 1024
        peak_rss_delta_mb = (peak_rss - start_rss_from_monitor) / 1024 / 1024

        print(f"Loop time: {duration_sec:.2f} seconds")
        print(f"RSS delta: {delta_rss_mb:.2f} MB")
        print(f"Peak RSS delta: {peak_rss_delta_mb:.2f} MB")
        print(f"madvise time: {madvise_time:.6f} seconds")

        return duration_sec, delta_rss_mb, peak_rss_delta_mb


if __name__ == "__main__":
    import gc
    import subprocess

    parser = argparse.ArgumentParser(description="Memmap tradeoff experiment")
    parser.add_argument(
        "--path",
        type=str,
        default="/home/pbinder/bionemo-framework/tahoe_memmap/col_ptr.npy",
        help="Path to .npy file",
    )
    parser.add_argument("--dtype", type=str, default="uint32", help="Numpy dtype (e.g. uint32)")
    parser.add_argument("--log_file", type=str, default="madvise_results.tsv", help="File to save results")
    args = parser.parse_args()

    chunk_sizes = [1, 10, 100, 1000]
    madvise_intervals = [None]#, 1, 10, 100, 1_000]
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
                            duration_sec, delta_rss_mb, peak_rss_delta_mb = main(
                                path=args.path,
                                num_samples=num_samples,
                                chunk_size=chunk_size,
                                dtype_str=args.dtype,
                                madvise_interval=madvise_interval,
                                shuffle=shuffle,
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
