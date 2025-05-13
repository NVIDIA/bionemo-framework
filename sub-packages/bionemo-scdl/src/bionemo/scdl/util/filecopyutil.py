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


def extend_files(
    first: str, second: str, buffer_size_b: int = 10 * 1024 * 1024, delete_file2_on_complete: bool = False, offset=0
):
    """Concatenates the contents of `second` into `first` using memory-efficient operations.

    Shrinks `second` incrementally after reading each chunk. This is not multi-processing safe.

    Parameters:
    - first (str): Path to the first file (will be extended).
    - second (str): Path to the second file (data will be read from here).
    - buffer_size_b (int): Size of the buffer to use for reading/writing data.
    - delete_file2_on_complete (bool): Whether to delete the second file after operation.

    """
    with open(first, "r+b") as f1, open(second, "rb") as f2:
        size1 = os.path.getsize(first)
        size2 = os.path.getsize(second)

        # Resize file1 to the final size to accommodate both files
        f1.seek(size1 + size2 - 1 - offset)
        f1.write(b"\0")  # Extend file1

        # Move data from file2 to file1 in chunks
        read_position = offset  # Start reading from the beginning of file2
        write_position = size1  # Start appending at the end of original data1
        f2.seek(read_position)

        while read_position < size2:
            # Determine how much to read/write in this iteration
            chunk_size = min(buffer_size_b, size2 - read_position)

            # Read data from file2
            new_data = f2.read(chunk_size)

            # Write the new data into file1
            f1.seek(write_position)
            f1.write(new_data)

            # Update pointers
            read_position += chunk_size
            write_position += chunk_size
            f2.seek(read_position)

    if delete_file2_on_complete:
        os.remove(second)
