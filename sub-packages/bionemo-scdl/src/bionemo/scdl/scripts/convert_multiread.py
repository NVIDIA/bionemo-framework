import tempfile
from typing import Any
# from bionemo.core.data.load import load
# import numpy as np
import os

# from bionemo.scdl.io.single_cell_memmap_dataset import SingleCellMemMapDataset
 
# # Paths to the files
# file1 = "data1.npy"
# file2 = "data2.npy"

# def extend(first: Any, second: Any, buffer_size_b = 1024*1024):
#     """
#     Takes in two SCDL mmapped arrays
#     and concatenates the items in second to first.
    
#     This function return first, which has been modified.
#     """
    
#     # Define buffer size (smaller than size1 or size2 to minimize memory/disk usage)
    
#     # Open both files
#     with open(first, "r+b") as f1, open(second, "rb") as f2:
#         size1 = os.path.getsize(first)
#         size2 = os.path.getsize(second)
    
#         # Resize file1 to the final size incrementally
#         f1.seek(size1 + size2 - 1)
#         f1.write(b'\0')  # Extend file1
    
#         # Move data from file2 to file1 in chunks
#         remaining_data1 = size1  # Tracks unprocessed part of data1
#         write_position = size1  # Start appending at the end of original data1
#         read_position = 0       # Start reading from the beginning of file2
    
#         while remaining_data1 > 0 or read_position < size2:
#             # Determine how much to read/write in this iteration
#             chunk_size = min(buffer_size_b, remaining_data1, size2 - read_position)
    
#             # Buffer to temporarily hold `data1.npy` data that will be overwritten
#             f1.seek(write_position - remaining_data1)
#             buffer = f1.read(chunk_size)
    
#             # Write the new data from `data2.npy` into `data1.npy`
#             f2.seek(read_position)
#             new_data = f2.read(chunk_size)
#             f1.seek(write_position)
#             f1.write(new_data)
    
#             # Write back the buffered data to its new position in `data1.npy`
#             f1.seek(write_position - remaining_data1)
#             f1.write(buffer)
    
#             # Update pointers
#             remaining_data1 -= chunk_size
#             read_position += chunk_size
#             write_position += chunk_size
 


import os

def extend(first: str, second: str,
            buffer_size_b: int = 1024 * 1024,
            delete_file2_on_complete: bool = False):
    """
    Concatenates the contents of `second` into `first` using memory-efficient operations.
    
    Parameters:
    - first (str): Path to the first file (will be extended).
    - second (str): Path to the second file (data will be read from here).
    - buffer_size_b (int): Size of the buffer to use for reading/writing data.
    
    Returns:
    - None
    """
    # Open both files
    with open(first, "r+b") as f1, open(second, "rb") as f2:
        size1 = os.path.getsize(first)
        size2 = os.path.getsize(second)
        
        # Resize file1 to the final size to accommodate both files
        f1.seek(size1 + size2 - 1)
        f1.write(b'\0')  # Extend file1
        
        # Move data from file2 to file1 in chunks
        read_position = 0       # Start reading from the beginning of file2
        write_position = size1  # Start appending at the end of original data1

        while read_position < size2:
            # Determine how much to read/write in this iteration
            chunk_size = min(buffer_size_b, size2 - read_position)
            
            # Read data from file2
            f2.seek(read_position)
            new_data = f2.read(chunk_size)
            
            # Write the new data into file1
            f1.seek(write_position)
            f1.write(new_data)
            
            # Update pointers
            read_position += chunk_size
            write_position += chunk_size
        
    if delete_file2_on_complete:
        os.remove(second)

# Delete the second file to free space
import random

def generate_random_numbers_file(filename,
                                 num_lines=10,
                                 start=0):
    """
    Generate a file with random numbers, one per line.

    Parameters:
        filename (str): Name of the file to save the random numbers.
        num_lines (int): Number of lines of random numbers. Default is 1000.
    """
    with open(filename, 'w') as file:
        for _ in range(num_lines):
            file.write(f"{start}\n")
            start += 1



if __name__ == "__main__":

    # ## Read three anndatas and convert to SCDLs
    # train_dir =  load("single_cell/testdata-20240506") / "cellxgene_2023-12-15_small" / "input_data" / "train"
    # # directory_path = Path(directory_path)
    # ann_data_paths = sorted(train_dir.rglob("*.h5ad"))

    # scdl_arr = []
    # with tempfile.TemporaryDirectory() as temp_dir:
    # ## read in three SCDL's into a list
    #     for path in ann_data_paths:
    #         scdl_arr.append(SingleCellMemMapDataset(str(path)))
    # ## extend scdl_1 by scdl_2 -> SCDL_p1
    # ## First, extend the rowptr array, then record the number of rows and increment the second array by that
    # # for i in range(1, len(scdl_arr)):
        
    # Example usage
    generate_random_numbers_file("file_1.txt", 12, 22)
    generate_random_numbers_file("second.txt", 1000, 40)
    extend("file_1.txt", "second.txt")
    ## extend SCDL_p1 by scdl_3