import os

def _extend(first: str, second: str,
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

    Effects:
    - first is modifed to contain the data from second.
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