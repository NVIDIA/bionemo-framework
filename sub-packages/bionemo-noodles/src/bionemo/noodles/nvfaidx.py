from typing import Optional
from bionemo.noodles import PyIndexedMmapFastaReader


class SequenceAccessor:
    # NOTE: we could totally handle this stuff in Rust if we want.
    def __init__(self, reader, seqid, length):
        self.reader = reader
        self.seqid = seqid
        self.length = length

    def __getitem__(self, key):
        if isinstance(key, slice):
            # Provide defaults for missing arguments in the slice.
            start = key.start if key.start is not None else 0
            stop = key.stop if key.stop is not None else self.length

            # Handle negative cases, remember, you can be arbitrarily negative in a slice.
            if start < 0:
                start += self.length
            if stop < 0:
                stop += self.length

            # Clamp normalized indices to valid range
            start = max(0, min(self.length, start))
            stop = max(0, min(self.length, stop))

            # Bounds checking after normalization
            if start > stop:
                return ""  # Return empty string for an empty slice

            # Construct region string
            region_str = f"{self.seqid}:{start + 1}-{stop}"  # +1 for 1-based indexing
            return self.reader.read_sequence_mmap(region_str)

        elif isinstance(key, int):
            # Normalize single integer for negative indexing
            if key < 0:
                key += self.length

            # Bounds checking
            if key < 0 or key >= self.length:
                raise IndexError(f"Position {key} is out of bounds for '{self.seqid}' with length {self.length}.")

            # Query single nucleotide by creating a 1-length region
            region_str = f"{self.seqid}:{key + 1}-{key + 1}"  # +1 for 1-based indexing
            return self.reader.read_sequence_mmap(region_str)

        else:
            raise TypeError("Index must be an integer or a slice.")


class NvFaidx:
    def __init__(self, fasta_path, faidx_path: Optional[str]=None, ignore_existing_fai=True):
        ''' Construct a dict-like object representing a memmapped, indexed FASTA file.

        Args:
            fasta_path (str): Path to the FASTA file.
            faidx_path (str): Path to the FAI index file. If None, one will be created.
            ignore_existing_fai (bool): If True, ignore any existing FAI file and create an in-memory index. Note that this will also ignore `faidx_path`.
        '''
        if ignore_existing_fai:
            self.reader = PyIndexedMmapFastaReader(fasta_path, ignore_existing_fai=ignore_existing_fai)
        elif faidx_path is not None:
            self.reader = PyIndexedMmapFastaReader.from_fasta_and_faidx(fasta_path, faidx_path)
        else:
            self.reader = PyIndexedMmapFastaReader(fasta_path)

        self.records = {record.name: record for record in self.reader.records()}

    def __getitem__(self, seqid):
        if seqid not in self.records:
            raise KeyError(f"Sequence '{seqid}' not found in index.")

        # Return a SequenceAccessor for slicing access
        record_length = self.records[seqid].length
        return SequenceAccessor(self.reader, seqid, record_length)

    def __contains__(self, seqid):
        return seqid in self.records

    def __len__(self):
        return len(self.records)

    def keys(self):
        return self.records.keys()