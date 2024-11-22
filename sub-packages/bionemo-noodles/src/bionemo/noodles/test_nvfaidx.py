import pytest
from bionemo.noodles.nvfaidx import NvFaidx
from bionemo.noodles import IndexedMmapFastaReader
import os
import tempfile
import random
import pyfaidx
import torch

def test_memmap_index_iso():
    # This tests a specific edge case that was failing.
    fasta_path = 'sub-packages/bionemo-noodles/tests/bionemo/noodles/data/sample.fasta'
    index = IndexedMmapFastaReader(fasta_path)

    assert index.read_sequence_mmap('chr4:1-10000') == 'CCCCCCCCCCCCACGT'
    assert index.read_sequence_mmap('chr4:1-17') == 'CCCCCCCCCCCCACGT'

def test_memmap_index():
    # This should probably be a test in rust land.
    fasta_path = 'sub-packages/bionemo-noodles/tests/bionemo/noodles/data/sample.fasta'
    index = IndexedMmapFastaReader(fasta_path)
    assert index.read_sequence_mmap('chr1:1-1') == 'A'
    assert index.read_sequence_mmap('chr1:1-2') == 'AC'
    assert index.read_sequence_mmap('chr1:1-100000') == 'ACTGACTGACTG'
    assert index.read_sequence_mmap('chr2:1-2') == 'GG'
    assert index.read_sequence_mmap('chr2:1-1000000') == 'GGTCAAGGTCAA'
    # Recall to get python based indexing we add 1 to both start and end, so 1-13 is a 12 character string(full sequence)
    assert index.read_sequence_mmap('chr2:1-11') == 'GGTCAAGGTCA'
    assert index.read_sequence_mmap('chr2:1-12') == 'GGTCAAGGTCAA'
    assert index.read_sequence_mmap('chr2:1-13') == 'GGTCAAGGTCAA'

    assert index.read_sequence_mmap('chr3:1-2') == 'AG'
    assert index.read_sequence_mmap('chr3:1-13') == 'AGTCAAGGTCCAC'
    assert index.read_sequence_mmap('chr3:1-14') == 'AGTCAAGGTCCACG' # adds first character from next line
    assert index.read_sequence_mmap('chr3:1-83') == 'AGTCAAGGTCCACGTCAAGGTCCCGGTCAAGGTCCGTGTCAAGGTCCTAGTCAAGGTCAACGTCAAGGTCACGGTCAAGGTCA'
    assert index.read_sequence_mmap('chr3:1-84') == 'AGTCAAGGTCCACGTCAAGGTCCCGGTCAAGGTCCGTGTCAAGGTCCTAGTCAAGGTCAACGTCAAGGTCACGGTCAAGGTCAG'
    assert index.read_sequence_mmap('chr3:1-10000') == 'AGTCAAGGTCCACGTCAAGGTCCCGGTCAAGGTCCGTGTCAAGGTCCTAGTCAAGGTCAACGTCAAGGTCACGGTCAAGGTCAG'
    assert index.read_sequence_mmap('chr3:84-84') == 'G'

    # Handles End of Index 
    # Full sequence
    assert index.read_sequence_mmap('chr5:1-1000000') == 'A'
    # Only one char, should succeed
    assert index.read_sequence_mmap('chr5:1-2') == 'A'


    # Handles end of multi line but non-full sequence entry
    # Full sequence
    assert index.read_sequence_mmap('chr4:1-16') == 'CCCCCCCCCCCCACGT'
    assert index.read_sequence_mmap('chr4:1-17') == 'CCCCCCCCCCCCACGT'
    assert index.read_sequence_mmap('chr4:1-1000000') == 'CCCCCCCCCCCCACGT'

    assert index.read_sequence_mmap('chr4:1-17') == 'CCCCCCCCCCCCACGT'

    assert index.read_sequence_mmap('chr4:3-16') == 'CCCCCCCCCCACGT'
    assert index.read_sequence_mmap('chr4:17-17') == ''

def test_getitem_bounds():
    # NOTE make this the correct path, check this file in since we are checking exactness of queries.
    index = NvFaidx('sub-packages/bionemo-noodles/tests/bionemo/noodles/data/sample.fasta')
    # first element
    assert index['chr1'][0] == 'A'
    # normal, in range, query
    assert index['chr1'][1:4] == 'CTG'
    # Going beyond the max bound in a slice should truncate at the end of the sequence
    assert index['chr1'][1:10000] == 'CTGACTGACTG'
    # Slice up to the last element
    assert index['chr1'][0:-1] == 'ACTGACTGACT'
    # equivalent to above
    assert index['chr1'][:-1] == 'ACTGACTGACT'
    # -1 should get the last element
    assert index['chr1'][-1:] == 'G'

def test_nvfaidx_python_interface():
    # This should probably be a test in rust land.
    index = NvFaidx('sub-packages/bionemo-noodles/tests/bionemo/noodles/data/sample.fasta')
    assert index['chr1'][0:1] == 'A'
    assert index['chr1'][0:2] == 'AC'
    assert index['chr1'][0:100000] == 'ACTGACTGACTG'
    assert index['chr2'][0:2] == 'GG'
    assert index['chr2'][0:100000] == 'GGTCAAGGTCAA'

    assert index['chr3'][0:2] == 'AG'
    assert index['chr3'][0:13] == 'AGTCAAGGTCCAC'
    # in progress
    assert index['chr3'][0:14] == 'AGTCAAGGTCCACG' # adds first character from next line
    assert index['chr3'][0:83] == 'AGTCAAGGTCCACGTCAAGGTCCCGGTCAAGGTCCGTGTCAAGGTCCTAGTCAAGGTCAACGTCAAGGTCACGGTCAAGGTCA'
    assert index['chr3'][0:84] == 'AGTCAAGGTCCACGTCAAGGTCCCGGTCAAGGTCCGTGTCAAGGTCCTAGTCAAGGTCAACGTCAAGGTCACGGTCAAGGTCAG'
    assert index['chr3'][0:10000] == 'AGTCAAGGTCCACGTCAAGGTCCCGGTCAAGGTCCGTGTCAAGGTCCTAGTCAAGGTCAACGTCAAGGTCACGGTCAAGGTCAG'
    assert index['chr3'][83:84] == 'G'

    # Handles End of Index 
    # Full sequence
    assert index['chr5'][0:1000000] == 'A'
    # chr5 has one char, even though this spans 2, it returns len(1)
    assert index['chr5'][0:2] == 'A'


    # Handles end of multi line but non-full sequence entry
    # Full sequence
    assert index['chr4'][0:16] == 'CCCCCCCCCCCCACGT'
    assert index['chr4'][0:17] == 'CCCCCCCCCCCCACGT'
    assert index['chr4'][0:1000000] == 'CCCCCCCCCCCCACGT'

    # This one failing is bad, it means we are not calculating the newlines correctly in some conditions.
    assert index['chr4'][0:17] == 'CCCCCCCCCCCCACGT'

    # Should see this is out of bounds and return empty or throw an error
    # assert index['chr4'][17:17] == ''

def test_generated_failure():
    # 'contig2'1000-2000
    fasta = '/workspaces/bionemo-framework/test.fasta'
    nvfaidx_fasta = NvFaidx(fasta)
    seq1 = nvfaidx_fasta['contig2'][1000:2000]
    seq2 = nvfaidx_fasta.reader.read_sequence_mmap('contig2:1001-2000')
    assert seq1 == seq2

def test_pyfaidx_nvfaidx_equivalence():
    fasta = create_test_fasta(num_seqs=2, seq_length=200000)
    pyfaidx_fasta = pyfaidx.Fasta(fasta)
    nvfaidx_fasta = NvFaidx(fasta)

    correct = 0
    for i in range(100):
        # Deterministically generate regions to grab
        seqid = f"contig{i % 2 + 1}"
        start = i * 1000
        end = start + 1000

        if not pyfaidx_fasta[seqid][start:end] == nvfaidx_fasta[seqid][start:end]:
            raise Exception(f"Pyfaidx and NvFaidx do not match. {correct=}")
        correct += 1

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, fasta_path, fasta_cls):
        self.fasta = fasta_cls(fasta_path)
        self.keys = list(self.fasta.keys())

    def __len__(self):
        # Gigantic, we dont care.
        return 99999999999

    def __getitem__(self, idx):
        # Always return the same thing to keep it easy, we assume the fasta_created is doing the right thing.
        return str(self.fasta['contig1'][150000:160000])


@pytest.mark.skip
@pytest.mark.xfail(reason="This is a known failure mode for pyfaidx that we are trying to prevent with nvfaidx.")
def _test_parallel_index_creation_pyfaidx():
    ''' 
    PyFaidx is a python replacement for faidx that provides a dictionary-like interface to reference genomes. Pyfaidx 
    is not process safe, and therefore does not play nice with pytorch dataloaders.

    Ref: https://github.com/mdshw5/pyfaidx/issues/211

    Naively, this problem can be fixed by keeping index objects private to each process. However, instantiating this object can be quite slow. 
        In the case of hg38, this can take between 15-30 seconds.

    For a good solution we need three things:
        1) Safe index creation, in multi-process or multi-node scenarios, this should be restricted to a single node where all workers block until it is complete (not implemented above)
        2) Index object instantion must be fast.
        3) Read-only use of the index object must be both thread safe and process safe with python.
    '''
    fasta = create_test_fasta(num_seqs=2, seq_length=200000)
    dl = torch.utils.data.DataLoader(TestDataset(fasta, fasta_cls = pyfaidx.Fasta), batch_size=16, num_workers=16)
    max_i = 1000
    for i, batch in enumerate(dl):
        # assert length of all elements in batch is 10000
        if i > max_i: break
        lens = [len(x) for x in batch]
        lens_equal = [x == 10000 for x in lens]
        assert all(lens_equal), (set(lens), sum(lens_equal))

def test_parallel_index_creation_nvfaidx():
    fasta = create_test_fasta(num_seqs=2, seq_length=200000)

    dl = torch.utils.data.DataLoader(TestDataset(fasta, fasta_cls = NvFaidx), batch_size=32, num_workers=16)
    max_i = 1000
    # NOTE this shouldnt be failing uh oh
    for i, batch in enumerate(dl):
        if i > max_i: break
        lens = [len(x) for x in batch]
        lens_equal = [x == 10000 for x in lens]
        assert all(lens_equal), (set(lens), sum(lens_equal))

def demo_failure_mode():
    ''' 
    PyFaidx is a python replacement for faidx that provides a dictionary-like interface to reference genomes. Pyfaidx 
    is not process safe, and therefore does not play nice with pytorch dataloaders.

    Ref: https://github.com/mdshw5/pyfaidx/issues/211

    Naively, this problem can be fixed by keeping index objects private to each process. However, instantiating this object can be quite slow. 
        In the case of hg38, this can take between 20-30 seconds.

    For a good solution we need three things:
        1) Safe index creation, in multi-process or multi-node scenarios, this should be restricted to a single node where all workers block until it is complete (not implemented above)
        2) Index object instantion must be fast.
        3) Read-only use of the index object must be both thread safe and process safe with python.
    '''
    fasta = create_test_fasta(num_seqs=2, seq_length=200000)
    dl = torch.utils.data.DataLoader(TestDataset(fasta, fasta_cls = pyfaidx.Fasta), batch_size=16, num_workers=16)
    max_i = 1000
    passed=True
    failure_set = set()
    for i, batch in enumerate(dl):
        # assert length of all elements in batch is 10000
        if i > max_i: break
        lens = [len(x) for x in batch]
        lens_equal = [x == 10000 for x in lens]
        if not all(lens_equal):
            passed = False
            failure_set = set(lens)
            break
    print(f"pyfaidx {passed=}, {failure_set=}")

    passed=True
    failure_set = set()
    dl = torch.utils.data.DataLoader(TestDataset(fasta, fasta_cls = NvFaidx), batch_size=16, num_workers=16)
    for i, batch in enumerate(dl):
        # assert length of all elements in batch is 10000
        if i > max_i: break
        lens = [len(x) for x in batch]
        lens_equal = [x == 10000 for x in lens]
    print(f"nvfaidx {passed=}, {failure_set=}")


## Benchmarks
def measure_index_creation_time():
    '''Observed performance.

    8x speedup for NvFaidx when using
    '''
    import time
    # Too slow gen a big genome
    fasta = create_test_fasta(num_seqs=1, seq_length=200_000)
    # Remove the .fai file to prevent cheating.
    if os.path.exists(fasta + ".fai"):
        os.remove(fasta + ".fai")
    start = time.time()
    _ = pyfaidx.Fasta(fasta)
    end = time.time()
    elapsed_pyfaidx = end - start

    # Remove the .fai file to prevent cheating.
    if os.path.exists(fasta + ".fai"):
        os.remove(fasta + ".fai")
    start = time.time()
    _ = NvFaidx(fasta)
    end = time.time()
    elapsed_nvfaidx = end - start

    print(f"pyfaidx: {elapsed_pyfaidx=}")
    print(f"nvfaidx: {elapsed_nvfaidx=}")
    print(f"nvfaidx faster by: {elapsed_pyfaidx/elapsed_nvfaidx=}")

def measure_query_time():
    '''Observed perf:

    2.3x faster nvfaidx when doing queries through our SequenceAccessor implementation in python land.
    '''
    import time
    num_iters = 1000
    fasta = create_test_fasta(num_seqs=10, seq_length=200000)

    # So we are a little slower
    fasta_idx = NvFaidx(fasta)
    start = time.time()
    for i in range(num_iters):
        query_res = fasta_idx['contig1'][150000:160000]
    end= time.time()
    elapsed_nvfaidx = end - start


    fasta_idx = pyfaidx.Fasta(fasta)
    start = time.time()
    for i in range(num_iters):
        query_res = fasta_idx['contig1'][150000:160000]
    end= time.time()
    elapsed_pyfaidx = end - start

    print(f"pyfaidx query/s: {elapsed_pyfaidx/num_iters=}")
    print(f"nvfaidx query/s: {elapsed_nvfaidx/num_iters=}")
    print(f"nvfaidx faster by: {elapsed_pyfaidx/elapsed_nvfaidx=}")

# Utility function
def create_test_fasta(num_seqs=2, seq_length=1000):
    """
    Creates a FASTA file with random sequences.
    
    Args:
        num_seqs (int): Number of sequences to include in the FASTA file.
        seq_length (int): Length of each sequence.
    
    Returns:
        str: File path to the generated FASTA file.
    """
    temp_dir = tempfile.mkdtemp()
    fasta_path = os.path.join(temp_dir, "test.fasta")
    
    with open(fasta_path, "w") as fasta_file:
        for i in range(1, num_seqs + 1):
            # Write the header
            fasta_file.write(f">contig{i}\n")
            
            # Generate a random sequence of the specified length
            sequence = ''.join(random.choices("ACGT", k=seq_length))
            
            # Split the sequence into lines of 60 characters for FASTA formatting
            for j in range(0, len(sequence), 80):
                fasta_file.write(sequence[j:j+80] + "\n")
    
    return fasta_path

measure_query_time()
measure_index_creation_time()
