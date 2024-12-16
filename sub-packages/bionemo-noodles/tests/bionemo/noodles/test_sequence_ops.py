from bionemo.noodles import reverse_sequence, complement_sequence, transcribe_sequence, back_transcribe_sequence, upper
from bionemo.noodles.nvfaidx import NvFaidx
import pytest
import pathlib

@pytest.fixture
def sample_fasta():
    return str(pathlib.Path(__file__).parent.parent.parent / "bionemo/noodles/data/sample.fasta")

def test_reverse_sequence():
    assert reverse_sequence("ACGTACGTACGT") == "TGCATGCATGCA"

def test_reverse_sequence_equivalence(sample_fasta):
    idx = NvFaidx(sample_fasta)
    print(idx['chr1'])
    complement_sequence(idx['chr1'].sequence())
    reverse_sequence(idx['chr1'].sequence())
    transcribe_sequence(idx['chr1'].sequence())
    back_transcribe_sequence(idx['chr1'].sequence())
    upper(idx['chr1'].sequence())

def test_complement_sequence():
    assert complement_sequence("ACGTACGTACGT") == "TGCATGCATGCA"
    assert complement_sequence(complement_sequence("ACGTACGTACGT")) == "ACGTACGTACGT"

def test_complement_sequence_equivalence():
    ...

def test_transcribe_sequence():
    assert transcribe_sequence("ACGTACGTACGT") == "ACGUACGUACGU"
    assert back_transcribe_sequence(transcribe_sequence("ACGTACGTACGT")) == "ACGTACGTACGT"

def test_back_transcribe_sequence_equivalence():
    ...

def test_back_transcribe_sequence():
    assert back_transcribe_sequence("ACGUACGUACGU") == "ACGTACGTACGT"
    assert transcribe_sequence(back_transcribe_sequence("ACGUACGUACGU")) == "ACGUACGUACGU"

def test_upper_sequence():
    assert upper("acgtacgtacgt") == "ACGTACGTACGT"