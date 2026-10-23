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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

"""Physical circular cleanup on real published DNA, without claiming caller/search parity.

An independent forward enumerator uses the archived probe's declared convention:
first ATG after a stop, next in-frame stop, 90-1800 nt including the stop. All 50
archived fixture peptides anchor that convention. Three full genome copies give
the expected middle-copy physical ORFs without a coordinate-cut prefix artifact.
The actual inputs use the pinned upstream pseudocircular extension instead.

This is not an ORFipy or MMseqs rerun, an initiation-site assay, reverse-strand
annotation validation, or proof that the paper used the cleaned ORF inventory.
"""

import hashlib
import inspect
import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from Bio import SeqIO
from Bio.Seq import Seq

from bionemo.evo2_phage_gen.protein_evidence import remove_pseudocircular_extension_orfs


FIXTURES = Path(__file__).with_name("fixtures")
RECORD_IDS = ("s4row1", "s1row17", "s1row29", "s1row66", "s1row160")


@dataclass(frozen=True)
class ForwardORF:
    start: int
    end: int
    dna: str
    protein: str


def _forward_orfs(sequence: str, minimum: int = 90) -> list[ForwardORF]:
    """Enumerate the stated convention independently of the cleanup implementation."""
    calls = []
    for frame in range(3):
        start = None
        for position in range(frame, len(sequence) - 2, 3):
            codon = sequence[position : position + 3]
            if codon == "ATG" and start is None:
                start = position
            if codon in {"TAA", "TAG", "TGA"}:
                if start is not None and minimum <= position + 3 - start <= 1800:
                    dna = sequence[start : position + 3]
                    calls.append(ForwardORF(start, position + 3, dna, str(Seq(dna).translate(cds=True))))
                start = None
    return calls


def _physical_key(call: ForwardORF, length: int, cut: int = 0):
    # Physical coding start/stop positions distinguish identical peptides at
    # different loci; peptides alone cannot identify a genomic molecule.
    return ((call.start + cut) % length, (call.end + cut) % length, call.protein)


@pytest.fixture(scope="module")
def publication():
    payload = json.loads((FIXTURES / "arc_aai_publication.json").read_text())
    for record in payload["records"]:
        assert hashlib.sha256(record["sequence"].encode()).hexdigest() == record["sequence_sha256"]
    return payload


def _extended_sequence(upstream, root: Path, genome_id: str, sequence: str):
    root.mkdir()
    source = root / "source.fasta"
    source.write_text(f">{genome_id}\n{sequence}\n")
    output = root / "extended.fasta"
    upstream.append_upstream_of_last_frame_stop(str(source), str(output))
    extended = str(SeqIO.read(output, "fasta").seq)
    assert extended.startswith(sequence)
    return source, extended


def _write_calls(root: Path, genome_id: str, calls: list[ForwardORF]):
    dna_path, protein_path = root / "orfs.fasta", root / "proteins.fasta"
    dna_lines, protein_lines, by_id = [], [], {}
    for index, call in enumerate(calls, start=1):
        record_id = f"{genome_id}_ORF.{index}"
        header = f">{record_id} [{call.start}-{call.end}](+) type:complete length:{call.end - call.start}"
        dna_lines.extend((header, call.dna))
        protein_lines.extend((header, call.protein))
        by_id[record_id] = call
    dna_path.write_text("\n".join(dna_lines) + "\n")
    protein_path.write_text("\n".join(protein_lines) + "\n")
    return dna_path, protein_path, by_id


def _retained_calls(dna_path: Path, protein_path: Path, by_id: dict[str, ForwardORF]):
    dna = list(SeqIO.parse(dna_path, "fasta"))
    proteins = list(SeqIO.parse(protein_path, "fasta"))
    assert [record.id for record in dna] == [record.id for record in proteins]
    assert len({record.id for record in dna}) == len(dna)
    for nucleotide, protein in zip(dna, proteins, strict=True):
        expected = by_id[nucleotide.id]
        assert str(nucleotide.seq) == expected.dna
        assert str(protein.seq) == expected.protein == str(nucleotide.seq.translate(cds=True))
        assert nucleotide.description == protein.description
    return [by_id[record.id] for record in dna]


def test_archived_calls_anchor_enumerator(publication, load_script, tmp_path):
    """The independently enumerated convention recovers every archived coordinate/peptide."""
    upstream = load_script(FIXTURES / "arc_pipeline_upstream.py", "arc_circular_archived")
    archived_count = 0
    for record in publication["records"]:
        _, extended = _extended_sequence(upstream, tmp_path / record["id"], record["id"], record["sequence"])
        calls = {(call.start, call.end, call.protein) for call in _forward_orfs(extended)}
        archived = [row for row in publication["orfs"] if row["id"].rsplit("_", 1)[0] == record["id"]]
        for row in archived:
            assert row["strand"] == "+"
            assert (row["start"], row["end"], row["protein"]) in calls
            archived_count += 1
    assert archived_count == 50


@pytest.mark.parametrize("record_id", RECORD_IDS)
def test_real_cut_cleanup(publication, load_script, tmp_path, record_id):
    """Cuts must recover the independently enumerated physical inventory, with real translations."""
    upstream = load_script(FIXTURES / "arc_pipeline_upstream.py", "arc_circular_cuts")
    sequence = next(row["sequence"] for row in publication["records"] if row["id"] == record_id)
    length = len(sequence)
    # Middle-copy calls have complete preceding and following circular context.
    # No prefix-removal heuristic or production cleanup helper derives this oracle.
    expected = {
        _physical_key(call, length) for call in _forward_orfs(sequence * 3) if length <= call.start < 2 * length
    }
    assert expected
    for cut in (0, 1, 1000, length - 100):
        rotated = sequence[cut:] + sequence[:cut]
        root = tmp_path / f"cut-{cut}"
        source, extended = _extended_sequence(upstream, root, record_id, rotated)
        calls = _forward_orfs(extended)
        dna_path, protein_path, by_id = _write_calls(root, record_id, calls)
        remove_pseudocircular_extension_orfs(source, dna_path, protein_path)
        retained = _retained_calls(dna_path, protein_path, by_id)
        assert all(call.start < length for call in retained)
        observed = [_physical_key(call, length, cut) for call in retained]
        assert len(observed) == len(set(observed)) == len(expected)
        assert set(observed) == expected
        if record_id == "s4row1" and cut == 0:
            # Real 39-aa A-prefix tail and complete cross-origin 522-aa A.
            assert any((call.start, call.end) == (16, 136) and len(call.protein) == 39 for call in calls)
            assert not any((call.start, call.end) == (16, 136) for call in retained)
            assert any((call.start, call.end) == (3953, 5522) and len(call.protein) == 522 for call in retained)


def test_real_distinct_loci_survive(publication, load_script, tmp_path):
    """Real DNA with identical peptide tails must not collapse different physical stops.

    The two six-nt complete calls deliberately stress the cleanup utility below
    the maintained 72-nt and archived 90-nt caller minima. They are real translated
    sequence intervals, not a claim that either pipeline would call these proteins.
    """
    upstream = load_script(FIXTURES / "arc_pipeline_upstream.py", "arc_circular_distinct_loci")
    sequence = next(row["sequence"] for row in publication["records"] if row["id"] == "s1row29")
    cut = 1000
    rotated = sequence[cut:] + sequence[:cut]
    source, extended = _extended_sequence(upstream, tmp_path / "distinct", "s1row29", rotated)
    intervals = {(252, 258), (1539, 1545), (5188, 5647)}
    calls = [call for call in _forward_orfs(extended, minimum=6) if (call.start, call.end) in intervals]
    assert {(call.start, call.end) for call in calls} == intervals
    short = [call for call in calls if call.protein == "M"]
    spanning = next(call for call in calls if call.end > len(sequence))
    assert len(short) == 2 and len(spanning.protein) == 152
    assert all(spanning.protein.endswith(call.protein) for call in short)
    assert len({call.end % len(sequence) for call in calls}) == 3
    root = source.parent
    dna_path, protein_path, by_id = _write_calls(root, "s1row29", calls)
    remove_pseudocircular_extension_orfs(source, dna_path, protein_path)
    retained = _retained_calls(dna_path, protein_path, by_id)
    assert retained == calls
    # Explicit mutation control: peptide-only suffix removal would lose the
    # two independently translated loci despite their different physical stops.
    peptide_only = [
        call
        for call in calls
        if not any(len(other.protein) > len(call.protein) and other.protein.endswith(call.protein) for other in calls)
    ]
    assert len(peptide_only) == 1


def test_real_prefix_oracle_detects_mutant(publication, load_script, tmp_path):
    """Omitting actual prefix-tail removal must fail the independently derived inventory."""
    upstream = load_script(FIXTURES / "arc_pipeline_upstream.py", "arc_circular_prefix_mutant")
    sequence = next(row["sequence"] for row in publication["records"] if row["id"] == "s4row1")
    source, extended = _extended_sequence(upstream, tmp_path / "mutant", "s4row1", sequence)
    dna_path, protein_path, by_id = _write_calls(source.parent, "s4row1", _forward_orfs(extended))
    implementation = inspect.getsource(remove_pseudocircular_extension_orfs)
    assert implementation.count("keep_ids.remove(record_id)") == 1
    mutated = implementation.replace("keep_ids.remove(record_id)", "keep_ids.add(record_id)")
    namespace = remove_pseudocircular_extension_orfs.__globals__.copy()
    exec(compile(mutated, "<prefix-removal-mutant>", "exec"), namespace)
    namespace["remove_pseudocircular_extension_orfs"](source, dna_path, protein_path)
    retained = _retained_calls(dna_path, protein_path, by_id)
    length = len(sequence)
    expected = {
        _physical_key(call, length) for call in _forward_orfs(sequence * 3) if length <= call.start < 2 * length
    }
    observed = {_physical_key(call, length) for call in retained}
    with pytest.raises(AssertionError):
        assert observed == expected
