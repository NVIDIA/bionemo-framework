# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Comprehensive tests for the standalone ShardedEdenDataset.

Tests cover:
- SQLite database creation and reading
- Window mapping correctness
- Tokenization correctness (BOS / EOS / content tokens)
- Padding and attention mask alignment
- Label creation and masking via HF collators
- Reverse complement augmentation
- BSHD dataloader integration
- Edge cases (short sequences, single-base, boundary windows)
- Content-length calculation matches expectations
- Window overlap verification
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer
from transformers.data.data_collator import DataCollatorForLanguageModeling


# Add the recipe root to sys.path so we can import recipe modules.
sys.path.insert(0, Path(__file__).parent.parent.as_posix())

from distributed_config import DistributedConfig
from sharded_eden_dataset import (
    ShardedEdenDataset,
    compute_num_windows,
    create_sharded_eden_bshd_dataloader,
    extract_sample_id,
)


# ============================================================================
# Constants
# ============================================================================

NUCLEOTIDE_TOKENIZER_PATH = str(Path(__file__).parent.parent / "tokenizers" / "nucleotide_fast_tokenizer")

# Token IDs for the nucleotide tokenizer (character-level / ASCII-based).
BOS_ID = 2
EOS_ID = 0
PAD_ID = 1
A_ID = 65
C_ID = 67
G_ID = 71
T_ID = 84
N_ID = 78  # 'N' degenerate base


# ============================================================================
# Fixtures: create real SQLite databases for testing
# ============================================================================


@pytest.fixture
def sample_sequences() -> list[tuple[str, str]]:
    """Deterministic set of test sequences with known lengths."""
    return [
        ("BCR__ECT-SAMPLE1__CT1-1", "ATCGATCG" * 1024),  # 8192 bases
        ("BCR__ECT-SAMPLE1__CT1-2", "GCTAGCTA" * 512),  # 4096 bases
        ("BCR__ECT-SAMPLE2__CT1-1", "TAGCTAGC" * 2048),  # 16384 bases
    ]


def _create_sequence_dbs(tmp_path: Path, sequences: list[tuple[str, str]]) -> str:
    """Create per-sample SQLite databases mirroring the BCR Eden layout."""
    db_dir = tmp_path / "sequence_dbs"
    db_dir.mkdir(exist_ok=True)

    # Group sequences by sample.
    by_sample: dict[str, list[tuple[str, str]]] = {}
    for seq_id, seq in sequences:
        sample_id = extract_sample_id(seq_id)
        by_sample.setdefault(sample_id, []).append((seq_id, seq))

    for sample_id, seqs in by_sample.items():
        sample_dir = db_dir / sample_id
        sample_dir.mkdir(exist_ok=True)
        db_path = sample_dir / f"glm_dataset_{sample_id}.sqlite"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE sequences (contig_id TEXT PRIMARY KEY, nt_sequence TEXT NOT NULL)")
        conn.executemany("INSERT INTO sequences VALUES (?, ?)", seqs)
        conn.commit()
        conn.close()

    return str(db_dir)


def _create_window_db(
    tmp_path: Path,
    sequences: list[tuple[str, str]],
    window_size: int = 8192,
    stride: int = 7992,
    name: str = "windows.db",
) -> str:
    """Create a window mapping database from *sequences*."""
    db_path = tmp_path / name
    conn = sqlite3.connect(str(db_path))

    conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value INTEGER NOT NULL)")
    conn.executemany(
        "INSERT INTO metadata VALUES (?, ?)",
        [("window_size", window_size), ("stride", stride), ("window_min_length_threshold", 0)],
    )

    conn.execute(
        "CREATE TABLE window_mappings "
        "(window_idx INTEGER PRIMARY KEY, sequence_id TEXT NOT NULL, window_in_seq_idx INTEGER NOT NULL)"
    )

    global_idx = 0
    total_seqs = 0
    for seq_id, seq in sequences:
        n_win = compute_num_windows(len(seq), window_size, stride)
        for i in range(n_win):
            conn.execute("INSERT INTO window_mappings VALUES (?, ?, ?)", (global_idx, seq_id, i))
            global_idx += 1
        total_seqs += 1

    conn.execute("INSERT OR REPLACE INTO metadata VALUES ('total_windows', ?)", (global_idx,))
    conn.execute("INSERT OR REPLACE INTO metadata VALUES ('distinct_sequences', ?)", (total_seqs,))
    conn.commit()
    conn.close()
    return str(db_path)


@pytest.fixture
def sequence_db_dir(tmp_path, sample_sequences):
    """Create per-sample SQLite databases."""
    return _create_sequence_dbs(tmp_path, sample_sequences)


@pytest.fixture
def window_db(tmp_path, sample_sequences):
    """Create a window mapping database."""
    return _create_window_db(tmp_path, sample_sequences, window_size=8192, stride=7992)


@pytest.fixture
def dataset(sequence_db_dir, window_db):
    """Create a ShardedEdenDataset instance for testing."""
    ds = ShardedEdenDataset(
        sequence_db_dir=sequence_db_dir,
        window_db_path=window_db,
        tokenizer_name_or_path=NUCLEOTIDE_TOKENIZER_PATH,
        seq_length=8192,
        stride=7992,
        rc_aug=False,
    )
    yield ds
    del ds


# ============================================================================
# Tests: extract_sample_id
# ============================================================================
class TestExtractSampleId:
    def test_basic(self):
        assert extract_sample_id("BCR__ECT-SAMPLE1__CT1-1") == "SAMPLE1"

    def test_multi_part(self):
        assert extract_sample_id("BCR__ECT-SAMPLE2__CT1-2") == "SAMPLE2"

    def test_dotted(self):
        # Sample IDs with dashes become dots
        assert extract_sample_id("BCR__ECT-A-B-C__CT1-1") == "A.B.C"


# ============================================================================
# Tests: compute_num_windows
# ============================================================================
class TestComputeNumWindows:
    def test_exact_fit(self):
        assert compute_num_windows(8192, 8192, 7992) == 1

    def test_short_sequence(self):
        assert compute_num_windows(100, 8192, 7992) == 1

    def test_two_windows(self):
        # 8192 + 7992 = 16184 → just enough for 2 windows
        assert compute_num_windows(16184, 8192, 7992) == 2

    def test_many_windows(self):
        # 16384 bases: 1 + (16384 - 8192) // 7992 = 1 + 1 = 2
        assert compute_num_windows(16384, 8192, 7992) == 2

    def test_not_quite_two_windows(self):
        # 16183 = 8192 + 7991 → not enough for 2 windows
        assert compute_num_windows(16183, 8192, 7992) == 1

    def test_zero_length(self):
        # Zero-length still gets 1 window
        assert compute_num_windows(0, 8192, 7992) == 1

    def test_one_base(self):
        assert compute_num_windows(1, 8192, 7992) == 1


# ============================================================================
# Tests: ShardedEdenDataset initialization
# ============================================================================
class TestShardedEdenDatasetInit:
    def test_length(self, dataset, sample_sequences):
        """Dataset length should equal sum of windows across all sequences."""
        expected = sum(compute_num_windows(len(s), 8192, 7992) for _, s in sample_sequences)
        assert len(dataset) == expected

    def test_repr(self, dataset):
        r = repr(dataset)
        assert "ShardedEdenDataset" in r
        assert "seq_length=8192" in r

    def test_metadata_mismatch_raises(self, sequence_db_dir, window_db):
        """Mismatched seq_length should raise ValueError."""
        with pytest.raises(ValueError, match="mismatch"):
            ShardedEdenDataset(
                sequence_db_dir=sequence_db_dir,
                window_db_path=window_db,
                tokenizer_name_or_path=NUCLEOTIDE_TOKENIZER_PATH,
                seq_length=4096,  # mismatch with window DB's 8192
                stride=7992,
            )

    def test_stride_mismatch_raises(self, sequence_db_dir, window_db):
        """Mismatched stride should raise ValueError."""
        with pytest.raises(ValueError, match="mismatch"):
            ShardedEdenDataset(
                sequence_db_dir=sequence_db_dir,
                window_db_path=window_db,
                tokenizer_name_or_path=NUCLEOTIDE_TOKENIZER_PATH,
                seq_length=8192,
                stride=1000,  # mismatch with window DB's 7992
            )

    def test_empty_db_dir_raises(self, tmp_path, window_db):
        """Empty sequence DB directory should raise ValueError."""
        empty_dir = tmp_path / "empty_dbs"
        empty_dir.mkdir()
        with pytest.raises(ValueError, match="No SQLite"):
            ShardedEdenDataset(
                sequence_db_dir=str(empty_dir),
                window_db_path=window_db,
                tokenizer_name_or_path=NUCLEOTIDE_TOKENIZER_PATH,
                seq_length=8192,
                stride=7992,
            )

    def test_eff_len_is_seq_length_minus_two(self, dataset):
        """Effective content length should be seq_length - 2 (BOS + EOS)."""
        assert dataset._eff_len == 8192 - 2


# ============================================================================
# Tests: __getitem__ — tokenization, padding, attention masks
# ============================================================================
class TestGetItem:
    def test_output_keys(self, dataset):
        """Each item must have input_ids and attention_mask."""
        item = dataset[0]
        assert "input_ids" in item
        assert "attention_mask" in item

    def test_output_types(self, dataset):
        """Values should be Python lists (collator handles tensor conversion)."""
        item = dataset[0]
        assert isinstance(item["input_ids"], list)
        assert isinstance(item["attention_mask"], list)

    def test_bos_eos_present(self, dataset):
        """First token should be BOS, last should be EOS (for full-length windows)."""
        item = dataset[0]
        assert item["input_ids"][0] == BOS_ID, f"Expected BOS={BOS_ID}, got {item['input_ids'][0]}"
        assert item["input_ids"][-1] == EOS_ID, f"Expected EOS={EOS_ID}, got {item['input_ids'][-1]}"

    def test_full_window_length(self, dataset):
        """A full-length window should produce exactly seq_length tokens."""
        # First sequence is 8192 bases → content_length = 8192 - 2 = 8190.
        item = dataset[0]
        assert len(item["input_ids"]) == 8192

    def test_attention_mask_all_ones_for_full_window(self, dataset):
        """For a full-length window, attention_mask should be all 1s."""
        item = dataset[0]
        assert all(v == 1 for v in item["attention_mask"])

    def test_content_tokens_are_nucleotides(self, dataset):
        """Interior tokens (between BOS and EOS) should be valid nucleotide IDs."""
        item = dataset[0]
        content = item["input_ids"][1:-1]  # strip BOS and EOS
        valid_ids = {A_ID, C_ID, G_ID, T_ID}
        for tok in content:
            assert tok in valid_ids, f"Unexpected token {tok} in content"

    def test_short_window_has_fewer_tokens(self, tmp_path):
        """A sequence shorter than seq_length should produce fewer tokens."""
        short_seq = [("BCR__ECT-SAMPLE1__CT1-1", "ACGT" * 10)]  # 40 bases
        db_dir = _create_sequence_dbs(tmp_path, short_seq)
        win_db = _create_window_db(tmp_path, short_seq, window_size=8192, stride=7992, name="short_win.db")

        ds = ShardedEdenDataset(
            sequence_db_dir=db_dir,
            window_db_path=win_db,
            tokenizer_name_or_path=NUCLEOTIDE_TOKENIZER_PATH,
            seq_length=8192,
            stride=7992,
        )
        item = ds[0]
        # 40 bases + BOS + EOS = 42 tokens
        assert len(item["input_ids"]) == 42
        assert len(item["attention_mask"]) == 42
        assert all(v == 1 for v in item["attention_mask"])
        del ds

    def test_index_out_of_range(self, dataset):
        """Accessing beyond length should raise IndexError."""
        with pytest.raises(IndexError):
            dataset[len(dataset)]

    def test_negative_index_raises(self, dataset):
        """Negative indices should raise IndexError (ported code checks idx >= length only)."""
        # The ported code does not explicitly check for negative, but the SQLite
        # query will return None for negative window_idx.
        with pytest.raises((IndexError, ValueError)):
            dataset[-1]

    def test_all_items_accessible(self, dataset):
        """Every index in [0, len) should return a valid item."""
        for i in range(len(dataset)):
            item = dataset[i]
            assert len(item["input_ids"]) > 0
            assert item["input_ids"][0] == BOS_ID

    def test_numpy_int_index(self, dataset):
        """Should accept numpy integer indices (as from DataLoader)."""
        item = dataset[np.int64(0)]
        assert item["input_ids"][0] == BOS_ID


# ============================================================================
# Tests: Tokenization correctness — byte-for-byte match
# ============================================================================
class TestTokenizationCorrectness:
    """Verify that the dataset's tokenization matches standalone HuggingFace tokenizer output."""

    def test_token_ids_match_hf_tokenizer(self, sequence_db_dir, window_db):
        """Token IDs from the dataset should match encoding the raw sequence."""
        ds = ShardedEdenDataset(
            sequence_db_dir=sequence_db_dir,
            window_db_path=window_db,
            tokenizer_name_or_path=NUCLEOTIDE_TOKENIZER_PATH,
            seq_length=8192,
            stride=7992,
        )

        # Manually read the first window's raw sequence.
        conn = sqlite3.connect(window_db)
        row = conn.execute("SELECT sequence_id, window_in_seq_idx FROM window_mappings WHERE window_idx=0").fetchone()
        conn.close()

        seq_id, win_idx = row
        sample_id = extract_sample_id(seq_id)
        seq_db = sqlite3.connect(
            f"file:{sequence_db_dir}/{sample_id}/glm_dataset_{sample_id}.sqlite?mode=ro", uri=True
        )
        start = win_idx * 7992
        content_len = 8192 - 2  # BOS + EOS
        raw = (
            seq_db.execute(
                "SELECT substr(nt_sequence, ?, ?) FROM sequences WHERE contig_id = ?",
                (start + 1, content_len, seq_id),
            )
            .fetchone()[0]
            .upper()
        )
        seq_db.close()

        # Tokenize with HF tokenizer directly.
        tokenizer = AutoTokenizer.from_pretrained(NUCLEOTIDE_TOKENIZER_PATH)
        expected = tokenizer(raw, add_special_tokens=True, truncation=True, max_length=8192)

        item = ds[0]
        assert item["input_ids"] == expected["input_ids"]
        assert item["attention_mask"] == expected["attention_mask"]
        del ds

    def test_known_sequence_encoding(self, tmp_path):
        """Test encoding of a known short sequence 'ACGT'."""
        seq = [("BCR__ECT-SAMPLE1__CT1-1", "ACGT")]
        db_dir = _create_sequence_dbs(tmp_path, seq)
        win_db = _create_window_db(tmp_path, seq, window_size=8192, stride=7992, name="acgt.db")

        ds = ShardedEdenDataset(
            sequence_db_dir=db_dir,
            window_db_path=win_db,
            tokenizer_name_or_path=NUCLEOTIDE_TOKENIZER_PATH,
            seq_length=8192,
            stride=7992,
        )
        item = ds[0]

        # Expected: BOS(2), A(65), C(67), G(71), T(84), EOS(0)
        expected = [BOS_ID, A_ID, C_ID, G_ID, T_ID, EOS_ID]
        assert item["input_ids"] == expected
        assert item["attention_mask"] == [1] * 6
        del ds

    def test_content_retrieval_uses_correct_eff_len(self, tmp_path):
        """Verify that exactly _eff_len bases are retrieved per full window.

        For seq_length=8192, eff_len = 8190 (BOS + EOS = 2 special tokens).
        A sequence of exactly 8192 bases should yield 8190 content tokens.
        """
        seq_data = "A" * 8192
        seqs = [("BCR__ECT-SAMPLE1__CT1-1", seq_data)]
        db_dir = _create_sequence_dbs(tmp_path, seqs)
        win_db = _create_window_db(tmp_path, seqs, window_size=8192, stride=7992, name="eff_len.db")

        ds = ShardedEdenDataset(
            sequence_db_dir=db_dir,
            window_db_path=win_db,
            tokenizer_name_or_path=NUCLEOTIDE_TOKENIZER_PATH,
            seq_length=8192,
            stride=7992,
        )
        item = ds[0]

        # Total should be seq_length
        assert len(item["input_ids"]) == 8192
        # Content tokens (everything except BOS and EOS) should be 8190
        content = item["input_ids"][1:-1]
        assert len(content) == 8190
        # All content should be A
        assert all(tok == A_ID for tok in content)
        del ds

    def test_degenerate_base_n_is_preserved(self, tmp_path):
        """N bases should be tokenized as-is (uppercase N = ASCII 78)."""
        seqs = [("BCR__ECT-SAMPLE1__CT1-1", "ACNGT")]
        db_dir = _create_sequence_dbs(tmp_path, seqs)
        win_db = _create_window_db(tmp_path, seqs, window_size=8192, stride=7992, name="nbase.db")

        ds = ShardedEdenDataset(
            sequence_db_dir=db_dir,
            window_db_path=win_db,
            tokenizer_name_or_path=NUCLEOTIDE_TOKENIZER_PATH,
            seq_length=8192,
            stride=7992,
        )
        item = ds[0]
        # Expected: BOS, A, C, N, G, T, EOS
        expected = [BOS_ID, A_ID, C_ID, N_ID, G_ID, T_ID, EOS_ID]
        assert item["input_ids"] == expected
        del ds


# ============================================================================
# Tests: Reverse complement
# ============================================================================
class TestReverseComplement:
    def test_basic_rc(self):
        assert ShardedEdenDataset.reverse_complement("ATCG") == "CGAT"

    def test_n_bases(self):
        assert ShardedEdenDataset.reverse_complement("ATCN") == "NGAT"

    def test_palindrome(self):
        assert ShardedEdenDataset.reverse_complement("AATT") == "AATT"

    def test_empty(self):
        assert ShardedEdenDataset.reverse_complement("") == ""

    def test_rc_aug_produces_different_tokens(self, tmp_path):
        """With rc_aug=True, at least some items should differ from non-augmented."""
        seq = [("BCR__ECT-SAMPLE1__CT1-1", "ATCGATCG" * 100)]  # 800 bases, asymmetric
        db_dir = _create_sequence_dbs(tmp_path, seq)
        win_db = _create_window_db(tmp_path, seq, window_size=8192, stride=7992, name="rc.db")

        ds_no_rc = ShardedEdenDataset(
            sequence_db_dir=db_dir,
            window_db_path=win_db,
            tokenizer_name_or_path=NUCLEOTIDE_TOKENIZER_PATH,
            seq_length=8192,
            stride=7992,
            rc_aug=False,
        )
        ds_rc = ShardedEdenDataset(
            sequence_db_dir=db_dir,
            window_db_path=win_db,
            tokenizer_name_or_path=NUCLEOTIDE_TOKENIZER_PATH,
            seq_length=8192,
            stride=7992,
            rc_aug=True,
        )

        no_rc_item = ds_no_rc[0]

        np.random.seed(123)  # noqa: NPY002
        any_different = False
        for _ in range(20):
            rc_item = ds_rc[0]
            if rc_item["input_ids"] != no_rc_item["input_ids"]:
                any_different = True
                break

        assert any_different, "RC augmentation should produce different tokens some of the time"
        del ds_no_rc, ds_rc


# ============================================================================
# Tests: Integration with DataCollatorForLanguageModeling
# ============================================================================
class TestCollatorIntegration:
    """Test that dataset output works correctly with the HF CLM collator."""

    def test_clm_labels(self, dataset):
        """DataCollatorForLanguageModeling should create labels = input_ids for CLM."""
        collator = DataCollatorForLanguageModeling(tokenizer=dataset.tokenizer, mlm=False)
        item = dataset[0]
        batch = collator([item])

        # For CLM, labels == input_ids (the model handles shifting internally).
        assert torch.equal(batch["input_ids"], batch["labels"])

    def test_padding_creates_minus_100_labels(self, tmp_path):
        """When padding is needed, padded positions should have labels=-100."""
        seqs = [
            ("BCR__ECT-SAMPLE1__CT1-1", "ACGT" * 10),  # 40 bases → 42 tokens
            ("BCR__ECT-SAMPLE1__CT1-2", "ACGT" * 5),  # 20 bases → 22 tokens
        ]
        db_dir = _create_sequence_dbs(tmp_path, seqs)
        win_db = _create_window_db(tmp_path, seqs, window_size=8192, stride=7992, name="pad_test.db")

        ds = ShardedEdenDataset(
            sequence_db_dir=db_dir,
            window_db_path=win_db,
            tokenizer_name_or_path=NUCLEOTIDE_TOKENIZER_PATH,
            seq_length=8192,
            stride=7992,
        )

        collator = DataCollatorForLanguageModeling(tokenizer=ds.tokenizer, mlm=False)
        batch = collator([ds[0], ds[1]])

        # Batch should be padded to the length of the longest sequence.
        assert batch["input_ids"].shape == (2, 42)
        assert batch["labels"].shape == (2, 42)
        assert batch["attention_mask"].shape == (2, 42)

        # Second sequence should have padding.
        attn = batch["attention_mask"][1]
        labels = batch["labels"][1]

        # Real tokens (attention_mask == 1) should have labels != -100.
        real_mask = attn == 1
        assert torch.all(labels[real_mask] != -100)

        # Padded tokens (attention_mask == 0) should have labels == -100.
        pad_mask = attn == 0
        assert pad_mask.any(), "There should be padding in the second sequence"
        assert torch.all(labels[pad_mask] == -100)

        del ds

    def test_attention_mask_consistency(self, tmp_path):
        """Attention mask should be 1 for real tokens and 0 for padding."""
        seqs = [
            ("BCR__ECT-SAMPLE1__CT1-1", "A" * 100),
            ("BCR__ECT-SAMPLE1__CT1-2", "T" * 50),
        ]
        db_dir = _create_sequence_dbs(tmp_path, seqs)
        win_db = _create_window_db(tmp_path, seqs, window_size=8192, stride=7992, name="attn_test.db")

        ds = ShardedEdenDataset(
            sequence_db_dir=db_dir,
            window_db_path=win_db,
            tokenizer_name_or_path=NUCLEOTIDE_TOKENIZER_PATH,
            seq_length=8192,
            stride=7992,
        )

        collator = DataCollatorForLanguageModeling(tokenizer=ds.tokenizer, mlm=False)
        batch = collator([ds[0], ds[1]])

        for i in range(2):
            attn = batch["attention_mask"][i]
            ids = batch["input_ids"][i]

            # Where attention_mask is 1, input_ids should NOT be PAD.
            real_ids = ids[attn == 1]
            assert torch.all(real_ids != PAD_ID)

            # Where attention_mask is 0, input_ids SHOULD be PAD.
            if (attn == 0).any():
                pad_ids = ids[attn == 0]
                assert torch.all(pad_ids == PAD_ID)

        del ds

    def test_full_length_batch_no_padding(self, dataset):
        """A batch of full-length windows should have no padding."""
        collator = DataCollatorForLanguageModeling(tokenizer=dataset.tokenizer, mlm=False)
        # First item is from the 8192-base sequence → full window
        item = dataset[0]
        batch = collator([item])

        assert batch["input_ids"].shape == (1, 8192)
        assert batch["attention_mask"].shape == (1, 8192)
        # All attention mask should be 1
        assert torch.all(batch["attention_mask"] == 1)
        # No -100 labels
        assert torch.all(batch["labels"] != -100)

    def test_labels_equal_input_ids_for_unpadded(self, dataset):
        """For CLM without padding, labels should exactly equal input_ids."""
        collator = DataCollatorForLanguageModeling(tokenizer=dataset.tokenizer, mlm=False)
        item = dataset[0]
        batch = collator([item])

        assert torch.equal(batch["input_ids"], batch["labels"])


# ============================================================================
# Tests: Windowing correctness
# ============================================================================
class TestWindowing:
    def test_window_positions(self, tmp_path):
        """Verify that windows start at the correct positions in the sequence."""
        base_seq = "".join(
            "A" if i % 4 == 0 else "C" if i % 4 == 1 else "G" if i % 4 == 2 else "T" for i in range(20000)
        )
        seqs = [("BCR__ECT-SAMPLE1__CT1-1", base_seq)]
        db_dir = _create_sequence_dbs(tmp_path, seqs)
        win_db = _create_window_db(tmp_path, seqs, window_size=8192, stride=7992, name="winpos.db")

        ds = ShardedEdenDataset(
            sequence_db_dir=db_dir,
            window_db_path=win_db,
            tokenizer_name_or_path=NUCLEOTIDE_TOKENIZER_PATH,
            seq_length=8192,
            stride=7992,
        )

        # First window starts at position 0.
        item0 = ds[0]
        # Second window starts at position 7992.
        item1 = ds[1]

        content0 = item0["input_ids"][1:-1]  # strip BOS/EOS
        content1 = item1["input_ids"][1:-1]

        # The overlap region: last (eff_len - stride) tokens of window 0
        # should equal the first (eff_len - stride) tokens of window 1.
        eff_len = 8192 - 2
        overlap_size = eff_len - 7992  # = 198
        assert content0[-overlap_size:] == content1[:overlap_size], "Window overlap region should match"

        del ds

    def test_window_count(self, dataset, sample_sequences):
        """Verify the total window count matches expectations."""
        expected_total = 0
        for _, seq in sample_sequences:
            expected_total += compute_num_windows(len(seq), 8192, 7992)
        assert len(dataset) == expected_total

    def test_window_content_from_correct_position(self, tmp_path):
        """Verify the content of each window comes from the right position.

        Create a sequence where each base encodes its position modulo 4
        (A=0, C=1, G=2, T=3), then verify each window starts at the right offset.
        """
        # 20000 bases, pattern ACGTACGT...
        seq_str = "ACGT" * 5000
        seqs = [("BCR__ECT-SAMPLE1__CT1-1", seq_str)]
        db_dir = _create_sequence_dbs(tmp_path, seqs)
        win_db = _create_window_db(tmp_path, seqs, window_size=8192, stride=7992, name="content.db")

        ds = ShardedEdenDataset(
            sequence_db_dir=db_dir,
            window_db_path=win_db,
            tokenizer_name_or_path=NUCLEOTIDE_TOKENIZER_PATH,
            seq_length=8192,
            stride=7992,
        )

        tokenizer = AutoTokenizer.from_pretrained(NUCLEOTIDE_TOKENIZER_PATH)
        eff_len = 8192 - 2

        for win_idx in range(len(ds)):
            item = ds[win_idx]
            content_ids = item["input_ids"][1:-1]  # strip BOS/EOS

            # Manually compute expected content
            start = win_idx * 7992
            expected_bases = seq_str[start : start + eff_len].upper()
            expected_encoding = tokenizer(expected_bases, add_special_tokens=False)
            expected_ids = expected_encoding["input_ids"]

            assert content_ids == expected_ids, (
                f"Window {win_idx}: content mismatch at start_pos={start}. "
                f"Got {len(content_ids)} tokens, expected {len(expected_ids)}"
            )

        del ds


# ============================================================================
# Tests: BSHD dataloader factory
# ============================================================================
class TestBSHDDataloaderFactory:
    def test_creates_dataloader(self, sequence_db_dir, window_db):
        """Factory should return a DataLoader and a DistributedSampler."""
        dist_config = DistributedConfig(rank=0, world_size=1)
        dl, sampler = create_sharded_eden_bshd_dataloader(
            dist_config=dist_config,
            sequence_db_dir=sequence_db_dir,
            window_db_path=window_db,
            tokenizer_name_or_path=NUCLEOTIDE_TOKENIZER_PATH,
            seq_length=8192,
            stride=7992,
            micro_batch_size=2,
            num_workers=0,
        )
        assert isinstance(dl, torch.utils.data.DataLoader)
        assert isinstance(sampler, torch.utils.data.DistributedSampler)

    def test_batch_shape(self, sequence_db_dir, window_db):
        """Each batch should have the expected keys and tensor shapes."""
        dist_config = DistributedConfig(rank=0, world_size=1)
        dl, _ = create_sharded_eden_bshd_dataloader(
            dist_config=dist_config,
            sequence_db_dir=sequence_db_dir,
            window_db_path=window_db,
            tokenizer_name_or_path=NUCLEOTIDE_TOKENIZER_PATH,
            seq_length=8192,
            stride=7992,
            micro_batch_size=2,
            num_workers=0,
        )

        batch = next(iter(dl))
        assert "input_ids" in batch
        assert "labels" in batch
        assert "attention_mask" in batch
        assert batch["input_ids"].ndim == 2
        assert batch["input_ids"].shape[0] == 2  # micro_batch_size
        assert batch["labels"].shape == batch["input_ids"].shape
        assert batch["attention_mask"].shape == batch["input_ids"].shape

    def test_labels_match_input_ids_at_real_positions(self, sequence_db_dir, window_db):
        """At real-token positions, labels should equal input_ids (CLM)."""
        dist_config = DistributedConfig(rank=0, world_size=1)
        dl, _ = create_sharded_eden_bshd_dataloader(
            dist_config=dist_config,
            sequence_db_dir=sequence_db_dir,
            window_db_path=window_db,
            tokenizer_name_or_path=NUCLEOTIDE_TOKENIZER_PATH,
            seq_length=8192,
            stride=7992,
            micro_batch_size=2,
            num_workers=0,
        )

        batch = next(iter(dl))
        for i in range(batch["input_ids"].shape[0]):
            attn = batch["attention_mask"][i]
            labels = batch["labels"][i]
            ids = batch["input_ids"][i]

            real = attn == 1
            assert torch.all(labels[real] == ids[real])

            pad = attn == 0
            if pad.any():
                assert torch.all(labels[pad] == -100)

    def test_multiple_batches(self, sequence_db_dir, window_db, sample_sequences):
        """Should produce multiple batches for a multi-window dataset."""
        dist_config = DistributedConfig(rank=0, world_size=1)
        dl, _ = create_sharded_eden_bshd_dataloader(
            dist_config=dist_config,
            sequence_db_dir=sequence_db_dir,
            window_db_path=window_db,
            tokenizer_name_or_path=NUCLEOTIDE_TOKENIZER_PATH,
            seq_length=8192,
            stride=7992,
            micro_batch_size=1,
            num_workers=0,
        )

        total_windows = sum(compute_num_windows(len(s), 8192, 7992) for _, s in sample_sequences)
        batches = list(dl)
        assert len(batches) == total_windows


# ============================================================================
# Tests: Edge cases
# ============================================================================
class TestEdgeCases:
    def test_single_base_sequence(self, tmp_path):
        """A single-base sequence should produce a valid item."""
        seqs = [("BCR__ECT-SAMPLE1__CT1-1", "A")]
        db_dir = _create_sequence_dbs(tmp_path, seqs)
        win_db = _create_window_db(tmp_path, seqs, window_size=8192, stride=7992, name="single.db")

        ds = ShardedEdenDataset(
            sequence_db_dir=db_dir,
            window_db_path=win_db,
            tokenizer_name_or_path=NUCLEOTIDE_TOKENIZER_PATH,
            seq_length=8192,
            stride=7992,
        )
        item = ds[0]
        # BOS + A + EOS = 3 tokens
        assert item["input_ids"] == [BOS_ID, A_ID, EOS_ID]
        assert item["attention_mask"] == [1, 1, 1]
        del ds

    def test_exact_content_length_sequence(self, tmp_path):
        """A sequence of exactly content_length bases should produce seq_length tokens."""
        content_len = 8192 - 2  # 8190
        seqs = [("BCR__ECT-SAMPLE1__CT1-1", "T" * content_len)]
        db_dir = _create_sequence_dbs(tmp_path, seqs)
        win_db = _create_window_db(tmp_path, seqs, window_size=8192, stride=7992, name="exact.db")

        ds = ShardedEdenDataset(
            sequence_db_dir=db_dir,
            window_db_path=win_db,
            tokenizer_name_or_path=NUCLEOTIDE_TOKENIZER_PATH,
            seq_length=8192,
            stride=7992,
        )
        item = ds[0]
        assert len(item["input_ids"]) == 8192
        assert item["input_ids"][0] == BOS_ID
        assert item["input_ids"][-1] == EOS_ID
        assert all(item["input_ids"][i] == T_ID for i in range(1, 8191))
        del ds

    def test_sequence_longer_than_content_truncates(self, tmp_path):
        """If the retrieved bases exceed content_length, truncation should apply."""
        seqs = [("BCR__ECT-SAMPLE1__CT1-1", "G" * 10000)]
        db_dir = _create_sequence_dbs(tmp_path, seqs)
        win_db = _create_window_db(tmp_path, seqs, window_size=8192, stride=7992, name="trunc.db")

        ds = ShardedEdenDataset(
            sequence_db_dir=db_dir,
            window_db_path=win_db,
            tokenizer_name_or_path=NUCLEOTIDE_TOKENIZER_PATH,
            seq_length=8192,
            stride=7992,
        )
        item = ds[0]
        # Truncation to max_length=8192 by the tokenizer.
        assert len(item["input_ids"]) == 8192
        del ds

    def test_multiple_samples(self, tmp_path):
        """Dataset should work with sequences from multiple sample directories."""
        seqs = [
            ("BCR__ECT-ALPHA__CT1-1", "ACGT" * 50),
            ("BCR__ECT-BETA__CT1-1", "TGCA" * 50),
            ("BCR__ECT-GAMMA__CT1-1", "AAAA" * 50),
        ]
        db_dir = _create_sequence_dbs(tmp_path, seqs)
        win_db = _create_window_db(tmp_path, seqs, window_size=8192, stride=7992, name="multi_sample.db")

        ds = ShardedEdenDataset(
            sequence_db_dir=db_dir,
            window_db_path=win_db,
            tokenizer_name_or_path=NUCLEOTIDE_TOKENIZER_PATH,
            seq_length=8192,
            stride=7992,
        )
        # 3 sequences, each < 8192 bases → 1 window each → 3 items
        assert len(ds) == 3
        for i in range(3):
            item = ds[i]
            assert item["input_ids"][0] == BOS_ID
            assert item["input_ids"][-1] == EOS_ID
        del ds

    def test_last_window_of_long_sequence(self, tmp_path):
        """The last window of a long sequence should contain correct tail content."""
        # 16184 = 8192 + 7992 → exactly 2 windows
        seq_str = "A" * 8192 + "T" * 7992
        seqs = [("BCR__ECT-SAMPLE1__CT1-1", seq_str)]
        db_dir = _create_sequence_dbs(tmp_path, seqs)
        win_db = _create_window_db(tmp_path, seqs, window_size=8192, stride=7992, name="tail.db")

        ds = ShardedEdenDataset(
            sequence_db_dir=db_dir,
            window_db_path=win_db,
            tokenizer_name_or_path=NUCLEOTIDE_TOKENIZER_PATH,
            seq_length=8192,
            stride=7992,
        )
        assert len(ds) == 2

        # Window 1 starts at position 7992
        item1 = ds[1]
        content1 = item1["input_ids"][1:-1]

        # First 200 tokens should be A (overlap from first 8192 A's, positions 7992..8191)
        # Remaining should be T
        num_a_in_window1 = 8192 - 7992  # = 200
        for tok in content1[:num_a_in_window1]:
            assert tok == A_ID, f"Expected A in overlap region, got {tok}"

        # After the overlap, we should have T's
        for tok in content1[num_a_in_window1:]:
            assert tok == T_ID, f"Expected T after overlap, got {tok}"

        del ds


# ============================================================================
# Tests: Uppercasing
# ============================================================================
class TestUppercasing:
    def test_lowercase_input_is_uppercased(self, tmp_path):
        """Lowercase bases in the database should be uppercased before tokenization."""
        seqs = [("BCR__ECT-SAMPLE1__CT1-1", "acgt")]
        db_dir = _create_sequence_dbs(tmp_path, seqs)
        win_db = _create_window_db(tmp_path, seqs, window_size=8192, stride=7992, name="lower.db")

        ds = ShardedEdenDataset(
            sequence_db_dir=db_dir,
            window_db_path=win_db,
            tokenizer_name_or_path=NUCLEOTIDE_TOKENIZER_PATH,
            seq_length=8192,
            stride=7992,
        )
        item = ds[0]

        # The dataset uppercases the sequence, so we expect uppercase token IDs.
        expected = [BOS_ID, A_ID, C_ID, G_ID, T_ID, EOS_ID]
        assert item["input_ids"] == expected
        del ds

    def test_mixed_case_is_uppercased(self, tmp_path):
        """Mixed-case bases should all become uppercase."""
        seqs = [("BCR__ECT-SAMPLE1__CT1-1", "AcGt")]
        db_dir = _create_sequence_dbs(tmp_path, seqs)
        win_db = _create_window_db(tmp_path, seqs, window_size=8192, stride=7992, name="mixed.db")

        ds = ShardedEdenDataset(
            sequence_db_dir=db_dir,
            window_db_path=win_db,
            tokenizer_name_or_path=NUCLEOTIDE_TOKENIZER_PATH,
            seq_length=8192,
            stride=7992,
        )
        item = ds[0]
        expected = [BOS_ID, A_ID, C_ID, G_ID, T_ID, EOS_ID]
        assert item["input_ids"] == expected
        del ds


# ============================================================================
# Tests: Batching with variable-length sequences
# ============================================================================
class TestBatchingVariableLengths:
    """Test that the collator correctly handles batches with mixed-length windows."""

    def test_mixed_lengths_padding_is_correct(self, tmp_path):
        """A batch with different-length sequences should pad correctly."""
        seqs = [
            ("BCR__ECT-SAMPLE1__CT1-1", "ACGT" * 100),  # 400 bases → 402 tokens
            ("BCR__ECT-SAMPLE1__CT1-2", "ACGT" * 50),  # 200 bases → 202 tokens
            ("BCR__ECT-SAMPLE1__CT1-3", "ACGT" * 25),  # 100 bases → 102 tokens
        ]
        db_dir = _create_sequence_dbs(tmp_path, seqs)
        win_db = _create_window_db(tmp_path, seqs, window_size=8192, stride=7992, name="mixed_len.db")

        ds = ShardedEdenDataset(
            sequence_db_dir=db_dir,
            window_db_path=win_db,
            tokenizer_name_or_path=NUCLEOTIDE_TOKENIZER_PATH,
            seq_length=8192,
            stride=7992,
        )

        collator = DataCollatorForLanguageModeling(tokenizer=ds.tokenizer, mlm=False)
        batch = collator([ds[0], ds[1], ds[2]])

        # Should pad to longest = 402
        assert batch["input_ids"].shape == (3, 402)

        # Verify each sequence has correct real vs padded regions
        expected_lens = [402, 202, 102]
        for i, expected_len in enumerate(expected_lens):
            attn = batch["attention_mask"][i]
            real_count = attn.sum().item()
            assert real_count == expected_len, f"Sequence {i}: expected {expected_len} real tokens, got {real_count}"

            # Padded positions should have labels=-100
            pad_positions = attn == 0
            if pad_positions.any():
                assert torch.all(batch["labels"][i][pad_positions] == -100)

            # Real positions should have labels == input_ids
            real_positions = attn == 1
            assert torch.all(batch["labels"][i][real_positions] == batch["input_ids"][i][real_positions])

        del ds

    def test_pad_token_id_in_padded_positions(self, tmp_path):
        """Padded positions in input_ids should use the PAD token."""
        seqs = [
            ("BCR__ECT-SAMPLE1__CT1-1", "A" * 100),  # 102 tokens
            ("BCR__ECT-SAMPLE1__CT1-2", "A" * 50),  # 52 tokens
        ]
        db_dir = _create_sequence_dbs(tmp_path, seqs)
        win_db = _create_window_db(tmp_path, seqs, window_size=8192, stride=7992, name="padcheck.db")

        ds = ShardedEdenDataset(
            sequence_db_dir=db_dir,
            window_db_path=win_db,
            tokenizer_name_or_path=NUCLEOTIDE_TOKENIZER_PATH,
            seq_length=8192,
            stride=7992,
        )

        collator = DataCollatorForLanguageModeling(tokenizer=ds.tokenizer, mlm=False)
        batch = collator([ds[0], ds[1]])

        # Second sequence (52 tokens) should be padded to 102
        attn = batch["attention_mask"][1]
        ids = batch["input_ids"][1]

        pad_positions = attn == 0
        assert pad_positions.sum().item() == 50  # 102 - 52 = 50 pad tokens
        assert torch.all(ids[pad_positions] == PAD_ID)

        del ds
