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

"""
Comprehensive tests for SCDL header implementation and schema compliance.

Tests all header functionality including serialization, deserialization, validation,
and compliance with the SCDL schema specification.
"""

import json
import tempfile
from pathlib import Path

import pytest

from bionemo.scdl.schema.header import (
    ArrayDType,
    ArrayInfo,
    Backend,
    FeatureIndexInfo,
    HeaderReader,
    SCDLHeader,
    create_header_from_arrays,
    merge_headers,
    validate_header_compatibility,
)
from bionemo.scdl.schema.headerutil import Endianness, HeaderSerializationError
from bionemo.scdl.schema.magic import SCDL_MAGIC_NUMBER
from bionemo.scdl.schema.version import CurrentSCDLVersion, SCDLVersion

from ._expected_version import EXPECTED_SCDL_VERSION


class TestArrayDType:
    """Test ArrayDType enum and conversion methods."""

    def test_enum_values(self):
        """Test that enum values match expected integers."""
        assert ArrayDType.UINT8_ARRAY == 1
        assert ArrayDType.UINT16_ARRAY == 2
        assert ArrayDType.UINT32_ARRAY == 3
        assert ArrayDType.UINT64_ARRAY == 4
        assert ArrayDType.FLOAT16_ARRAY == 5
        assert ArrayDType.FLOAT32_ARRAY == 6
        assert ArrayDType.FLOAT64_ARRAY == 7
        assert ArrayDType.STRING_ARRAY == 8
        assert ArrayDType.FIXED_STRING_ARRAY == 9

    def test_numpy_dtype_string(self):
        """Test numpy dtype string conversion."""
        assert ArrayDType.UINT8_ARRAY.numpy_dtype_string == "uint8"
        assert ArrayDType.UINT16_ARRAY.numpy_dtype_string == "uint16"
        assert ArrayDType.UINT32_ARRAY.numpy_dtype_string == "uint32"
        assert ArrayDType.UINT64_ARRAY.numpy_dtype_string == "uint64"
        assert ArrayDType.FLOAT16_ARRAY.numpy_dtype_string == "float16"
        assert ArrayDType.FLOAT32_ARRAY.numpy_dtype_string == "float32"
        assert ArrayDType.FLOAT64_ARRAY.numpy_dtype_string == "float64"
        assert ArrayDType.STRING_ARRAY.numpy_dtype_string == "string"
        assert ArrayDType.FIXED_STRING_ARRAY.numpy_dtype_string == "fixed_string"

    def test_from_numpy_dtype_strings(self):
        """Test conversion from numpy dtype strings."""
        assert ArrayDType.from_numpy_dtype("uint8") == ArrayDType.UINT8_ARRAY
        assert ArrayDType.from_numpy_dtype("uint16") == ArrayDType.UINT16_ARRAY
        assert ArrayDType.from_numpy_dtype("uint32") == ArrayDType.UINT32_ARRAY
        assert ArrayDType.from_numpy_dtype("uint64") == ArrayDType.UINT64_ARRAY
        assert ArrayDType.from_numpy_dtype("float16") == ArrayDType.FLOAT16_ARRAY
        assert ArrayDType.from_numpy_dtype("float32") == ArrayDType.FLOAT32_ARRAY
        assert ArrayDType.from_numpy_dtype("float64") == ArrayDType.FLOAT64_ARRAY

    def test_from_numpy_dtype_objects(self):
        """Test conversion from numpy dtype objects."""
        import numpy as np

        # Test numpy dtype instances
        assert ArrayDType.from_numpy_dtype(np.dtype("float32")) == ArrayDType.FLOAT32_ARRAY
        assert ArrayDType.from_numpy_dtype(np.dtype("float64")) == ArrayDType.FLOAT64_ARRAY
        assert ArrayDType.from_numpy_dtype(np.dtype("uint32")) == ArrayDType.UINT32_ARRAY
        assert ArrayDType.from_numpy_dtype(np.dtype("uint64")) == ArrayDType.UINT64_ARRAY

        # Test numpy type classes (this was the bug)
        assert ArrayDType.from_numpy_dtype(np.float32) == ArrayDType.FLOAT32_ARRAY
        assert ArrayDType.from_numpy_dtype(np.float64) == ArrayDType.FLOAT64_ARRAY
        assert ArrayDType.from_numpy_dtype(np.uint32) == ArrayDType.UINT32_ARRAY
        assert ArrayDType.from_numpy_dtype(np.uint64) == ArrayDType.UINT64_ARRAY

        # Test actual array dtypes (the original error case)
        arr = np.array([1.0], dtype=np.float32)
        assert ArrayDType.from_numpy_dtype(arr.dtype) == ArrayDType.FLOAT32_ARRAY

    def test_from_numpy_dtype_variations(self):
        """Test conversion from various numpy dtype format variations."""
        import numpy as np

        # Test endianness variations
        assert ArrayDType.from_numpy_dtype(np.dtype("<f4")) == ArrayDType.FLOAT32_ARRAY
        assert ArrayDType.from_numpy_dtype(np.dtype(">f4")) == ArrayDType.FLOAT32_ARRAY
        assert ArrayDType.from_numpy_dtype(np.dtype("<f8")) == ArrayDType.FLOAT64_ARRAY
        assert ArrayDType.from_numpy_dtype(np.dtype("<u4")) == ArrayDType.UINT32_ARRAY
        assert ArrayDType.from_numpy_dtype(np.dtype("<u8")) == ArrayDType.UINT64_ARRAY

    def test_from_numpy_dtype_invalid(self):
        """Test that invalid dtypes raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported numpy dtype"):
            ArrayDType.from_numpy_dtype("invalid_dtype")

        with pytest.raises(ValueError, match="Unsupported numpy dtype"):
            ArrayDType.from_numpy_dtype("complex128")


class TestBackend:
    """Test Backend enum."""

    def test_backend_values(self):
        """Test backend enum values."""
        assert Backend.MEMMAP_V0 == 1


class TestArrayInfo:
    """Test ArrayInfo class functionality."""

    def test_basic_creation(self):
        """Test basic ArrayInfo creation."""
        array_info = ArrayInfo(name="test_array.dat", length=1000, dtype=ArrayDType.FLOAT32_ARRAY)
        assert array_info.name == "test_array.dat"
        assert array_info.length == 1000
        assert array_info.dtype == ArrayDType.FLOAT32_ARRAY
        assert array_info.shape is None

    def test_creation_with_shape(self):
        """Test ArrayInfo creation with shape."""
        array_info = ArrayInfo(name="shaped_array.dat", length=2000, dtype=ArrayDType.UINT32_ARRAY, shape=(100, 20))
        assert array_info.shape == (100, 20)

    def test_validation_empty_name(self):
        """Test validation fails for empty name."""
        array_info = ArrayInfo(name="", length=100, dtype=ArrayDType.UINT8_ARRAY)
        with pytest.raises(HeaderSerializationError, match="Array name cannot be empty"):
            array_info._validate()

    def test_validation_whitespace_name(self):
        """Test validation fails for whitespace-only name."""
        array_info = ArrayInfo(name="   ", length=100, dtype=ArrayDType.UINT8_ARRAY)
        with pytest.raises(HeaderSerializationError, match="Array name cannot be empty"):
            array_info._validate()

    def test_validation_negative_length(self):
        """Test validation fails for negative length."""
        array_info = ArrayInfo(name="test.dat", length=-1, dtype=ArrayDType.UINT8_ARRAY)
        with pytest.raises(HeaderSerializationError, match="Array length cannot be negative"):
            array_info._validate()

    def test_validation_empty_shape(self):
        """Test validation fails for empty shape."""
        array_info = ArrayInfo(name="test.dat", length=100, dtype=ArrayDType.UINT8_ARRAY, shape=())
        with pytest.raises(HeaderSerializationError, match="Shape cannot be empty when specified"):
            array_info._validate()

    def test_validation_zero_shape_dimension(self):
        """Test validation fails for zero shape dimension."""
        array_info = ArrayInfo(name="test.dat", length=100, dtype=ArrayDType.UINT8_ARRAY, shape=(10, 0, 5))
        with pytest.raises(HeaderSerializationError, match="Shape dimension 1 must be positive"):
            array_info._validate()

    def test_serialization_without_shape(self):
        """Test serialization of ArrayInfo without shape."""
        from bionemo.scdl.schema.headerutil import BinaryHeaderCodec

        array_info = ArrayInfo(name="test.dat", length=1000, dtype=ArrayDType.FLOAT32_ARRAY)

        codec = BinaryHeaderCodec(Endianness.NETWORK)
        serialized = array_info.serialize(codec)

        # Verify we can deserialize it back
        deserialized, consumed = ArrayInfo.deserialize(codec, serialized)

        assert deserialized.name == array_info.name
        assert deserialized.length == array_info.length
        assert deserialized.dtype == array_info.dtype
        assert deserialized.shape is None
        assert consumed == len(serialized)

    def test_serialization_with_shape(self):
        """Test serialization of ArrayInfo with shape."""
        from bionemo.scdl.schema.headerutil import BinaryHeaderCodec

        array_info = ArrayInfo(name="shaped.dat", length=2000, dtype=ArrayDType.UINT16_ARRAY, shape=(100, 20))

        codec = BinaryHeaderCodec(Endianness.NETWORK)
        serialized = array_info.serialize(codec)

        # Verify we can deserialize it back
        deserialized, consumed = ArrayInfo.deserialize(codec, serialized)

        assert deserialized.name == array_info.name
        assert deserialized.length == array_info.length
        assert deserialized.dtype == array_info.dtype
        assert deserialized.shape == array_info.shape
        assert consumed == len(serialized)

    def test_invalid_dtype_deserialization(self):
        """Test deserialization with invalid dtype value."""
        from bionemo.scdl.schema.headerutil import BinaryHeaderCodec

        codec = BinaryHeaderCodec(Endianness.NETWORK)

        # Create data with invalid dtype (999)
        data = b""
        data += codec.pack_string("test.dat")
        data += codec.pack_uint64(1000)
        data += codec.pack_uint32(999)  # Invalid dtype
        data += codec.pack_uint8(0)  # no shape

        with pytest.raises(HeaderSerializationError, match="Invalid ArrayDType value"):
            ArrayInfo.deserialize(codec, data)

    def test_calculate_size(self):
        """Test size calculation."""
        array_info = ArrayInfo(name="test.dat", length=1000, dtype=ArrayDType.FLOAT32_ARRAY, shape=(100, 10))

        expected_size = (
            4  # name_len
            + len("test.dat".encode("utf-8"))  # name
            + 8  # length
            + 4  # dtype
            + 1  # has_shape
            + 4  # shape_dims
            + 4 * 2  # shape (2 dimensions)
        )

        assert array_info.calculate_size() == expected_size

    def test_unicode_name(self):
        """Test ArrayInfo with Unicode name."""
        from bionemo.scdl.schema.headerutil import BinaryHeaderCodec

        array_info = ArrayInfo(name="测试文件.dat", length=500, dtype=ArrayDType.UINT8_ARRAY)

        codec = BinaryHeaderCodec(Endianness.NETWORK)
        serialized = array_info.serialize(codec)
        deserialized, _ = ArrayInfo.deserialize(codec, serialized)

        assert deserialized.name == array_info.name


class TestFeatureIndexInfo:
    """Test FeatureIndexInfo class functionality."""

    def test_basic_creation(self):
        """Test basic FeatureIndexInfo creation."""
        feature_index = FeatureIndexInfo(name="gene_index", length=25000, dtype=ArrayDType.STRING_ARRAY)
        assert feature_index.name == "gene_index"
        assert feature_index.length == 25000
        assert feature_index.dtype == ArrayDType.STRING_ARRAY
        assert feature_index.index_files == []
        assert feature_index.shape is None

    def test_creation_with_files(self):
        """Test FeatureIndexInfo creation with index files."""
        files = ["index1.dat", "index2.dat"]
        feature_index = FeatureIndexInfo(
            name="complex_index", length=10000, dtype=ArrayDType.UINT32_ARRAY, index_files=files, shape=(100, 100)
        )
        assert feature_index.index_files == files
        assert feature_index.shape == (100, 100)

    def test_serialization_without_files(self):
        """Test serialization without index files."""
        from bionemo.scdl.schema.headerutil import BinaryHeaderCodec

        feature_index = FeatureIndexInfo(name="simple_index", length=1000, dtype=ArrayDType.UINT64_ARRAY)

        codec = BinaryHeaderCodec(Endianness.NETWORK)
        serialized = feature_index.serialize(codec)
        deserialized, consumed = FeatureIndexInfo.deserialize(codec, serialized)

        assert deserialized.name == feature_index.name
        assert deserialized.length == feature_index.length
        assert deserialized.dtype == feature_index.dtype
        assert deserialized.index_files == []
        assert deserialized.shape is None
        assert consumed == len(serialized)

    def test_serialization_with_files_and_shape(self):
        """Test serialization with index files and shape."""
        from bionemo.scdl.schema.headerutil import BinaryHeaderCodec

        files = ["file1.idx", "file2.idx", "file3.idx"]
        feature_index = FeatureIndexInfo(
            name="multi_file_index", length=5000, dtype=ArrayDType.FLOAT32_ARRAY, index_files=files, shape=(50, 100)
        )

        codec = BinaryHeaderCodec(Endianness.NETWORK)
        serialized = feature_index.serialize(codec)
        deserialized, consumed = FeatureIndexInfo.deserialize(codec, serialized)

        assert deserialized.name == feature_index.name
        assert deserialized.length == feature_index.length
        assert deserialized.dtype == feature_index.dtype
        assert deserialized.index_files == files
        assert deserialized.shape == feature_index.shape
        assert consumed == len(serialized)

    def test_validation_empty_file_path(self):
        """Test validation fails for empty file path."""
        feature_index = FeatureIndexInfo(
            name="test_index", length=100, dtype=ArrayDType.UINT8_ARRAY, index_files=["valid.dat", "", "another.dat"]
        )
        with pytest.raises(HeaderSerializationError, match="FeatureIndex file path 1 cannot be empty"):
            feature_index._validate()


class TestSCDLHeader:
    """Test SCDLHeader class functionality."""

    def test_basic_creation(self):
        """Test basic header creation."""
        header = SCDLHeader()
        assert header.version == EXPECTED_SCDL_VERSION
        assert header.endianness == Endianness.NETWORK
        assert header.backend == Backend.MEMMAP_V0
        assert len(header.arrays) == 0
        assert len(header.feature_indices) == 0

    def test_creation_with_custom_version(self):
        """Test header creation with custom version."""
        version = SCDLVersion()
        version.major = 1
        version.minor = 2
        version.point = 3

        header = SCDLHeader(version=version)
        assert header.version.major == 1
        assert header.version.minor == 2
        assert header.version.point == 3

    def test_add_get_remove_array(self):
        """Test array management methods."""
        header = SCDLHeader()

        array1 = ArrayInfo("test1.dat", 100, ArrayDType.UINT8_ARRAY)
        array2 = ArrayInfo("test2.dat", 200, ArrayDType.FLOAT32_ARRAY)

        # Test adding
        header.add_array(array1)
        header.add_array(array2)
        assert len(header.arrays) == 2

        # Test getting
        found = header.get_array("test1.dat")
        assert found is not None
        assert found.name == "test1.dat"

        not_found = header.get_array("nonexistent.dat")
        assert not_found is None

        # Test removing
        removed = header.remove_array("test1.dat")
        assert removed is True
        assert len(header.arrays) == 1

        not_removed = header.remove_array("nonexistent.dat")
        assert not_removed is False

    def test_add_get_remove_feature_index(self):
        """Test feature index management methods."""
        header = SCDLHeader()

        fi1 = FeatureIndexInfo("index1", 1000, ArrayDType.STRING_ARRAY)
        fi2 = FeatureIndexInfo("index2", 2000, ArrayDType.UINT32_ARRAY)

        # Test adding
        header.add_feature_index(fi1)
        header.add_feature_index(fi2)
        assert len(header.feature_indices) == 2

        # Test getting
        found = header.get_feature_index("index1")
        assert found is not None
        assert found.name == "index1"

        not_found = header.get_feature_index("nonexistent")
        assert not_found is None

        # Test removing
        removed = header.remove_feature_index("index1")
        assert removed is True
        assert len(header.feature_indices) == 1

        not_removed = header.remove_feature_index("nonexistent")
        assert not_removed is False

    def test_core_header_size(self):
        """Test that core header size constant matches schema."""
        # Schema specifies 16 bytes for core header
        assert SCDLHeader.CORE_HEADER_SIZE == 16

    def test_basic_serialization(self):
        """Test basic header serialization/deserialization."""
        header = SCDLHeader()

        # Add some content
        array = ArrayInfo("test.dat", 1000, ArrayDType.FLOAT32_ARRAY, (100, 10))
        header.add_array(array)

        fi = FeatureIndexInfo("genes", 25000, ArrayDType.STRING_ARRAY)
        header.add_feature_index(fi)

        # Serialize
        serialized = header.serialize()

        # Should start with magic number
        assert serialized[:4] == SCDL_MAGIC_NUMBER

        # Deserialize
        deserialized = SCDLHeader.deserialize(serialized)

        assert deserialized.version.major == header.version.major
        assert deserialized.version.minor == header.version.minor
        assert deserialized.version.point == header.version.point
        assert deserialized.backend == header.backend
        assert len(deserialized.arrays) == 1
        assert len(deserialized.feature_indices) == 1

        # Check array content
        deser_array = deserialized.arrays[0]
        assert deser_array.name == array.name
        assert deser_array.length == array.length
        assert deser_array.dtype == array.dtype
        assert deser_array.shape == array.shape

        # Check feature index content
        deser_fi = deserialized.feature_indices[0]
        assert deser_fi.name == fi.name
        assert deser_fi.length == fi.length
        assert deser_fi.dtype == fi.dtype

    def test_empty_header_serialization(self):
        """Test serialization of empty header."""
        header = SCDLHeader()
        serialized = header.serialize()

        # Should be exactly core header size + 4 bytes for feature index count
        expected_size = SCDLHeader.CORE_HEADER_SIZE + 4
        assert len(serialized) == expected_size

        deserialized = SCDLHeader.deserialize(serialized)
        assert len(deserialized.arrays) == 0
        assert len(deserialized.feature_indices) == 0

    def test_invalid_magic_number(self):
        """Test deserialization with invalid magic number."""
        # Create invalid data with wrong magic number
        invalid_data = b"FAKE" + b"\x00" * 20

        with pytest.raises(HeaderSerializationError, match="Invalid magic number"):
            SCDLHeader.deserialize(invalid_data)

    def test_insufficient_data(self):
        """Test deserialization with insufficient data."""
        # Data too short for core header
        short_data = b"SCDL\x00\x00"

        with pytest.raises(HeaderSerializationError, match="Header data too short"):
            SCDLHeader.deserialize(short_data)

    def test_invalid_endianness(self):
        """Test deserialization with invalid endianness."""
        from bionemo.scdl.schema.headerutil import BinaryHeaderCodec

        codec = BinaryHeaderCodec(Endianness.NETWORK)

        # Create header with invalid endianness
        data = SCDL_MAGIC_NUMBER
        data += codec.pack_uint8(0)  # version major
        data += codec.pack_uint8(0)  # version minor
        data += codec.pack_uint8(2)  # version point
        data += codec.pack_uint8(99)  # invalid endianness
        data += codec.pack_uint32(1)  # backend
        data += codec.pack_uint32(0)  # array count
        data += codec.pack_uint32(0)  # feature index count

        with pytest.raises(HeaderSerializationError, match="Invalid endianness"):
            SCDLHeader.deserialize(data)

    def test_invalid_backend(self):
        """Test deserialization with invalid backend."""
        from bionemo.scdl.schema.headerutil import BinaryHeaderCodec

        codec = BinaryHeaderCodec(Endianness.NETWORK)

        # Create header with invalid backend
        data = SCDL_MAGIC_NUMBER
        data += codec.pack_uint8(0)  # version major
        data += codec.pack_uint8(0)  # version minor
        data += codec.pack_uint8(2)  # version point
        data += codec.pack_uint8(1)  # endianness
        data += codec.pack_uint32(999)  # invalid backend
        data += codec.pack_uint32(0)  # array count
        data += codec.pack_uint32(0)  # feature index count

        with pytest.raises(HeaderSerializationError, match="Invalid backend value"):
            SCDLHeader.deserialize(data)

    def test_validation_duplicate_array_names(self):
        """Test validation fails for duplicate array names."""
        header = SCDLHeader()
        header.add_array(ArrayInfo("test.dat", 100, ArrayDType.UINT8_ARRAY))
        header.add_array(ArrayInfo("test.dat", 200, ArrayDType.FLOAT32_ARRAY))

        with pytest.raises(HeaderSerializationError, match="Duplicate array names found"):
            header.validate()

    def test_validation_duplicate_feature_index_names(self):
        """Test validation fails for duplicate feature index names."""
        header = SCDLHeader()
        header.add_feature_index(FeatureIndexInfo("index", 100, ArrayDType.UINT8_ARRAY))
        header.add_feature_index(FeatureIndexInfo("index", 200, ArrayDType.FLOAT32_ARRAY))

        with pytest.raises(HeaderSerializationError, match="Duplicate feature index names found"):
            header.validate()

    def test_validation_name_conflicts(self):
        """Test validation fails for name conflicts between arrays and feature indices."""
        header = SCDLHeader()
        header.add_array(ArrayInfo("conflict", 100, ArrayDType.UINT8_ARRAY))
        header.add_feature_index(FeatureIndexInfo("conflict", 200, ArrayDType.FLOAT32_ARRAY))

        with pytest.raises(HeaderSerializationError, match="Name conflicts between arrays and feature indices"):
            header.validate()

    def test_validation_future_version(self):
        """Test validation fails for unsupported future version."""
        version = SCDLVersion()
        version.major = 999
        version.minor = 0
        version.point = 0

        header = SCDLHeader(version=version)

        with pytest.raises(HeaderSerializationError, match="Unsupported version"):
            header.validate()

    def test_save_load_file(self):
        """Test saving and loading header from file."""
        header = SCDLHeader()
        header.add_array(ArrayInfo("test.dat", 1000, ArrayDType.FLOAT32_ARRAY))

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Save to file
            header.save(tmp_path)

            # Load from file
            loaded_header = SCDLHeader.load(tmp_path)

            assert loaded_header.version.major == header.version.major
            assert loaded_header.version.minor == header.version.minor
            assert loaded_header.version.point == header.version.point
            assert len(loaded_header.arrays) == 1
            assert loaded_header.arrays[0].name == "test.dat"

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file."""
        with pytest.raises(HeaderSerializationError, match="Header file not found"):
            SCDLHeader.load("/nonexistent/path/header.bin")

    def test_calculate_total_size(self):
        """Test total size calculation."""
        header = SCDLHeader()

        # Empty header should be core size + feature index count
        expected_empty = SCDLHeader.CORE_HEADER_SIZE + 4
        assert header.calculate_total_size() == expected_empty

        # Add array
        array = ArrayInfo("test.dat", 1000, ArrayDType.FLOAT32_ARRAY, (100, 10))
        header.add_array(array)

        expected_with_array = expected_empty + array.calculate_size()
        assert header.calculate_total_size() == expected_with_array

        # Add feature index
        fi = FeatureIndexInfo("index", 2000, ArrayDType.STRING_ARRAY, ["file1.idx"])
        header.add_feature_index(fi)

        expected_with_fi = expected_with_array + fi.calculate_size()
        assert header.calculate_total_size() == expected_with_fi

    def test_string_representations(self):
        """Test string representation methods."""
        header = SCDLHeader()
        header.add_array(ArrayInfo("test.dat", 1000, ArrayDType.FLOAT32_ARRAY))

        str_repr = str(header)
        assert "SCDLHeader" in str_repr
        assert "arrays=1" in str_repr
        assert "feature_indices=0" in str_repr

        repr_str = repr(header)
        assert repr_str == str_repr

    def test_json_output(self):
        """Test JSON representation."""
        header = SCDLHeader()
        array = ArrayInfo("test.dat", 1000, ArrayDType.FLOAT32_ARRAY, (100, 10))
        header.add_array(array)

        json_str = header.to_json()
        json_data = json.loads(json_str)

        assert json_data["version"]["major"] == EXPECTED_SCDL_VERSION.major
        assert json_data["version"]["minor"] == EXPECTED_SCDL_VERSION.minor
        assert json_data["version"]["point"] == EXPECTED_SCDL_VERSION.point
        assert json_data["backend"] == "MEMMAP_V0"
        assert len(json_data["arrays"]) == 1
        assert json_data["arrays"][0]["name"] == "test.dat"
        assert json_data["arrays"][0]["shape"] == [100, 10]


class TestSchemaCompliance:
    """Test compliance with SCDL schema specification."""

    def test_magic_number_specification(self):
        """Test magic number matches schema specification."""
        # Schema specifies 'SCDL' (0x5343444C)
        assert SCDL_MAGIC_NUMBER == b"SCDL"
        assert len(SCDL_MAGIC_NUMBER) == 4

    def test_current_version_matches_schema(self):
        """Test current version matches schema documentation."""
        # Schema documents version 0.1.0
        current = CurrentSCDLVersion()
        assert current == EXPECTED_SCDL_VERSION

    def test_endianness_specification(self):
        """Test endianness handling matches schema."""
        # Schema requires NETWORK byte order (value 1)
        header = SCDLHeader()
        assert header.endianness == Endianness.NETWORK

        # Serialize and check endianness byte
        serialized = header.serialize()
        endianness_byte = serialized[7]  # Offset 0x07 per schema
        assert endianness_byte == 1  # NETWORK = 1 per schema

    def test_core_header_layout(self):
        """Test core header layout matches schema specification."""
        header = SCDLHeader()
        serialized = header.serialize()

        # Schema specifies 16-byte core header
        assert len(serialized) >= 16

        # Magic number at offset 0x00 (4 bytes)
        assert serialized[0:4] == SCDL_MAGIC_NUMBER

        # Version at offsets 0x04, 0x05, 0x06 (3 bytes)
        assert serialized[4] == EXPECTED_SCDL_VERSION.major  # major
        assert serialized[5] == EXPECTED_SCDL_VERSION.minor  # minor
        assert serialized[6] == EXPECTED_SCDL_VERSION.point  # point

        # Endianness at offset 0x07 (1 byte)
        assert serialized[7] == 1  # NETWORK

        # Backend at offset 0x08 (4 bytes) - should be MEMMAP_V0 = 1
        from bionemo.scdl.schema.headerutil import BinaryHeaderCodec

        codec = BinaryHeaderCodec(Endianness.NETWORK)
        backend_value = codec.unpack_uint32(serialized[8:12])
        assert backend_value == 1  # MEMMAP_V0

        # Array count at offset 0x0C (4 bytes)
        array_count = codec.unpack_uint32(serialized[12:16])
        assert array_count == 0  # Empty header

    def test_array_descriptor_layout(self):
        """Test array descriptor layout matches schema."""
        from bionemo.scdl.schema.headerutil import BinaryHeaderCodec

        header = SCDLHeader()
        array = ArrayInfo("test.dat", 1000, ArrayDType.FLOAT32_ARRAY, (100, 10))
        header.add_array(array)

        serialized = header.serialize()
        codec = BinaryHeaderCodec(Endianness.NETWORK)

        # Skip core header (16 bytes)
        offset = 16

        # Array descriptor should start with name_len (4 bytes)
        name_len = codec.unpack_uint32(serialized[offset : offset + 4])
        assert name_len == len("test.dat".encode("utf-8"))
        offset += 4

        # Then name (UTF-8 encoded)
        name = serialized[offset : offset + name_len].decode("utf-8")
        assert name == "test.dat"
        offset += name_len

        # Then length (8 bytes)
        length = codec.unpack_uint64(serialized[offset : offset + 8])
        assert length == 1000
        offset += 8

        # Then dtype (4 bytes)
        dtype_value = codec.unpack_uint32(serialized[offset : offset + 4])
        assert dtype_value == int(ArrayDType.FLOAT32_ARRAY)
        offset += 4

        # Then has_shape (1 byte)
        has_shape = codec.unpack_uint8(serialized[offset : offset + 1])
        assert has_shape == 1  # True
        offset += 1

        # Then shape_dims (4 bytes)
        shape_dims = codec.unpack_uint32(serialized[offset : offset + 4])
        assert shape_dims == 2
        offset += 4

        # Then shape array (4 bytes * dimensions)
        shape = []
        for _ in range(shape_dims):
            dim = codec.unpack_uint32(serialized[offset : offset + 4])
            shape.append(dim)
            offset += 4
        assert shape == [100, 10]

    def test_feature_index_extension_layout(self):
        """Test feature index extension layout."""
        from bionemo.scdl.schema.headerutil import BinaryHeaderCodec

        header = SCDLHeader()
        fi = FeatureIndexInfo("genes", 25000, ArrayDType.STRING_ARRAY, ["index.dat"])
        header.add_feature_index(fi)

        serialized = header.serialize()
        codec = BinaryHeaderCodec(Endianness.NETWORK)

        # Skip core header (16 bytes) - no arrays
        offset = 16

        # Feature index count (4 bytes)
        fi_count = codec.unpack_uint32(serialized[offset : offset + 4])
        assert fi_count == 1
        offset += 4

        # Feature index descriptor should start with name_len
        name_len = codec.unpack_uint32(serialized[offset : offset + 4])
        assert name_len == len("genes".encode("utf-8"))


class TestUtilityFunctions:
    """Test utility functions."""

    def test_create_header_from_arrays(self):
        """Test header creation from array files."""
        files = ["array1.dat", "array2.dat", "array3.dat"]
        header = create_header_from_arrays(files)

        assert len(header.arrays) == 3
        assert header.backend == Backend.MEMMAP_V0

        # Check array names match filenames
        names = [array.name for array in header.arrays]
        expected_names = ["array1.dat", "array2.dat", "array3.dat"]
        assert names == expected_names

    def test_validate_header_compatibility_compatible(self):
        """Test validation of compatible headers."""
        header1 = SCDLHeader()
        header1.add_array(ArrayInfo("array1.dat", 100, ArrayDType.UINT8_ARRAY))

        header2 = SCDLHeader()
        header2.add_array(ArrayInfo("array2.dat", 200, ArrayDType.FLOAT32_ARRAY))

        assert validate_header_compatibility(header1, header2) is True

    def test_validate_header_compatibility_different_major_version(self):
        """Test validation fails for different major versions."""
        version1 = SCDLVersion()
        version1.major = 0
        version1.minor = 0
        version1.point = 2

        version2 = SCDLVersion()
        version2.major = 1
        version2.minor = 0
        version2.point = 0

        header1 = SCDLHeader(version=version1)
        header2 = SCDLHeader(version=version2)

        assert validate_header_compatibility(header1, header2) is False

    def test_validate_header_compatibility_different_backend(self):
        """Test validation fails for different backends."""
        header1 = SCDLHeader(backend=Backend.MEMMAP_V0)
        # Note: We only have one backend currently, so this test is theoretical
        # but demonstrates the validation logic
        header2 = SCDLHeader(backend=Backend.MEMMAP_V0)  # Same for now

        # Manually set different backend for testing
        header2.backend = 999  # Invalid backend

        assert validate_header_compatibility(header1, header2) is False

    def test_validate_header_compatibility_conflicting_array_names(self):
        """Test validation fails for conflicting array names."""
        header1 = SCDLHeader()
        header1.add_array(ArrayInfo("conflict.dat", 100, ArrayDType.UINT8_ARRAY))

        header2 = SCDLHeader()
        header2.add_array(ArrayInfo("conflict.dat", 200, ArrayDType.FLOAT32_ARRAY))

        assert validate_header_compatibility(header1, header2) is False

    def test_merge_headers_success(self):
        """Test successful header merging."""
        header1 = SCDLHeader()
        header1.add_array(ArrayInfo("array1.dat", 100, ArrayDType.UINT8_ARRAY))
        header1.add_feature_index(FeatureIndexInfo("index1", 1000, ArrayDType.STRING_ARRAY))

        header2 = SCDLHeader()
        header2.add_array(ArrayInfo("array2.dat", 200, ArrayDType.FLOAT32_ARRAY))
        header2.add_feature_index(FeatureIndexInfo("index2", 2000, ArrayDType.UINT32_ARRAY))

        merged = merge_headers(header1, header2)

        assert len(merged.arrays) == 2
        assert len(merged.feature_indices) == 2

        array_names = [array.name for array in merged.arrays]
        assert "array1.dat" in array_names
        assert "array2.dat" in array_names

        fi_names = [fi.name for fi in merged.feature_indices]
        assert "index1" in fi_names
        assert "index2" in fi_names

    def test_merge_headers_incompatible(self):
        """Test merging incompatible headers fails."""
        header1 = SCDLHeader()
        header1.add_array(ArrayInfo("conflict.dat", 100, ArrayDType.UINT8_ARRAY))

        header2 = SCDLHeader()
        header2.add_array(ArrayInfo("conflict.dat", 200, ArrayDType.FLOAT32_ARRAY))

        with pytest.raises(HeaderSerializationError, match="Headers are not compatible"):
            merge_headers(header1, header2)


class TestHeaderReader:
    """Test HeaderReader optimized reading functionality."""

    def test_header_reader_basic(self):
        """Test basic HeaderReader functionality."""
        header = SCDLHeader()
        header.add_array(ArrayInfo("test.dat", 1000, ArrayDType.FLOAT32_ARRAY))

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Save header
            header.save(tmp_path)

            # Create reader
            reader = HeaderReader(tmp_path)

            # Test magic validation
            assert reader.validate_magic() is True

            # Test version reading
            version = reader.get_version()
            assert version == EXPECTED_SCDL_VERSION

            # Test backend reading
            backend = reader.get_backend()
            assert backend == Backend.MEMMAP_V0

            # Test array count reading
            array_count = reader.get_array_count()
            assert array_count == 1

            # Test full header reading
            full_header = reader.get_full_header()
            assert len(full_header.arrays) == 1
            assert full_header.arrays[0].name == "test.dat"

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_header_reader_invalid_magic(self):
        """Test HeaderReader with invalid magic number."""
        # Create file with invalid magic
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"FAKE" + b"\x00" * 20)
            tmp_path = tmp.name

        try:
            reader = HeaderReader(tmp_path)
            assert reader.validate_magic() is False

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_header_reader_caching(self):
        """Test that HeaderReader caches results appropriately."""
        header = SCDLHeader()

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name

        try:
            header.save(tmp_path)
            reader = HeaderReader(tmp_path)

            # First call should read from file
            version1 = reader.get_version()
            # Second call should use cache
            version2 = reader.get_version()

            assert version1.major == version2.major
            assert version1.minor == version2.minor
            assert version1.point == version2.point

        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestBackwardsCompatibility:
    """Test backwards compatibility features."""

    def test_header_without_feature_indices(self):
        """Test reading headers without feature indices (backwards compatibility)."""
        from bionemo.scdl.schema.headerutil import BinaryHeaderCodec

        # Create header data without feature indices (older format)
        codec = BinaryHeaderCodec(Endianness.NETWORK)
        data = SCDL_MAGIC_NUMBER
        data += codec.pack_uint8(0)  # version major
        data += codec.pack_uint8(0)  # version minor
        data += codec.pack_uint8(1)  # version point (older)
        data += codec.pack_uint8(1)  # endianness
        data += codec.pack_uint32(1)  # backend
        data += codec.pack_uint32(0)  # array count
        # No feature index count - this simulates older format

        # Should deserialize successfully with empty feature indices
        header = SCDLHeader.deserialize(data)
        assert len(header.arrays) == 0
        assert len(header.feature_indices) == 0
        assert header.version.point == 1


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_maximum_size_limits(self):
        """Test behavior with large data structures."""
        header = SCDLHeader()

        # Test with very long array name
        long_name = "a" * 1000
        array = ArrayInfo(long_name, 1000000, ArrayDType.FLOAT64_ARRAY)
        header.add_array(array)

        # Should serialize and deserialize successfully
        serialized = header.serialize()
        deserialized = SCDLHeader.deserialize(serialized)
        assert deserialized.arrays[0].name == long_name

    def test_unicode_handling(self):
        """Test proper Unicode handling throughout."""
        header = SCDLHeader()

        # Array with Unicode name
        unicode_name = "数据文件.dat"
        array = ArrayInfo(unicode_name, 1000, ArrayDType.FLOAT32_ARRAY)
        header.add_array(array)

        # Feature index with Unicode name and files
        unicode_fi_name = "基因索引"
        unicode_files = ["文件1.idx", "文件2.idx"]
        fi = FeatureIndexInfo(unicode_fi_name, 5000, ArrayDType.STRING_ARRAY, unicode_files)
        header.add_feature_index(fi)

        # Should handle Unicode correctly
        serialized = header.serialize()
        deserialized = SCDLHeader.deserialize(serialized)

        assert deserialized.arrays[0].name == unicode_name
        assert deserialized.feature_indices[0].name == unicode_fi_name
        assert deserialized.feature_indices[0].index_files == unicode_files

    def test_zero_length_arrays(self):
        """Test handling of zero-length arrays."""
        header = SCDLHeader()
        array = ArrayInfo("empty.dat", 0, ArrayDType.UINT8_ARRAY)
        header.add_array(array)

        serialized = header.serialize()
        deserialized = SCDLHeader.deserialize(serialized)

        assert deserialized.arrays[0].length == 0

    def test_single_dimension_shape(self):
        """Test arrays with single-dimension shapes."""
        header = SCDLHeader()
        array = ArrayInfo("vector.dat", 1000, ArrayDType.FLOAT32_ARRAY, (1000,))
        header.add_array(array)

        serialized = header.serialize()
        deserialized = SCDLHeader.deserialize(serialized)

        assert deserialized.arrays[0].shape == (1000,)

    def test_high_dimensional_arrays(self):
        """Test arrays with many dimensions."""
        header = SCDLHeader()
        shape = (10, 10, 10, 10, 10)  # 5D array
        array = ArrayInfo("5d.dat", 100000, ArrayDType.FLOAT64_ARRAY, shape)
        header.add_array(array)

        serialized = header.serialize()
        deserialized = SCDLHeader.deserialize(serialized)

        assert deserialized.arrays[0].shape == shape
