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
Comprehensive tests for the headerutil module.

Tests all functionality of BinaryHeaderCodec including integer packing/unpacking,
floating point operations, string handling, error conditions, and utility methods.
"""

import struct
import pytest
from typing import List, Tuple

from bionemo.scdl.schema.headerutil import (
    BinaryHeaderCodec,
    Endianness,
    HeaderSerializationError,
)


class TestEndianness:
    """Test the Endianness enum."""
    
    def test_endianness_values(self):
        """Test that endianness enum has expected values."""
        assert Endianness.NETWORK.value == '!'


class TestBinaryHeaderCodecInitialization:
    """Test BinaryHeaderCodec initialization."""
    
    def test_default_initialization(self):
        """Test default initialization uses NETWORK endianness."""
        codec = BinaryHeaderCodec()
        assert codec.endianness == '!'
    
    def test_network_endianness(self):
        """Test explicit network endianness."""
        codec = BinaryHeaderCodec(Endianness.NETWORK)
        assert codec.endianness == '!'


class TestIntegerPacking:
    """Test integer packing and unpacking methods."""
    
    @pytest.fixture
    def codec(self):
        """Create a codec for testing."""
        return BinaryHeaderCodec(Endianness.NETWORK)
    
    def test_uint8_pack_unpack(self, codec):
        """Test uint8 packing and unpacking."""
        # Test valid values
        test_values = [0, 1, 127, 128, 255]
        for value in test_values:
            packed = codec.pack_uint8(value)
            assert len(packed) == 1
            unpacked = codec.unpack_uint8(packed)
            assert unpacked == value
    
    def test_uint8_out_of_range(self, codec):
        """Test uint8 with out of range values."""
        with pytest.raises(HeaderSerializationError, match="uint8 value -1 out of range"):
            codec.pack_uint8(-1)
        
        with pytest.raises(HeaderSerializationError, match="uint8 value 256 out of range"):
            codec.pack_uint8(256)
    
    def test_uint8_invalid_type(self, codec):
        """Test uint8 with invalid type."""
        with pytest.raises(HeaderSerializationError, match="Expected integer for uint8"):
            codec.pack_uint8("not an int")
    
    def test_uint8_insufficient_data(self, codec):
        """Test uint8 unpacking with insufficient data."""
        with pytest.raises(HeaderSerializationError, match="Insufficient data for uint8"):
            codec.unpack_uint8(b'')
    
    def test_uint16_pack_unpack(self, codec):
        """Test uint16 packing and unpacking."""
        test_values = [0, 1, 32767, 32768, 65535]
        for value in test_values:
            packed = codec.pack_uint16(value)
            assert len(packed) == 2
            unpacked = codec.unpack_uint16(packed)
            assert unpacked == value
    
    def test_uint16_out_of_range(self, codec):
        """Test uint16 with out of range values."""
        with pytest.raises(HeaderSerializationError, match="uint16 value -1 out of range"):
            codec.pack_uint16(-1)
        
        with pytest.raises(HeaderSerializationError, match="uint16 value 65536 out of range"):
            codec.pack_uint16(65536)
    
    def test_uint16_insufficient_data(self, codec):
        """Test uint16 unpacking with insufficient data."""
        with pytest.raises(HeaderSerializationError, match="Insufficient data for uint16"):
            codec.unpack_uint16(b'\x00')
    
    def test_uint32_pack_unpack(self, codec):
        """Test uint32 packing and unpacking."""
        test_values = [0, 1, 2147483647, 2147483648, 4294967295]
        for value in test_values:
            packed = codec.pack_uint32(value)
            assert len(packed) == 4
            unpacked = codec.unpack_uint32(packed)
            assert unpacked == value
    
    def test_uint32_out_of_range(self, codec):
        """Test uint32 with out of range values."""
        with pytest.raises(HeaderSerializationError, match="uint32 value -1 out of range"):
            codec.pack_uint32(-1)
        
        with pytest.raises(HeaderSerializationError, match="uint32 value 4294967296 out of range"):
            codec.pack_uint32(4294967296)
    
    def test_uint32_insufficient_data(self, codec):
        """Test uint32 unpacking with insufficient data."""
        with pytest.raises(HeaderSerializationError, match="Insufficient data for uint32"):
            codec.unpack_uint32(b'\x00\x00\x00')
    
    def test_uint64_pack_unpack(self, codec):
        """Test uint64 packing and unpacking."""
        test_values = [0, 1, 9223372036854775807, 9223372036854775808, 18446744073709551615]
        for value in test_values:
            packed = codec.pack_uint64(value)
            assert len(packed) == 8
            unpacked = codec.unpack_uint64(packed)
            assert unpacked == value
    
    def test_uint64_out_of_range(self, codec):
        """Test uint64 with out of range values."""
        with pytest.raises(HeaderSerializationError, match="uint64 value -1 out of range"):
            codec.pack_uint64(-1)
        
        with pytest.raises(HeaderSerializationError, match="uint64 value 18446744073709551616 out of range"):
            codec.pack_uint64(18446744073709551616)
    
    def test_uint64_insufficient_data(self, codec):
        """Test uint64 unpacking with insufficient data."""
        with pytest.raises(HeaderSerializationError, match="Insufficient data for uint64"):
            codec.unpack_uint64(b'\x00\x00\x00\x00\x00\x00\x00')


class TestFloatingPointPacking:
    """Test floating point packing and unpacking methods."""
    
    @pytest.fixture
    def codec(self):
        """Create a codec for testing."""
        return BinaryHeaderCodec(Endianness.NETWORK)
    
    def test_float16_pack_unpack(self, codec):
        """Test float16 packing and unpacking."""
        test_values = [0.0, 1.0, -1.0, 3.14159, -2.5]
        for value in test_values:
            packed = codec.pack_float16(value)
            assert len(packed) == 2
            unpacked = codec.unpack_float16(packed)
            # Float16 has limited precision, so we check approximate equality
            assert abs(unpacked - value) < 0.01 or (value == 0.0 and unpacked == 0.0)
    
    def test_float16_insufficient_data(self, codec):
        """Test float16 unpacking with insufficient data."""
        with pytest.raises(HeaderSerializationError, match="Insufficient data for float16"):
            codec.unpack_float16(b'\x00')
    
    def test_float32_pack_unpack(self, codec):
        """Test float32 packing and unpacking."""
        test_values = [0.0, 1.0, -1.0, 3.14159265, -2.5, 1e10, -1e-10]
        for value in test_values:
            packed = codec.pack_float32(value)
            assert len(packed) == 4
            unpacked = codec.unpack_float32(packed)
            # Check approximate equality for floating point
            if value == 0.0:
                assert unpacked == 0.0
            else:
                assert abs((unpacked - value) / value) < 1e-6
    
    def test_float32_insufficient_data(self, codec):
        """Test float32 unpacking with insufficient data."""
        with pytest.raises(HeaderSerializationError, match="Insufficient data for float32"):
            codec.unpack_float32(b'\x00\x00\x00')
    
    def test_float_overflow_conditions(self, codec):
        """Test floating point overflow conditions."""
        # Large values should raise HeaderSerializationError
        large_value = 1e50
        with pytest.raises(HeaderSerializationError, match="Cannot pack float32 value"):
            codec.pack_float32(large_value)
        
        # Test with a value that can be represented as infinity
        import math
        packed_inf = codec.pack_float32(float('inf'))
        unpacked_inf = codec.unpack_float32(packed_inf)
        assert math.isinf(unpacked_inf) and unpacked_inf > 0
        
        packed_neg_inf = codec.pack_float32(float('-inf'))
        unpacked_neg_inf = codec.unpack_float32(packed_neg_inf)
        assert math.isinf(unpacked_neg_inf) and unpacked_neg_inf < 0


class TestStringPacking:
    """Test string packing and unpacking methods."""
    
    @pytest.fixture
    def codec(self):
        """Create a codec for testing."""
        return BinaryHeaderCodec(Endianness.NETWORK)
    
    def test_pack_unpack_string(self, codec):
        """Test basic string packing and unpacking."""
        test_strings = ["", "hello", "world", "Hello, 世界!", "🚀🌟✨"]
        
        for test_string in test_strings:
            packed = codec.pack_string(test_string)
            # Should have length prefix (4 bytes) + UTF-8 encoded string
            expected_length = 4 + len(test_string.encode('utf-8'))
            assert len(packed) == expected_length
            
            unpacked, consumed = codec.unpack_string(packed)
            assert unpacked == test_string
            assert consumed == len(packed)
    
    def test_pack_string_with_max_length(self, codec):
        """Test string packing with maximum length limit."""
        test_string = "hello world"
        
        # Should work within limit
        packed = codec.pack_string(test_string, max_length=20)
        unpacked, _ = codec.unpack_string(packed, max_length=20)
        assert unpacked == test_string
        
        # Should fail when exceeding limit
        with pytest.raises(HeaderSerializationError, match="String too long"):
            codec.pack_string(test_string, max_length=5)
    
    def test_unpack_string_with_max_length(self, codec):
        """Test string unpacking with maximum length limit."""
        test_string = "hello world"
        packed = codec.pack_string(test_string)
        
        # Should fail when exceeding unpack limit
        with pytest.raises(HeaderSerializationError, match="String too long"):
            codec.unpack_string(packed, max_length=5)
    
    def test_pack_string_invalid_type(self, codec):
        """Test string packing with invalid type."""
        with pytest.raises(HeaderSerializationError, match="Expected string"):
            codec.pack_string(123)
    
    def test_unpack_string_insufficient_data(self, codec):
        """Test string unpacking with insufficient data."""
        # Not enough data for length prefix
        with pytest.raises(HeaderSerializationError, match="Insufficient data for string length"):
            codec.unpack_string(b'\x00\x00')
        
        # Length prefix indicates more data than available
        invalid_data = codec.pack_uint32(10) + b'short'
        with pytest.raises(HeaderSerializationError, match="Insufficient data for string"):
            codec.unpack_string(invalid_data)
    
    def test_unpack_string_invalid_utf8(self, codec):
        """Test string unpacking with invalid UTF-8."""
        # Create data with valid length but invalid UTF-8 bytes
        length_prefix = codec.pack_uint32(2)
        invalid_utf8 = b'\xff\xfe'  # Invalid UTF-8 sequence
        invalid_data = length_prefix + invalid_utf8
        
        with pytest.raises(HeaderSerializationError, match="Cannot decode UTF-8 string"):
            codec.unpack_string(invalid_data)
    
    def test_pack_fixed_string(self, codec):
        """Test fixed-size string packing."""
        test_cases = [
            ("hello", 10, b'\x00'),
            ("world", 8, b'\x20'),  # Space padding
            ("exact", 5, b'\x00'),   # Exact fit
        ]
        
        for string_val, size, padding in test_cases:
            packed = codec.pack_fixed_string(string_val, size, padding)
            assert len(packed) == size
            
            # Verify content
            expected = string_val.encode('utf-8') + padding * (size - len(string_val.encode('utf-8')))
            assert packed == expected
    
    def test_unpack_fixed_string(self, codec):
        """Test fixed-size string unpacking."""
        test_cases = [
            ("hello", 10, b'\x00'),
            ("world", 8, b'\x20'),
            ("exact", 5, b'\x00'),
        ]
        
        for original_string, size, padding in test_cases:
            packed = codec.pack_fixed_string(original_string, size, padding)
            unpacked = codec.unpack_fixed_string(packed, size, padding)
            assert unpacked == original_string
    
    def test_pack_fixed_string_too_long(self, codec):
        """Test fixed string packing when string is too long."""
        with pytest.raises(HeaderSerializationError, match="String too long"):
            codec.pack_fixed_string("this is too long", 5)
    
    def test_pack_fixed_string_invalid_size(self, codec):
        """Test fixed string packing with invalid size."""
        with pytest.raises(HeaderSerializationError, match="Size must be positive"):
            codec.pack_fixed_string("test", 0)
        
        with pytest.raises(HeaderSerializationError, match="Size must be positive"):
            codec.pack_fixed_string("test", -1)
    
    def test_fixed_string_invalid_padding(self, codec):
        """Test fixed string operations with invalid padding."""
        with pytest.raises(HeaderSerializationError, match="Padding must be single byte"):
            codec.pack_fixed_string("test", 10, b'\x00\x00')
        
        with pytest.raises(HeaderSerializationError, match="Padding must be single byte"):
            codec.unpack_fixed_string(b'test\x00\x00\x00\x00\x00\x00', 10, b'\x00\x00')
    
    def test_unpack_fixed_string_insufficient_data(self, codec):
        """Test fixed string unpacking with insufficient data."""
        with pytest.raises(HeaderSerializationError, match="Insufficient data"):
            codec.unpack_fixed_string(b'short', 10)
    
    def test_fixed_string_unicode(self, codec):
        """Test fixed string with Unicode characters."""
        unicode_string = "Hello, 世界!"
        size = 20
        
        packed = codec.pack_fixed_string(unicode_string, size)
        assert len(packed) == size
        
        unpacked = codec.unpack_fixed_string(packed, size)
        assert unpacked == unicode_string


class TestValidationMethods:
    """Test internal validation methods."""
    
    @pytest.fixture
    def codec(self):
        """Create a codec for testing."""
        return BinaryHeaderCodec(Endianness.NETWORK)
    
    def test_validate_data_length_invalid_type(self, codec):
        """Test data length validation with invalid data type."""
        with pytest.raises(HeaderSerializationError, match="Expected bytes"):
            codec._validate_data_length("not bytes", 4, "test")
    
    def test_validate_uint_range_invalid_type(self, codec):
        """Test uint range validation with invalid type."""
        with pytest.raises(HeaderSerializationError, match="Expected integer"):
            codec._validate_uint_range("not int", 0, 255, "test")


class TestUtilityMethods:
    """Test utility methods."""
    
    @pytest.fixture
    def codec(self):
        """Create a codec for testing."""
        return BinaryHeaderCodec(Endianness.NETWORK)
    
    def test_calculate_header_size(self, codec):
        """Test header size calculation."""
        field_specs = [
            ('uint8', None),
            ('uint16', None),
            ('uint32', None),
            ('uint64', None),
            ('float16', None),
            ('float32', None),
            ('fixed_string', 32),
        ]
        
        expected_size = 1 + 2 + 4 + 8 + 2 + 4 + 32  # 53 bytes
        actual_size = codec.calculate_header_size(field_specs)
        assert actual_size == expected_size
    
    def test_calculate_header_size_invalid_field_type(self, codec):
        """Test header size calculation with invalid field type."""
        field_specs = [('invalid_type', None)]
        
        with pytest.raises(HeaderSerializationError, match="Unknown field type"):
            codec.calculate_header_size(field_specs)
    
    def test_calculate_header_size_invalid_fixed_string_size(self, codec):
        """Test header size calculation with invalid fixed string size."""
        # Non-integer size
        with pytest.raises(HeaderSerializationError, match="fixed_string requires positive integer size"):
            codec.calculate_header_size([('fixed_string', 'not_int')])
        
        # Zero size
        with pytest.raises(HeaderSerializationError, match="fixed_string requires positive integer size"):
            codec.calculate_header_size([('fixed_string', 0)])
        
        # Negative size
        with pytest.raises(HeaderSerializationError, match="fixed_string requires positive integer size"):
            codec.calculate_header_size([('fixed_string', -1)])


class TestEndToEndScenarios:
    """Test complete end-to-end scenarios."""
    
    @pytest.fixture
    def codec(self):
        """Create a codec for testing."""
        return BinaryHeaderCodec(Endianness.NETWORK)
    
    def test_complete_header_example(self, codec):
        """Test a complete header creation and parsing scenario."""
        # Create a file header similar to the example in the module
        magic_number = 0x12345678
        version = 1
        flags = 0x0001
        data_offset = 128
        filename = "myfile.dat"
        description = "Test file"
        
        # Pack header fields
        header = b''
        header += codec.pack_uint32(magic_number)
        header += codec.pack_uint16(version)
        header += codec.pack_uint16(flags)
        header += codec.pack_uint64(data_offset)
        header += codec.pack_fixed_string(filename, 64)
        header += codec.pack_string(description)
        
        # Verify total size is as expected
        expected_size = 4 + 2 + 2 + 8 + 64 + 4 + len(description.encode('utf-8'))
        assert len(header) == expected_size
        
        # Unpack header
        offset = 0
        magic = codec.unpack_uint32(header[offset:offset+4])
        offset += 4
        ver = codec.unpack_uint16(header[offset:offset+2])
        offset += 2
        flgs = codec.unpack_uint16(header[offset:offset+2])
        offset += 2
        data_off = codec.unpack_uint64(header[offset:offset+8])
        offset += 8
        fname = codec.unpack_fixed_string(header[offset:offset+64], 64)
        offset += 64
        desc, consumed = codec.unpack_string(header[offset:])
        
        # Verify all values match
        assert magic == magic_number
        assert ver == version
        assert flgs == flags
        assert data_off == data_offset
        assert fname == filename
        assert desc == description
    
    def test_mixed_data_types(self, codec):
        """Test packing and unpacking mixed data types."""
        # Pack various data types together
        data = b''
        data += codec.pack_uint8(42)
        data += codec.pack_float32(3.14159)
        data += codec.pack_string("test")
        data += codec.pack_uint64(1234567890123456789)
        data += codec.pack_fixed_string("fixed", 10)
        
        # Unpack in the same order
        offset = 0
        
        val1 = codec.unpack_uint8(data[offset:offset+1])
        offset += 1
        assert val1 == 42
        
        val2 = codec.unpack_float32(data[offset:offset+4])
        offset += 4
        assert abs(val2 - 3.14159) < 1e-6
        
        val3, consumed = codec.unpack_string(data[offset:])
        offset += consumed
        assert val3 == "test"
        
        val4 = codec.unpack_uint64(data[offset:offset+8])
        offset += 8
        assert val4 == 1234567890123456789
        
        val5 = codec.unpack_fixed_string(data[offset:offset+10], 10)
        assert val5 == "fixed"


class TestErrorHandling:
    """Test comprehensive error handling."""
    
    @pytest.fixture
    def codec(self):
        """Create a codec for testing."""
        return BinaryHeaderCodec(Endianness.NETWORK)
    
    def test_header_serialization_error_inheritance(self):
        """Test that HeaderSerializationError is properly inherited."""
        error = HeaderSerializationError("test message")
        assert isinstance(error, Exception)
        assert str(error) == "test message"
    
    def test_all_pack_methods_type_validation(self, codec):
        """Test that all pack methods validate input types."""
        non_integer = "not an integer"
        non_float = "not a float"
        non_string = 123
        
        integer_methods = [
            codec.pack_uint8, codec.pack_uint16, 
            codec.pack_uint32, codec.pack_uint64
        ]
        
        for method in integer_methods:
            with pytest.raises(HeaderSerializationError):
                method(non_integer)
        
        # Float methods should accept integers and floats
        float_methods = [codec.pack_float16, codec.pack_float32]
        for method in float_methods:
            # These should work (int converted to float)
            method(42)
            method(42.0)
        
        string_methods = [
            lambda x: codec.pack_string(x),
            lambda x: codec.pack_fixed_string(x, 10)
        ]
        
        for method in string_methods:
            with pytest.raises(HeaderSerializationError):
                method(non_string)
    
    def test_all_unpack_methods_data_validation(self, codec):
        """Test that all unpack methods validate input data."""
        invalid_data_types = [None, "string", 123, []]
        
        unpack_methods = [
            (codec.unpack_uint8, 1),
            (codec.unpack_uint16, 2),
            (codec.unpack_uint32, 4),
            (codec.unpack_uint64, 8),
            (codec.unpack_float16, 2),
            (codec.unpack_float32, 4),
        ]
        
        for method, size in unpack_methods:
            for invalid_data in invalid_data_types:
                with pytest.raises(HeaderSerializationError):
                    method(invalid_data)