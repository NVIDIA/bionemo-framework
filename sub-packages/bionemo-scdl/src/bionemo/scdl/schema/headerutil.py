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


"""Cross-platform binary header serialization utilities.

This module provides tools for creating fixed-size binary headers that maintain
metadata about files in a cross-platform, non-user-readable format.
"""

import struct
from enum import Enum
from typing import List, Tuple, Union


class Endianness(Enum):
    """Byte order specifications for binary data serialization."""

    NETWORK = (
        "!"  # Network byte order (same as big-endian). This is a good standard, used by Protobuf and other libraries.
    )
    # LITTLE = '<'  # Little-endian (most common on x86/x64)
    # BIG = '>'     # Big-endian (network byte order)
    # NATIVE = '='  # Native system byte order


class HeaderSerializationError(Exception):
    """Raised when header serialization/deserialization fails."""

    pass


class BinaryHeaderCodec:
    """A robust codec for serializing and deserializing fixed-size binary headers.

    This class provides a clean API for packing and unpacking various data types
    to/from binary format, with consistent endianness handling and comprehensive
    error checking. Designed for creating cross-platform file headers in binary form.

    Args:
        endianness: Byte order for serialization (default: NETWORK)

    Example:
        >>> codec = BinaryHeaderCodec(Endianness.NETWORK)
        >>> data = codec.pack_uint32(42)
        >>> value = codec.unpack_uint32(data)
        >>> assert value == 42
    """

    def __init__(self, endianness: Endianness = Endianness.NETWORK):
        """Initialize the codec with specified byte order."""
        self.endianness = endianness.value

    # Integer packing/unpacking methods

    def pack_uint8(self, value: int) -> bytes:
        """Pack an 8-bit unsigned integer.

        Args:
            value: Integer value (0-255)

        Returns:
            1-byte binary representation

        Raises:
            HeaderSerializationError: If value is out of range
        """
        self._validate_uint_range(value, 0, 255, "uint8")
        return struct.pack(f"{self.endianness}B", value)

    def unpack_uint8(self, data: bytes) -> int:
        """Unpack an 8-bit unsigned integer.

        Args:
            data: Binary data (must be at least 1 byte)

        Returns:
            Unpacked integer value

        Raises:
            HeaderSerializationError: If data is insufficient or invalid
        """
        self._validate_data_length(data, 1, "uint8")
        return struct.unpack(f"{self.endianness}B", data[:1])[0]

    def pack_uint16(self, value: int) -> bytes:
        """Pack a 16-bit unsigned integer.

        Args:
            value: Integer value (0-65535)

        Returns:
            2-byte binary representation

        Raises:
            HeaderSerializationError: If value is out of range
        """
        self._validate_uint_range(value, 0, 65535, "uint16")
        return struct.pack(f"{self.endianness}H", value)

    def unpack_uint16(self, data: bytes) -> int:
        """Unpack a 16-bit unsigned integer.

        Args:
            data: Binary data (must be at least 2 bytes)

        Returns:
            Unpacked integer value

        Raises:
            HeaderSerializationError: If data is insufficient or invalid
        """
        self._validate_data_length(data, 2, "uint16")
        return struct.unpack(f"{self.endianness}H", data[:2])[0]

    def pack_uint32(self, value: int) -> bytes:
        """Pack a 32-bit unsigned integer.

        Args:
            value: Integer value (0-4294967295)

        Returns:
            4-byte binary representation

        Raises:
            HeaderSerializationError: If value is out of range
        """
        self._validate_uint_range(value, 0, 4294967295, "uint32")
        return struct.pack(f"{self.endianness}I", value)

    def unpack_uint32(self, data: bytes) -> int:
        """Unpack a 32-bit unsigned integer.

        Args:
            data: Binary data (must be at least 4 bytes)

        Returns:
            Unpacked integer value

        Raises:
            HeaderSerializationError: If data is insufficient or invalid
        """
        self._validate_data_length(data, 4, "uint32")
        return struct.unpack(f"{self.endianness}I", data[:4])[0]

    def pack_uint64(self, value: int) -> bytes:
        """Pack a 64-bit unsigned integer.

        Args:
            value: Integer value (0-18446744073709551615)

        Returns:
            8-byte binary representation

        Raises:
            HeaderSerializationError: If value is out of range
        """
        self._validate_uint_range(value, 0, 18446744073709551615, "uint64")
        return struct.pack(f"{self.endianness}Q", value)

    def unpack_uint64(self, data: bytes) -> int:
        """Unpack a 64-bit unsigned integer.

        Args:
            data: Binary data (must be at least 8 bytes)

        Returns:
            Unpacked integer value

        Raises:
            HeaderSerializationError: If data is insufficient or invalid
        """
        self._validate_data_length(data, 8, "uint64")
        return struct.unpack(f"{self.endianness}Q", data[:8])[0]

    # Floating point packing/unpacking methods

    def pack_float16(self, value: float) -> bytes:
        """Pack a 16-bit (half-precision) floating point number.

        Args:
            value: Float value

        Returns:
            2-byte binary representation

        Raises:
            HeaderSerializationError: If value cannot be represented
        """
        try:
            return struct.pack(f"{self.endianness}e", value)
        except (struct.error, OverflowError) as e:
            raise HeaderSerializationError(f"Cannot pack float16 value {value}: {e}")

    def unpack_float16(self, data: bytes) -> float:
        """Unpack a 16-bit (half-precision) floating point number.

        Args:
            data: Binary data (must be at least 2 bytes)

        Returns:
            Unpacked float value

        Raises:
            HeaderSerializationError: If data is insufficient or invalid
        """
        self._validate_data_length(data, 2, "float16")
        return struct.unpack(f"{self.endianness}e", data[:2])[0]

    def pack_float32(self, value: float) -> bytes:
        """Pack a 32-bit (single-precision) floating point number.

        Args:
            value: Float value

        Returns:
            4-byte binary representation

        Raises:
            HeaderSerializationError: If value cannot be represented
        """
        try:
            return struct.pack(f"{self.endianness}f", value)
        except (struct.error, OverflowError) as e:
            raise HeaderSerializationError(f"Cannot pack float32 value {value}: {e}")

    def unpack_float32(self, data: bytes) -> float:
        """Unpack a 32-bit (single-precision) floating point number.

        Args:
            data: Binary data (must be at least 4 bytes)

        Returns:
            Unpacked float value

        Raises:
            HeaderSerializationError: If data is insufficient or invalid
        """
        self._validate_data_length(data, 4, "float32")
        return struct.unpack(f"{self.endianness}f", data[:4])[0]

    # String and array methods (for variable-length data)

    def pack_string(self, value: str, max_length: int | None = None) -> bytes:
        """Pack a UTF-8 string with length prefix.

        Args:
            value: String to pack
            max_length: Optional maximum length limit

        Returns:
            Binary data: 4-byte length + UTF-8 encoded string

        Raises:
            HeaderSerializationError: If string is too long or encoding fails
        """
        if not isinstance(value, str):
            raise HeaderSerializationError(f"Expected string, got {type(value)}")

        try:
            encoded_string = value.encode("utf-8")
        except UnicodeEncodeError as e:
            raise HeaderSerializationError(f"Cannot encode string to UTF-8: {e}")

        length = len(encoded_string)

        if max_length is not None and length > max_length:
            raise HeaderSerializationError(f"String too long: {length} bytes > {max_length} bytes limit")

        return self.pack_uint32(length) + encoded_string

    def unpack_string(self, data: bytes, max_length: int | None = None) -> Tuple[str, int]:
        """Unpack a UTF-8 string with length prefix.

        Args:
            data: Binary data starting with 4-byte length prefix
            max_length: Optional maximum length limit

        Returns:
            Tuple of (unpacked string, total bytes consumed)

        Raises:
            HeaderSerializationError: If data is invalid or string too long
        """
        if len(data) < 4:
            raise HeaderSerializationError("Insufficient data for string length")

        length = self.unpack_uint32(data[:4])

        if max_length is not None and length > max_length:
            raise HeaderSerializationError(f"String too long: {length} bytes > {max_length} bytes limit")

        if len(data) < 4 + length:
            raise HeaderSerializationError(f"Insufficient data for string: need {4 + length} bytes, got {len(data)}")

        try:
            string_value = data[4 : 4 + length].decode("utf-8")
        except UnicodeDecodeError as e:
            raise HeaderSerializationError(f"Cannot decode UTF-8 string: {e}")

        return string_value, 4 + length

    def pack_fixed_string(self, value: str, size: int, padding: bytes = b"\x00") -> bytes:
        """Pack a string into a fixed-size field with padding.

        Useful for creating truly fixed-size headers where string fields
        have a predetermined maximum size.

        Args:
            value: String to pack
            size: Fixed size of the field in bytes
            padding: Byte value to use for padding (default: null bytes)

        Returns:
            Fixed-size binary data

        Raises:
            HeaderSerializationError: If string is too long or parameters invalid
        """
        if not isinstance(value, str):
            raise HeaderSerializationError(f"Expected string, got {type(value)}")

        if size <= 0:
            raise HeaderSerializationError(f"Size must be positive, got {size}")

        if len(padding) != 1:
            raise HeaderSerializationError(f"Padding must be single byte, got {len(padding)} bytes")

        try:
            encoded = value.encode("utf-8")
        except UnicodeEncodeError as e:
            raise HeaderSerializationError(f"Cannot encode string to UTF-8: {e}")

        if len(encoded) > size:
            raise HeaderSerializationError(f"String too long: {len(encoded)} bytes > {size} bytes field size")

        return encoded + padding * (size - len(encoded))

    def unpack_fixed_string(self, data: bytes, size: int, padding: bytes = b"\x00") -> str:
        """Unpack a string from a fixed-size field, removing padding.

        Args:
            data: Binary data (must be at least size bytes)
            size: Size of the fixed field in bytes
            padding: Padding byte to strip (default: null bytes)

        Returns:
            Unpacked string with padding removed

        Raises:
            HeaderSerializationError: If data is insufficient or invalid
        """
        if len(data) < size:
            raise HeaderSerializationError(f"Insufficient data: need {size} bytes, got {len(data)}")

        if len(padding) != 1:
            raise HeaderSerializationError(f"Padding must be single byte, got {len(padding)} bytes")

        field_data = data[:size]
        # Remove trailing padding
        string_data = field_data.rstrip(padding)

        try:
            return string_data.decode("utf-8")
        except UnicodeDecodeError as e:
            raise HeaderSerializationError(f"Cannot decode UTF-8 string: {e}")

    # Validation helper methods

    def _validate_uint_range(self, value: int, min_val: int, max_val: int, type_name: str) -> None:
        """Validate that an integer value is within the valid range for its type."""
        if not isinstance(value, int):
            raise HeaderSerializationError(f"Expected integer for {type_name}, got {type(value)}")

        if value < min_val or value > max_val:
            raise HeaderSerializationError(f"{type_name} value {value} out of range [{min_val}, {max_val}]")

    def _validate_data_length(self, data: bytes, required_length: int, type_name: str) -> None:
        """Validate that data has sufficient length for unpacking."""
        if not isinstance(data, (bytes, bytearray)):
            raise HeaderSerializationError(f"Expected bytes for {type_name}, got {type(data)}")

        if len(data) < required_length:
            raise HeaderSerializationError(
                f"Insufficient data for {type_name}: need {required_length} bytes, got {len(data)}"
            )

    # Utility methods for working with headers

    def calculate_header_size(self, field_specs: List[Tuple[str, Union[int, str]]]) -> int:
        """Calculate the total size of a header given field specifications.

        Args:
            field_specs: List of (field_type, size) tuples where:
                - field_type: 'uint8', 'uint16', 'uint32', 'uint64', 'float16', 'float32', 'fixed_string'
                - size: For fixed_string, the size in bytes; ignored for other types

        Returns:
            Total header size in bytes

        Example:
            >>> codec = BinaryHeaderCodec()
            >>> size = codec.calculate_header_size([
            ...     ('uint32', None),      # 4 bytes
            ...     ('uint16', None),      # 2 bytes
            ...     ('fixed_string', 64),  # 64 bytes
            ...     ('float32', None)      # 4 bytes
            ... ])
            >>> assert size == 74
        """
        size_map = {"uint8": 1, "uint16": 2, "uint32": 4, "uint64": 8, "float16": 2, "float32": 4}

        total_size = 0
        for field_type, field_size in field_specs:
            if field_type == "fixed_string":
                if not isinstance(field_size, int) or field_size <= 0:
                    raise HeaderSerializationError(f"fixed_string requires positive integer size, got {field_size}")
                total_size += field_size
            elif field_type in size_map:
                total_size += size_map[field_type]
            else:
                raise HeaderSerializationError(f"Unknown field type: {field_type}")

        return total_size


# Example usage (commented out - focus on core functionality)
"""
Example of how to use BinaryHeaderCodec for creating file headers:

if __name__ == '__main__':
    # Create a codec with network-endian byte order
    codec = BinaryHeaderCodec(Endianness.NETWORK)

    # Example: Create a simple file header
    magic_number = 0x12345678
    version = 1
    flags = 0x0001
    data_offset = 128
    filename = "myfile.dat"

    # Pack header fields
    header = b''
    header += codec.pack_uint32(magic_number)  # Magic number (4 bytes)
    header += codec.pack_uint16(version)       # Version (2 bytes)
    header += codec.pack_uint16(flags)         # Flags (2 bytes)
    header += codec.pack_uint64(data_offset)   # Data offset (8 bytes)
    header += codec.pack_fixed_string(filename, 64)  # Filename (64 bytes fixed)

    # Total header size: 4 + 2 + 2 + 8 + 64 = 80 bytes

    # Write header to file
    with open('example.bin', 'wb') as f:
        f.write(header)

    # Read and unpack header
    with open('example.bin', 'rb') as f:
        data = f.read()

    offset = 0
    magic = codec.unpack_uint32(data[offset:offset+4])
    offset += 4
    ver = codec.unpack_uint16(data[offset:offset+2])
    offset += 2
    flgs = codec.unpack_uint16(data[offset:offset+2])
    offset += 2
    data_off = codec.unpack_uint64(data[offset:offset+8])
    offset += 8
    fname = codec.unpack_fixed_string(data[offset:offset+64], 64)

    print(f"Magic: 0x{magic:08x}, Version: {ver}, Flags: 0x{flgs:04x}")
    print(f"Data offset: {data_off}, Filename: '{fname}'")
"""
