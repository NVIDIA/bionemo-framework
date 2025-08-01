"""
SCDL Archive Header Implementation

This module provides comprehensive header serialization/deserialization for SCDL archives,
implementing the formal specification defined in scdl-schema.md.
"""

from enum import IntEnum
from typing import List, Tuple, Optional, BinaryIO
import json
import struct
from pathlib import Path

from .headerutil import BinaryHeaderCodec, Endianness, HeaderSerializationError
from .version import SCDLVersion, CurrentSCDLVersion, SCDLBackends
from .magic import magic_number


class ArrayDType(IntEnum):
    """
    Numpy dtype specification for arrays in SCDL archives.
    
    Integer values are used in the binary format for efficient storage.
    """
    UINT8_ARRAY = 1
    UINT16_ARRAY = 2
    UINT32_ARRAY = 3
    UINT64_ARRAY = 4
    FLOAT16_ARRAY = 5
    FLOAT32_ARRAY = 6
    FLOAT64_ARRAY = 7
    STRING_ARRAY = 8
    FIXED_STRING_ARRAY = 9

    @property
    def numpy_dtype_string(self) -> str:
        """Get the corresponding NumPy dtype string."""
        dtype_map = {
            self.UINT8_ARRAY: 'uint8',
            self.UINT16_ARRAY: 'uint16',
            self.UINT32_ARRAY: 'uint32',
            self.UINT64_ARRAY: 'uint64',
            self.FLOAT16_ARRAY: 'float16',
            self.FLOAT32_ARRAY: 'float32',
            self.FLOAT64_ARRAY: 'float64',
            self.STRING_ARRAY: 'string',
            self.FIXED_STRING_ARRAY: 'fixed_string'
        }
        return dtype_map[self]


class Backend(IntEnum):
    """
    Backend implementations for SCDL archives.
    
    Defines how array data is stored and accessed.
    """
    MEMMAP_V0 = 1
    MEMMAP_V1 = 2
    HDF5_V0 = 3
    ZARR_V0 = 4

class ArrayInfo:
    """
    Information about an array in the SCDL archive.
    
    Represents metadata for a single array as defined in the SCDL schema specification.
    """
    
    def __init__(self, 
                 name: str, 
                 length: int, 
                 dtype: ArrayDType, 
                 shape: Optional[Tuple[int, ...]] = None):
        """
        Initialize array information.
        
        Args:
            name: Filename of the array
            length: Number of elements in the array
            dtype: Data type of the array elements
            shape: Optional shape tuple for multidimensional arrays
        """
        self.name = name
        self.length = length
        self.dtype = dtype
        self.shape = shape
    
    def serialize(self, codec: BinaryHeaderCodec) -> bytes:
        """
        Serialize this ArrayInfo to binary format.
        
        Args:
            codec: Binary codec for serialization
            
        Returns:
            Binary representation following SCDL schema
        """
        data = b''
        
        # name_len + name
        data += codec.pack_string(self.name)
        
        # length (uint64)
        data += codec.pack_uint64(self.length)
        
        # dtype (uint32 enum value)
        data += codec.pack_uint32(int(self.dtype))
        
        # has_shape + optional shape data
        if self.shape is not None:
            data += codec.pack_uint8(1)  # has_shape = true
            data += codec.pack_uint32(len(self.shape))  # shape_dims
            for dim in self.shape:
                data += codec.pack_uint32(dim)  # shape array
        else:
            data += codec.pack_uint8(0)  # has_shape = false
        
        return data
    
    @classmethod
    def deserialize(cls, codec: BinaryHeaderCodec, data: bytes, offset: int = 0) -> Tuple['ArrayInfo', int]:
        """
        Deserialize ArrayInfo from binary data.
        
        Args:
            codec: Binary codec for deserialization
            data: Binary data containing serialized ArrayInfo
            offset: Starting offset in data
            
        Returns:
            Tuple of (ArrayInfo instance, bytes consumed)
            
        Raises:
            HeaderSerializationError: If data is invalid
        """
        current_offset = offset
        
        # Read name
        name, name_bytes = codec.unpack_string(data[current_offset:])
        current_offset += name_bytes
        
        # Read length
        length = codec.unpack_uint64(data[current_offset:current_offset + 8])
        current_offset += 8
        
        # Read dtype
        dtype_value = codec.unpack_uint32(data[current_offset:current_offset + 4])
        current_offset += 4
        
        try:
            dtype = ArrayDType(dtype_value)
        except ValueError:
            raise HeaderSerializationError(f"Invalid ArrayDType value: {dtype_value}")
        
        # Read optional shape
        has_shape = codec.unpack_uint8(data[current_offset:current_offset + 1])
        current_offset += 1
        
        shape = None
        if has_shape:
            shape_dims = codec.unpack_uint32(data[current_offset:current_offset + 4])
            current_offset += 4
            
            shape = []
            for _ in range(shape_dims):
                dim = codec.unpack_uint32(data[current_offset:current_offset + 4])
                shape.append(dim)
                current_offset += 4
            shape = tuple(shape)
        
        array_info = cls(name=name, length=length, dtype=dtype, shape=shape)
        bytes_consumed = current_offset - offset
        
        return array_info, bytes_consumed
    
    def calculate_size(self) -> int:
        """Calculate the serialized size of this ArrayInfo in bytes."""
        # name_len (4) + name length + length (8) + dtype (4) + has_shape (1)
        size = 4 + len(self.name.encode('utf-8')) + 8 + 4 + 1
        
        if self.shape is not None:
            # shape_dims (4) + shape array (4 * dimensions)
            size += 4 + (4 * len(self.shape))
        
        return size
    
    def __str__(self) -> str:
        shape_str = f", shape={self.shape}" if self.shape else ""
        return f"ArrayInfo(name='{self.name}', length={self.length}, dtype={self.dtype.name}{shape_str})"
    
    def __repr__(self) -> str:
        return self.__str__()


class FeatureIndexInfo:
    """
    Information about a feature index in the SCDL archive.
    
    Feature indices provide fast lookups for specific features in the data.
    """
    
    def __init__(self, 
                 name: str, 
                 length: int, 
                 dtype: ArrayDType, 
                 shape: Optional[Tuple[int, ...]] = None):
        """
        Initialize feature index information.
        
        Args:
            name: Name of the feature index
            length: Number of entries in the index
            dtype: Data type of index entries
            shape: Optional shape for multidimensional indices
        """
        self.name = name
        self.length = length
        self.dtype = dtype
        self.shape = shape
    
    def __str__(self) -> str:
        shape_str = f", shape={self.shape}" if self.shape else ""
        return f"FeatureIndexInfo(name='{self.name}', length={self.length}, dtype={self.dtype.name}{shape_str})"
    
    def __repr__(self) -> str:
        return self.__str__()

class SCDLHeader:
    """
    Header for a SCDL archive following the official schema specification.
    
    Contains metadata about the archive including version, backend, and array information.
    The header is stored in binary format and is not human-readable by design.
    """
    
    # Core header size is fixed at 16 bytes
    CORE_HEADER_SIZE = 16
    
    def __init__(self, 
                 version: Optional[SCDLVersion] = None,
                 backend: Backend = Backend.MEMMAP_V0,
                 arrays: Optional[List[ArrayInfo]] = None):
        """
        Initialize SCDL header.
        
        Args:
            version: SCDL schema version (defaults to current version)
            backend: Storage backend type
            arrays: List of arrays in the archive
        """
        self.version = version or CurrentSCDLVersion()
        self.endianness = Endianness.NETWORK  # Always network byte order per spec
        self.backend = backend
        self.arrays = arrays or []
        
        # Create codec with network byte order
        self._codec = BinaryHeaderCodec(self.endianness)
    
    def add_array(self, array_info: ArrayInfo) -> None:
        """Add an array to the header."""
        self.arrays.append(array_info)
    
    def get_array(self, name: str) -> Optional[ArrayInfo]:
        """Get array info by name."""
        for array in self.arrays:
            if array.name == name:
                return array
        return None
    
    def remove_array(self, name: str) -> bool:
        """Remove array by name. Returns True if found and removed."""
        for i, array in enumerate(self.arrays):
            if array.name == name:
                del self.arrays[i]
                return True
        return False
    
    def serialize(self) -> bytes:
        """
        Serialize the header to binary format following SCDL schema.
        
        Returns:
            Binary representation of the complete header
            
        Raises:
            HeaderSerializationError: If serialization fails
        """
        try:
            data = b''
            
            # Core Header (16 bytes fixed)
            # Magic number (4 bytes)
            data += magic_number
            
            # Version (3 bytes: major, minor, point)
            data += self._codec.pack_uint8(self.version.major)
            data += self._codec.pack_uint8(self.version.minor)
            data += self._codec.pack_uint8(self.version.point)
            
            # Endianness (1 byte) - always NETWORK per spec
            data += self._codec.pack_uint8(1)  # NETWORK = 1
            
            # Backend (4 bytes)
            data += self._codec.pack_uint32(int(self.backend))
            
            # Array count (4 bytes)
            data += self._codec.pack_uint32(len(self.arrays))
            
            # Array descriptors (variable size)
            for array in self.arrays:
                data += array.serialize(self._codec)
            
            return data
            
        except Exception as e:
            raise HeaderSerializationError(f"Failed to serialize SCDL header: {e}")
    
    @classmethod
    def deserialize(cls, data: bytes) -> 'SCDLHeader':
        """
        Deserialize header from binary data.
        
        Args:
            data: Binary data containing SCDL header
            
        Returns:
            SCDLHeader instance
            
        Raises:
            HeaderSerializationError: If deserialization fails or data is invalid
        """
        if len(data) < cls.CORE_HEADER_SIZE:
            raise HeaderSerializationError(
                f"Header data too short: {len(data)} bytes < {cls.CORE_HEADER_SIZE} bytes minimum"
            )
        
        # Use network byte order for reading
        codec = BinaryHeaderCodec(Endianness.NETWORK)
        offset = 0
        
        try:
            # Validate magic number
            magic = data[offset:offset + 4]
            if magic != magic_number:
                raise HeaderSerializationError(
                    f"Invalid magic number: {magic} != {magic_number}"
                )
            offset += 4
            
            # Read version
            version_major = codec.unpack_uint8(data[offset:offset + 1])
            offset += 1
            version_minor = codec.unpack_uint8(data[offset:offset + 1])
            offset += 1
            version_point = codec.unpack_uint8(data[offset:offset + 1])
            offset += 1
            
            version = SCDLVersion()
            version.major = version_major
            version.minor = version_minor
            version.point = version_point
            
            # Read and validate endianness
            endianness_value = codec.unpack_uint8(data[offset:offset + 1])
            offset += 1
            if endianness_value != 1:  # Must be NETWORK
                raise HeaderSerializationError(
                    f"Invalid endianness: {endianness_value} (must be 1 for NETWORK)"
                )
            
            # Read backend
            backend_value = codec.unpack_uint32(data[offset:offset + 4])
            offset += 4
            try:
                backend = Backend(backend_value)
            except ValueError:
                raise HeaderSerializationError(f"Invalid backend value: {backend_value}")
            
            # Read array count
            array_count = codec.unpack_uint32(data[offset:offset + 4])
            offset += 4
            
            # Read array descriptors
            arrays = []
            for i in range(array_count):
                if offset >= len(data):
                    raise HeaderSerializationError(
                        f"Unexpected end of data while reading array {i}"
                    )
                
                array_info, bytes_consumed = ArrayInfo.deserialize(codec, data, offset)
                arrays.append(array_info)
                offset += bytes_consumed
            
            header = cls(version=version, backend=backend, arrays=arrays)
            return header
            
        except HeaderSerializationError:
            raise
        except Exception as e:
            raise HeaderSerializationError(f"Failed to deserialize SCDL header: {e}")
    
    def save(self, file_path: str) -> None:
        """
        Save the header to a binary file.
        
        Args:
            file_path: Path to save the header file
            
        Raises:
            HeaderSerializationError: If saving fails
        """
        try:
            with open(file_path, 'wb') as f:
                f.write(self.serialize())
        except Exception as e:
            raise HeaderSerializationError(f"Failed to save header to {file_path}: {e}")
    
    @classmethod
    def load(cls, file_path: str) -> 'SCDLHeader':
        """
        Load header from a binary file.
        
        Args:
            file_path: Path to the header file
            
        Returns:
            SCDLHeader instance
            
        Raises:
            HeaderSerializationError: If loading fails
        """
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            return cls.deserialize(data)
        except FileNotFoundError:
            raise HeaderSerializationError(f"Header file not found: {file_path}")
        except Exception as e:
            raise HeaderSerializationError(f"Failed to load header from {file_path}: {e}")
    
    def calculate_total_size(self) -> int:
        """Calculate the total serialized size of the header in bytes."""
        total_size = self.CORE_HEADER_SIZE
        for array in self.arrays:
            total_size += array.calculate_size()
        return total_size
    
    def validate(self) -> None:
        """
        Validate the header for consistency and correctness.
        
        Raises:
            HeaderSerializationError: If validation fails
        """
        # Check version compatibility
        if self.version.major > CurrentSCDLVersion.major:
            raise HeaderSerializationError(
                f"Unsupported version: {self.version} > {CurrentSCDLVersion}"
            )
        
        # Check array names are unique
        names = [array.name for array in self.arrays]
        if len(names) != len(set(names)):
            raise HeaderSerializationError("Duplicate array names found")
        
        # Check array names are valid
        for array in self.arrays:
            if not array.name or not array.name.strip():
                raise HeaderSerializationError("Empty array name found")
            if len(array.name.encode('utf-8')) > 1024:  # Reasonable limit
                raise HeaderSerializationError(f"Array name too long: {array.name}")
    
    def __str__(self) -> str:
        """Return a human-readable string representation of the header."""
        return (
            f"SCDLHeader(version={self.version}, backend={self.backend.name}, "
            f"arrays={len(self.arrays)})"
        )
    
    def __repr__(self) -> str:
        return self.__str__()

    def to_json(self) -> str:
        """
        Return a JSON string representation of the header.
        
        Note: This is for debugging/inspection only, not for serialization.
        """
        def default(o):
            if hasattr(o, 'name'):
                return o.name
            if hasattr(o, '__dict__'):
                return o.__dict__
            return str(o)
        
        data = {
            'version': {
                'major': self.version.major,
                'minor': self.version.minor,
                'point': self.version.point
            },
            'endianness': self.endianness.name,
            'backend': self.backend.name,
            'arrays': [
                {
                    'name': array.name,
                    'length': array.length,
                    'dtype': array.dtype.name,
                    'shape': array.shape
                }
                for array in self.arrays
            ]
        }
        
        return json.dumps(data, indent=2, default=default)

    def to_yaml(self) -> str:
        """
        Return a YAML string representation of the header.
        
        Note: This is for debugging/inspection only, not for serialization.
        """
        try:
            import yaml
        except ImportError:
            raise RuntimeError("PyYAML is required for YAML serialization")
        
        data = {
            'version': f"{self.version.major}.{self.version.minor}.{self.version.point}",
            'endianness': self.endianness.name,
            'backend': self.backend.name,
            'arrays': [
                {
                    'name': array.name,
                    'length': array.length,
                    'dtype': array.dtype.name,
                    'shape': list(array.shape) if array.shape else None
                }
                for array in self.arrays
            ]
        }
        
        return yaml.dump(data, default_flow_style=False)


# Utility functions for header operations

def create_header_from_arrays(array_files: List[str], 
                              backend: Backend = Backend.MEMMAP_V0,
                              version: Optional[SCDLVersion] = None) -> SCDLHeader:
    """
    Create a SCDL header by scanning array files.
    
    Args:
        array_files: List of array file paths to include
        backend: Storage backend to use
        version: Schema version (defaults to current)
        
    Returns:
        SCDLHeader with arrays automatically detected
        
    Note:
        This function creates placeholder ArrayInfo objects. 
        Real implementations should inspect files to determine actual properties.
    """
    header = SCDLHeader(version=version, backend=backend)
    
    for file_path in array_files:
        path = Path(file_path)
        array_info = ArrayInfo(
            name=path.name,
            length=0,  # Would be determined by inspecting file
            dtype=ArrayDType.FLOAT32_ARRAY,  # Would be determined by inspecting file
            shape=None  # Would be determined by inspecting file
        )
        header.add_array(array_info)
    
    return header


def validate_header_compatibility(header1: SCDLHeader, header2: SCDLHeader) -> bool:
    """
    Check if two headers are compatible for operations like merging.
    
    Args:
        header1: First header
        header2: Second header
        
    Returns:
        True if headers are compatible
    """
    # Check version compatibility (same major version)
    if header1.version.major != header2.version.major:
        return False
    
    # Check backend compatibility
    if header1.backend != header2.backend:
        return False
    
    # Check for conflicting array names
    names1 = {array.name for array in header1.arrays}
    names2 = {array.name for array in header2.arrays}
    
    if names1.intersection(names2):
        return False
    
    return True


def merge_headers(header1: SCDLHeader, header2: SCDLHeader) -> SCDLHeader:
    """
    Merge two compatible headers into a single header.
    
    Args:
        header1: First header
        header2: Second header
        
    Returns:
        Merged header
        
    Raises:
        HeaderSerializationError: If headers are incompatible
    """
    if not validate_header_compatibility(header1, header2):
        raise HeaderSerializationError("Headers are not compatible for merging")
    
    # Use the newer version
    if header1.version.minor >= header2.version.minor:
        version = header1.version
    else:
        version = header2.version
    
    merged_header = SCDLHeader(
        version=version,
        backend=header1.backend,
        arrays=header1.arrays + header2.arrays
    )
    
    return merged_header


class HeaderReader:
    """
    Optimized reader for SCDL headers with caching and validation.
    
    Provides efficient access to header information without full deserialization
    when only specific fields are needed.
    """
    
    def __init__(self, file_path: str):
        """Initialize with header file path."""
        self.file_path = file_path
        self._cached_header = None
        self._core_header_cached = False
        self._magic = None
        self._version = None
        self._backend = None
        self._array_count = None
    
    def validate_magic(self) -> bool:
        """Quickly validate magic number without full deserialization."""
        if self._magic is None:
            with open(self.file_path, 'rb') as f:
                self._magic = f.read(4)
        return self._magic == magic_number
    
    def get_version(self) -> SCDLVersion:
        """Get version information quickly."""
        self._ensure_core_header()
        return self._version
    
    def get_backend(self) -> Backend:
        """Get backend information quickly."""
        self._ensure_core_header()
        return self._backend
    
    def get_array_count(self) -> int:
        """Get array count quickly."""
        self._ensure_core_header()
        return self._array_count
    
    def get_full_header(self) -> SCDLHeader:
        """Get complete header (cached after first access)."""
        if self._cached_header is None:
            self._cached_header = SCDLHeader.load(self.file_path)
        return self._cached_header
    
    def _ensure_core_header(self):
        """Read core header fields if not cached."""
        if self._core_header_cached:
            return
        
        codec = BinaryHeaderCodec(Endianness.NETWORK)
        with open(self.file_path, 'rb') as f:
            core_data = f.read(SCDLHeader.CORE_HEADER_SIZE)
        
        if len(core_data) < SCDLHeader.CORE_HEADER_SIZE:
            raise HeaderSerializationError("Invalid header file")
        
        offset = 0
        
        # Magic number
        self._magic = core_data[offset:offset + 4]
        offset += 4
        
        # Version
        version = SCDLVersion()
        version.major = codec.unpack_uint8(core_data[offset:offset + 1])
        offset += 1
        version.minor = codec.unpack_uint8(core_data[offset:offset + 1])
        offset += 1
        version.point = codec.unpack_uint8(core_data[offset:offset + 1])
        offset += 1
        self._version = version
        
        # Skip endianness
        offset += 1
        
        # Backend
        backend_value = codec.unpack_uint32(core_data[offset:offset + 4])
        self._backend = Backend(backend_value)
        offset += 4
        
        # Array count
        self._array_count = codec.unpack_uint32(core_data[offset:offset + 4])
        
        self._core_header_cached = True
