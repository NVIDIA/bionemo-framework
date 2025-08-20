# SCDL Header API Reference

Quick reference for the SCDL header API classes and functions.

## Core Classes

### `SCDLHeader`

Main header class for SCDL archives.

```python
class SCDLHeader:
    def __init__(self, version=None, backend=Backend.MEMMAP_V0,
                 arrays=None, feature_indices=None)

    # Array management
    def add_array(self, array_info: ArrayInfo) -> None
    def get_array(self, name: str) -> Optional[ArrayInfo]
    def remove_array(self, name: str) -> bool

    # Feature index management
    def add_feature_index(self, feature_index: FeatureIndexInfo) -> None
    def get_feature_index(self, name: str) -> Optional[FeatureIndexInfo]
    def remove_feature_index(self, name: str) -> bool

    # Serialization
    def serialize(self) -> bytes
    @classmethod
    def deserialize(cls, data: bytes) -> 'SCDLHeader'

    # File I/O
    def save(self, file_path: str) -> None
    @classmethod
    def load(cls, file_path: str) -> 'SCDLHeader'

    # Validation and utilities
    def validate(self) -> None
    def calculate_total_size(self) -> int
    def to_json(self) -> str
    def to_yaml(self) -> str
```

### `ArrayInfo`

Information about arrays in the archive.

```python
class ArrayInfo:
    def __init__(self, name: str, length: int, dtype: ArrayDType,
                 shape: Optional[Tuple[int, ...]] = None)

    # Properties
    name: str                           # Array filename
    length: int                        # Number of elements
    dtype: ArrayDType                  # Data type
    shape: Optional[Tuple[int, ...]]   # Optional shape

    # Serialization
    def serialize(self, codec: BinaryHeaderCodec) -> bytes
    @classmethod
    def deserialize(cls, codec: BinaryHeaderCodec, data: bytes,
                   offset: int = 0) -> Tuple['ArrayInfo', int]

    # Utilities
    def calculate_size(self) -> int
```

### `FeatureIndexInfo`

Information about feature indices in the archive.

```python
class FeatureIndexInfo:
    def __init__(self, name: str, length: int, dtype: ArrayDType,
                 index_files: Optional[List[str]] = None,
                 shape: Optional[Tuple[int, ...]] = None)

    # Properties
    name: str                           # Index name
    length: int                        # Number of entries
    dtype: ArrayDType                  # Data type
    index_files: List[str]             # Associated index files
    shape: Optional[Tuple[int, ...]]   # Optional shape

    # Serialization
    def serialize(self, codec: BinaryHeaderCodec) -> bytes
    @classmethod
    def deserialize(cls, codec: BinaryHeaderCodec, data: bytes,
                   offset: int = 0) -> Tuple['FeatureIndexInfo', int]

    # Utilities
    def calculate_size(self) -> int
```

## Enums

### `ArrayDType`

Data types for arrays.

```python
class ArrayDType(IntEnum):
    UINT8_ARRAY = 1          # 8-bit unsigned integers
    UINT16_ARRAY = 2         # 16-bit unsigned integers
    UINT32_ARRAY = 3         # 32-bit unsigned integers
    UINT64_ARRAY = 4         # 64-bit unsigned integers
    FLOAT16_ARRAY = 5        # 16-bit floating point
    FLOAT32_ARRAY = 6        # 32-bit floating point
    FLOAT64_ARRAY = 7        # 64-bit floating point
    STRING_ARRAY = 8         # Variable-length strings
    FIXED_STRING_ARRAY = 9   # Fixed-length strings

    @property
    def numpy_dtype_string(self) -> str  # Get NumPy dtype string

    @classmethod
    def from_numpy_dtype(cls, dtype) -> 'ArrayDType'  # Convert from NumPy dtype
```

### `Backend`

Storage backend types.

```python
class Backend(IntEnum):
    MEMMAP_V0 = 1  # Memory-mapped backend
```

## Utility Functions

### Header Operations

```python
def create_header_from_arrays(array_files: List[str],
                             backend: Backend = Backend.MEMMAP_V0,
                             version: Optional[SCDLVersion] = None) -> SCDLHeader
    """Create header by scanning array files."""

def validate_header_compatibility(header1: SCDLHeader,
                                header2: SCDLHeader) -> bool
    """Check if two headers are compatible for merging."""

def merge_headers(header1: SCDLHeader, header2: SCDLHeader) -> SCDLHeader
    """Merge two compatible headers."""
```

### Optimized Reading

```python
class HeaderReader:
    def __init__(self, file_path: str)

    def validate_magic(self) -> bool              # Quick magic number check
    def get_version(self) -> SCDLVersion         # Get version info
    def get_backend(self) -> Backend             # Get backend info
    def get_array_count(self) -> int             # Get array count
    def get_full_header(self) -> SCDLHeader      # Get complete header
```

## Version Classes

```python
class SCDLVersion:
    major: int = 0
    minor: int = 0
    point: int = 0

    def __str__(self) -> str         # "major.minor.point"
    def __eq__(self, other) -> bool
    def __ne__(self, other) -> bool

class CurrentSCDLVersion(SCDLVersion):
    major: int = 0
    minor: int = 0
    point: int = 2
```

## Constants

```python
from bionemo.scdl.schema.magic import SCDL_MAGIC_NUMBER
from bionemo.scdl.schema.headerutil import Endianness

SCDL_MAGIC_NUMBER: bytes = b"SCDL"  # Archive magic number
Endianness.NETWORK  # Network byte order (required)
```

## Exceptions

```python
class HeaderSerializationError(Exception):
    """Raised when header operations fail."""
```

## Common Patterns

### Basic Header Creation

```python
from bionemo.scdl.schema.header import SCDLHeader, ArrayInfo, ArrayDType

header = SCDLHeader()
array = ArrayInfo("data.dat", 1000, ArrayDType.FLOAT32_ARRAY, (100, 10))
header.add_array(array)
header.save("header.bin")
```

### Error Handling

```python
from bionemo.scdl.schema.headerutil import HeaderSerializationError

try:
    header = SCDLHeader.load("header.bin")
    header.validate()
except HeaderSerializationError as e:
    print(f"Header error: {e}")
```

### Inspection

```python
header = SCDLHeader.load("header.bin")

# Quick inspection
print(f"Arrays: {len(header.arrays)}")
print(f"Feature indices: {len(header.feature_indices)}")
print(f"Total size: {header.calculate_total_size()} bytes")

# Detailed inspection
for array in header.arrays:
    print(f"Array {array.name}: {array.length} elements, {array.dtype.name}")

for fi in header.feature_indices:
    print(f"Index {fi.name}: {fi.length} entries, {len(fi.index_files)} files")
```

### Working with Large Headers

```python
from bionemo.scdl.schema.header import HeaderReader

# Efficient reading for large headers
reader = HeaderReader("large_header.bin")
if reader.validate_magic():
    print(f"Version: {reader.get_version()}")
    print(f"Arrays: {reader.get_array_count()}")

    # Only load full header when needed
    if reader.get_array_count() > 0:
        full_header = reader.get_full_header()
```

### Converting NumPy Types

```python
import numpy as np
from bionemo.scdl.schema.header import ArrayDType

# Convert various numpy dtypes to ArrayDType enums
array_dtype1 = ArrayDType.from_numpy_dtype(np.float32)  # Type class
array_dtype2 = ArrayDType.from_numpy_dtype("float32")  # String
array_dtype3 = ArrayDType.from_numpy_dtype(np.dtype("f4"))  # Dtype object

# Use in ArrayInfo creation
array = ArrayInfo("data.dat", 1000, array_dtype1)
```
