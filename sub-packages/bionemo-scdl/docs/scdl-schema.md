# SCDL Schema

Eric T. Dawson
1 August 2025

## Version

0.0.9

**Implementation Status:** ✅ Fully implemented and validated against this specification

## Overview

The SCDL schema defines the structure of a SCDL archive. This enables backwards compatibility,
clear versions and updates, and robust, safe loading of SCDL archives to and from disk.

## SCDL Archive Structure (v0.0.9)

The SCDL archive is a directory containing a binary header file and a series of arrays.
The header contains metadata about the file, such as the version, the endianness, and the arrays that are contained in the file.
The arrays are stored in a contiguous block of memory and are *not* user-readable by design. Users should not
have access to modify the header, which should only be modified by the SCDL library.

### Archive Header

The header is a binary file that contains the metadata for the archive. It is stored in the root of the archive.

#### Header Fields

- Magic Number: The magic number of the archive. This is stored as a 4 byte string. It is always 'SCDL'.

- Version: The version of the SCDL schema. This is is stored as three 8-bit integers.

  - Major version
  - Minor version
  - Point version

- Endianness: The endianness of the archive. This is stored as a single integer based on an enum, but the value is always NETWORK (big endian).

- Backend: The backend of the archive. This is stored as a single integer based on an enum.

- Arrays: A list of arrays in the archive. This is stored as a list of arrays.

  - Name: The name of the array. This is stored as a string.
  - Length: The length of the array. This is stored as a single integer.
  - Dtype: The dtype of the array. This is stored as a string based on an enum.
  - \[Optional\] Shape: The shape of the array. This is stored as a list of integers.

#### Archive Header Spec:

The SCDL archive header uses network byte order (big-endian) throughout and consists of the following fixed-width fields:

**Core Header (Fixed Size: 16 bytes)**

```
Offset | Size (bytes) | Type    | Field       | Description
-------|------|---------|-------------|------------------------------------------
0x00   | 4    | char[4] | magic       | Magic number: 'SCDL' (0x5343444C)
0x04   | 1    | uint8   | version_maj | Major version number
0x05   | 1    | uint8   | version_min | Minor version number
0x06   | 1    | uint8   | version_pt  | Point version number
0x07   | 1    | uint8   | endianness  | Endianness enum (always 0x01 = NETWORK)
0x08   | 4    | uint32  | backend     | Backend type enum value
0x0C   | 4    | uint32  | array_count | Number of arrays in the archive
```

**Array Descriptors (Variable Size)**

Following the core header, each array is described by a variable-length descriptor:

```
Offset | Size (bytes)      | Type         | Field      | Description
-------|-----------|--------------|------------|----------------------------------
0x00   | 4         | uint32       | name_len   | Length of array filename in bytes
0x04   | name_len  | char[]       | name       | UTF-8 encoded array filename
var    | 8         | uint64       | length     | Number of elements in array
var+8  | 4         | uint32       | dtype      | ArrayDType enum value
var+12 | 1         | uint8        | has_shape  | Shape present flag (0x00 or 0x01)
var+13 | 4         | uint32       | shape_dims | Number of dimensions (if has_shape)
var+17 | shape_dims*4 | uint32[]  | shape      | Shape array (if has_shape)
```

**Data Layout Notes:**

- All multi-byte integers use network byte order (big-endian)
- Strings are UTF-8 encoded without null termination
- String lengths do not include null terminators
- Shape field is optional; when present, has_shape = 0x01
- Total header size = 16 + sum(array_descriptor_sizes)
- Array data follows immediately after all array descriptors

**Validation Rules:**

- Magic number must exactly match 'SCDL' (0x5343444C)
- Endianness field must be 0x01 (NETWORK byte order)
- All string lengths must be > 0
- Array count must match the number of array descriptors present
- When has_shape = 0x01, shape_dims must be > 0
- Array names must be unique within the archive
- Feature index names must be unique within the archive
- No name conflicts between arrays and feature indices
- All strings must be valid UTF-8
- Array lengths and shape dimensions must be non-negative
- Shape dimensions must be positive when specified

### FeatureIndex Header

Each FeatureIndex may optionally store a header, but it's nice if it does! This helps secure the archive and
make sure it is more robust to failures.

**FeatureIndex Binary Format (Extension after Array Descriptors):**

```
Offset | Size (bytes)     | Type         | Field           | Description
-------|-----------|--------------|-----------------|----------------------------------
0x00   | 4         | uint32       | fi_count        | Number of feature indices
```

For each feature index:

```
Offset | Size (bytes)      | Type         | Field           | Description
-------|-----------|--------------|-----------------|----------------------------------
0x00   | 4         | uint32       | name_len        | Length of feature index name
0x04   | name_len  | char[]       | name            | UTF-8 encoded feature index name
var    | 8         | uint64       | length          | Number of entries in index
var+8  | 4         | uint32       | dtype           | ArrayDType enum value
var+12 | 4         | uint32       | files_count     | Number of index files
var+16 | variable  | string[]     | index_files     | Array of file path strings
var    | 1         | uint8        | has_shape       | Shape present flag (0x00 or 0x01)
var+1  | 4         | uint32       | shape_dims      | Number of dimensions (if has_shape)
var+5  | shape_dims*4 | uint32[]  | shape           | Shape array (if has_shape)
```

**Backwards Compatibility:**
Feature indices are stored after array descriptors as an optional extension. Older implementations that don't support feature indices will simply ignore the additional data, maintaining compatibility.

### Backend Header

Each backend may optionally implement its own header. Currently, only the MEMMAP_V0 backend is supported with integer enum value 1.
