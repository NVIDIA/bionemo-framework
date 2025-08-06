# SCDL Header System: Complete Guide

This guide provides comprehensive documentation for working with SCDL (Single Cell Data Library) headers, including how to integrate arrays, feature indices, and metadata into your applications.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Header Components](#header-components)
4. [Working with Arrays](#working-with-arrays)
5. [Working with Feature Indices](#working-with-feature-indices)
6. [Header Management](#header-management)
7. [Schema Compliance](#schema-compliance)
8. [Best Practices](#best-practices)
9. [Advanced Usage](#advanced-usage)
10. [Error Handling](#error-handling)
11. [Examples](#examples)

## Overview

The SCDL header system provides a robust, cross-platform way to manage metadata for single-cell data archives. Headers store information about:

- **Arrays**: The actual data matrices (gene expression, cell metadata, etc.)
- **Feature Indices**: Fast lookup structures for genes, cells, or other features
- **Metadata**: Version, backend type, and structural information

Key features:
- **Binary format**: Non-human-readable for security and integrity
- **Cross-platform**: Network byte order ensures consistency across systems
- **Versioned**: Supports schema evolution and backwards compatibility
- **Validated**: Comprehensive validation prevents corruption

## Quick Start

### Creating a Basic Header

```python
from bionemo.scdl.schema.header import SCDLHeader, ArrayInfo, ArrayDType

# Create a new header
header = SCDLHeader()

# Add an array for gene expression data
expression_array = ArrayInfo(
    name="gene_expression.dat",
    length=50000,  # 50k cells
    dtype=ArrayDType.FLOAT32_ARRAY,
    shape=(50000, 25000)  # 50k cells × 25k genes
)
header.add_array(expression_array)

# Save to file
header.save("archive_header.bin")
```

### Loading an Existing Header

```python
from bionemo.scdl.schema.header import SCDLHeader

# Load header from file
header = SCDLHeader.load("archive_header.bin")

# Inspect the contents
print(f"Header contains {len(header.arrays)} arrays")
for array in header.arrays:
    print(f"  - {array.name}: {array.length} elements, dtype={array.dtype.name}")
```

## Header Components

### Core Header (Fixed 16 bytes)

The core header contains essential metadata:

```python
header = SCDLHeader()
print(f"Version: {header.version}")        # e.g., "0.0.2"
print(f"Backend: {header.backend}")        # e.g., "MEMMAP_V0"
print(f"Endianness: {header.endianness}")  # Always "NETWORK"
```

### Arrays

Arrays represent the actual data files in your archive:

```python
from bionemo.scdl.schema.header import ArrayInfo, ArrayDType

# Different array types
arrays = [
    ArrayInfo("expression.dat", 100000, ArrayDType.FLOAT32_ARRAY, (1000, 100)),
    ArrayInfo("cell_types.dat", 1000, ArrayDType.STRING_ARRAY),
    ArrayInfo("gene_ids.dat", 100, ArrayDType.FIXED_STRING_ARRAY),
    ArrayInfo("metadata.dat", 1000, ArrayDType.UINT32_ARRAY),
]

for array in arrays:
    header.add_array(array)
```

### Feature Indices

Feature indices provide fast lookups and metadata for specific features:

```python
from bionemo.scdl.schema.header import FeatureIndexInfo

# Create gene index
gene_index = FeatureIndexInfo(
    name="gene_index",
    length=25000,
    dtype=ArrayDType.STRING_ARRAY,
    index_files=["gene_symbols.idx", "gene_ensembl.idx"],
    shape=(25000,)
)
header.add_feature_index(gene_index)

# Create cell index
cell_index = FeatureIndexInfo(
    name="cell_index", 
    length=50000,
    dtype=ArrayDType.UINT64_ARRAY,
    index_files=["cell_barcodes.idx"]
)
header.add_feature_index(cell_index)
```

## Working with Arrays

### Array Data Types

Choose the appropriate data type for your arrays:

```python
from bionemo.scdl.schema.header import ArrayDType

# Numeric data types
ArrayDType.UINT8_ARRAY     # 0-255 integers (quality scores, flags)
ArrayDType.UINT16_ARRAY    # 0-65535 integers (small counts)
ArrayDType.UINT32_ARRAY    # 0-4B integers (large counts, IDs)
ArrayDType.UINT64_ARRAY    # 0-18E integers (very large IDs)
ArrayDType.FLOAT16_ARRAY   # Half precision (compressed data)
ArrayDType.FLOAT32_ARRAY   # Single precision (standard expression)
ArrayDType.FLOAT64_ARRAY   # Double precision (high accuracy)

# String data types
ArrayDType.STRING_ARRAY         # Variable-length strings
ArrayDType.FIXED_STRING_ARRAY   # Fixed-length strings
```

### Array Shapes

Arrays can be 1D (vectors) or multi-dimensional:

```python
# 1D array (gene list)
gene_names = ArrayInfo("genes.dat", 25000, ArrayDType.STRING_ARRAY, (25000,))

# 2D array (expression matrix: cells × genes)
expression = ArrayInfo("expr.dat", 1250000000, ArrayDType.FLOAT32_ARRAY, (50000, 25000))

# 3D array (time series: timepoints × cells × genes)
timeseries = ArrayInfo("time.dat", 750000000, ArrayDType.FLOAT32_ARRAY, (30, 50000, 500))

# No shape specified (1D assumed)
simple_array = ArrayInfo("simple.dat", 1000, ArrayDType.UINT32_ARRAY)
```

### Managing Arrays

```python
# Add arrays
header.add_array(expression_array)

# Find arrays
found_array = header.get_array("gene_expression.dat")
if found_array:
    print(f"Found array with {found_array.length} elements")

# Remove arrays
removed = header.remove_array("old_data.dat")
if removed:
    print("Successfully removed array")

# List all arrays
print("Arrays in header:")
for array in header.arrays:
    shape_str = f", shape={array.shape}" if array.shape else ""
    print(f"  {array.name}: {array.length} elements{shape_str}")
```

## Working with Feature Indices

Feature indices provide fast lookups and can reference multiple index files:

### Creating Feature Indices

```python
# Simple feature index
simple_index = FeatureIndexInfo(
    name="cell_types",
    length=50000,
    dtype=ArrayDType.STRING_ARRAY
)

# Complex feature index with multiple files
gene_index = FeatureIndexInfo(
    name="gene_annotations",
    length=25000,
    dtype=ArrayDType.STRING_ARRAY,
    index_files=[
        "gene_symbols.idx",    # Human-readable gene symbols
        "gene_ensembl.idx",    # Ensembl gene IDs
        "gene_entrez.idx",     # Entrez gene IDs
        "gene_descriptions.idx" # Gene descriptions
    ],
    shape=(25000, 4)  # 25k genes × 4 annotation types
)

# Spatial index for spatial transcriptomics
spatial_index = FeatureIndexInfo(
    name="spatial_coordinates",
    length=10000,
    dtype=ArrayDType.FLOAT32_ARRAY,
    index_files=["coordinates.idx"],
    shape=(10000, 2)  # X, Y coordinates
)
```

### Managing Feature Indices

```python
# Add feature indices
header.add_feature_index(gene_index)
header.add_feature_index(spatial_index)

# Find feature indices
gene_idx = header.get_feature_index("gene_annotations")
if gene_idx:
    print(f"Gene index has {len(gene_idx.index_files)} associated files")

# Remove feature indices
removed = header.remove_feature_index("old_index")

# List all feature indices
print("Feature indices:")
for fi in header.feature_indices:
    files_str = f" ({len(fi.index_files)} files)" if fi.index_files else ""
    print(f"  {fi.name}: {fi.length} entries{files_str}")
```

## Header Management

### Creating Headers

```python
from bionemo.scdl.schema.header import SCDLHeader, Backend
from bionemo.scdl.schema.version import SCDLVersion

# Default header (recommended)
header = SCDLHeader()

# Custom version
custom_version = SCDLVersion()
custom_version.major = 0
custom_version.minor = 1
custom_version.point = 0
header = SCDLHeader(version=custom_version)

# Custom backend (currently only MEMMAP_V0 available)
header = SCDLHeader(backend=Backend.MEMMAP_V0)
```

### Saving and Loading

```python
# Save to file
header.save("my_archive_header.bin")

# Load from file
try:
    loaded_header = SCDLHeader.load("my_archive_header.bin")
    print(f"Loaded header with {len(loaded_header.arrays)} arrays")
except HeaderSerializationError as e:
    print(f"Failed to load header: {e}")
```

### Serialization

```python
# Serialize to bytes
binary_data = header.serialize()
print(f"Header size: {len(binary_data)} bytes")

# Deserialize from bytes
restored_header = SCDLHeader.deserialize(binary_data)
```

### Validation

```python
try:
    header.validate()
    print("Header is valid")
except HeaderSerializationError as e:
    print(f"Header validation failed: {e}")
```

## Schema Compliance

### Required Validation Rules

The header system enforces several validation rules per the SCDL schema:

1. **Magic Number**: Must be exactly 'SCDL' (0x5343444C)
2. **Endianness**: Must be NETWORK byte order (big-endian)
3. **Unique Names**: Array names and feature index names must be unique
4. **No Conflicts**: No name conflicts between arrays and feature indices
5. **Valid UTF-8**: All strings must be valid UTF-8
6. **Positive Dimensions**: All shape dimensions must be positive when specified
7. **Non-negative Lengths**: Array lengths must be non-negative

### Version Compatibility

```python
from bionemo.scdl.schema.version import CurrentSCDLVersion

# Check version compatibility
current = CurrentSCDLVersion()
print(f"Current schema version: {current}")  # 0.0.2

# Headers with newer major versions are rejected
header.validate()  # Will raise error if major version > current
```

## Best Practices

### Naming Conventions

```python
# Use descriptive, hierarchical names
arrays = [
    ArrayInfo("raw/gene_expression.dat", ...),
    ArrayInfo("processed/normalized_expression.dat", ...),
    ArrayInfo("metadata/cell_annotations.dat", ...),
    ArrayInfo("metadata/gene_annotations.dat", ...),
]

# Use consistent extensions
feature_indices = [
    FeatureIndexInfo("gene_symbols", ..., index_files=["genes.idx"]),
    FeatureIndexInfo("cell_barcodes", ..., index_files=["cells.idx"]),
]
```

### Data Type Selection

```python
# Choose appropriate precision
expression_data = ArrayInfo(
    "expression.dat", 
    1000000,
    ArrayDType.FLOAT32_ARRAY,  # Usually sufficient for expression data
    (1000, 1000)
)

# Use smaller types when possible
cell_types = ArrayInfo(
    "cell_types.dat",
    1000,
    ArrayDType.UINT8_ARRAY,  # If you have < 256 cell types
    (1000,)
)

# Use appropriate string types
gene_symbols = ArrayInfo(
    "gene_symbols.dat",
    25000,
    ArrayDType.STRING_ARRAY,  # Variable length gene names
    (25000,)
)
```

### Memory Efficiency

```python
# Calculate header size before creating large archives
total_size = header.calculate_total_size()
print(f"Header will use {total_size} bytes")

# Use shapes to document array structure
expression = ArrayInfo(
    "expression.dat",
    cells * genes,
    ArrayDType.FLOAT32_ARRAY,
    (cells, genes)  # Documents the matrix structure
)
```

## Advanced Usage

### Header Merging

```python
from bionemo.scdl.schema.header import merge_headers, validate_header_compatibility

# Create compatible headers
header1 = SCDLHeader()
header1.add_array(ArrayInfo("batch1.dat", 1000, ArrayDType.FLOAT32_ARRAY))

header2 = SCDLHeader()
header2.add_array(ArrayInfo("batch2.dat", 1000, ArrayDType.FLOAT32_ARRAY))

# Check compatibility
if validate_header_compatibility(header1, header2):
    merged = merge_headers(header1, header2)
    print(f"Merged header has {len(merged.arrays)} arrays")
else:
    print("Headers are not compatible")
```

### Optimized Reading

```python
from bionemo.scdl.schema.header import HeaderReader

# For frequent access, use HeaderReader for efficiency
reader = HeaderReader("large_archive_header.bin")

# Quick validation without full deserialization
if reader.validate_magic():
    print(f"Valid SCDL archive")
    print(f"Version: {reader.get_version()}")
    print(f"Array count: {reader.get_array_count()}")
    
    # Full header only when needed
    if reader.get_array_count() > 0:
        full_header = reader.get_full_header()
```

### Creating from Files

```python
from bionemo.scdl.schema.header import create_header_from_arrays

# Quick header from existing files
array_files = ["data1.dat", "data2.dat", "data3.dat"]
header = create_header_from_arrays(array_files)

# Note: This creates placeholder entries; you should update them:
for array in header.arrays:
    # Update with actual file information
    array.length = get_actual_length(array.name)
    array.dtype = determine_dtype(array.name)
    array.shape = get_actual_shape(array.name)
```

### Inspection and Debugging

```python
# JSON representation for debugging
json_str = header.to_json()
print(json_str)

# YAML representation (requires PyYAML)
try:
    yaml_str = header.to_yaml()
    print(yaml_str)
except RuntimeError:
    print("PyYAML not available")

# String representation
print(header)  # SCDLHeader(version=0.0.2, backend=MEMMAP_V0, arrays=3, feature_indices=1)
```

## Error Handling

### Common Errors and Solutions

```python
from bionemo.scdl.schema.headerutil import HeaderSerializationError

try:
    header = SCDLHeader.load("archive_header.bin")
except HeaderSerializationError as e:
    if "Header file not found" in str(e):
        print("Archive header file is missing")
        # Create new header or handle missing file
    elif "Invalid magic number" in str(e):
        print("File is not a valid SCDL header")
        # File is corrupted or wrong format
    elif "Unsupported version" in str(e):
        print("Header version is too new for this library")
        # Upgrade library or convert header
    else:
        print(f"Unexpected error: {e}")

# Validation errors
try:
    header.validate()
except HeaderSerializationError as e:
    if "Duplicate array names" in str(e):
        print("Fix duplicate array names")
    elif "Name conflicts" in str(e):
        print("Arrays and feature indices have conflicting names")
    elif "Empty array name" in str(e):
        print("All arrays must have non-empty names")
```

### Robust Header Creation

```python
def create_robust_header(arrays_data, feature_indices_data=None):
    """Create a header with comprehensive error handling."""
    header = SCDLHeader()
    
    # Add arrays with validation
    for array_data in arrays_data:
        try:
            array = ArrayInfo(**array_data)
            array._validate()  # Pre-validate
            header.add_array(array)
        except HeaderSerializationError as e:
            print(f"Skipping invalid array {array_data.get('name', 'unknown')}: {e}")
    
    # Add feature indices
    if feature_indices_data:
        for fi_data in feature_indices_data:
            try:
                fi = FeatureIndexInfo(**fi_data)
                fi._validate()  # Pre-validate
                header.add_feature_index(fi)
            except HeaderSerializationError as e:
                print(f"Skipping invalid feature index {fi_data.get('name', 'unknown')}: {e}")
    
    # Final validation
    try:
        header.validate()
        return header
    except HeaderSerializationError as e:
        print(f"Header validation failed: {e}")
        return None
```

## Examples

### Single-Cell RNA-seq Archive

```python
from bionemo.scdl.schema.header import SCDLHeader, ArrayInfo, FeatureIndexInfo, ArrayDType

# Create header for scRNA-seq data
header = SCDLHeader()

# Expression matrix (cells × genes)
expression = ArrayInfo(
    name="expression_matrix.dat",
    length=1250000000,  # 50k cells × 25k genes
    dtype=ArrayDType.FLOAT32_ARRAY,
    shape=(50000, 25000)
)
header.add_array(expression)

# Cell metadata
cell_metadata = ArrayInfo(
    name="cell_metadata.dat",
    length=50000,
    dtype=ArrayDType.STRING_ARRAY,  # JSON strings with metadata
    shape=(50000,)
)
header.add_array(cell_metadata)

# Gene information
gene_info = ArrayInfo(
    name="gene_info.dat", 
    length=25000,
    dtype=ArrayDType.STRING_ARRAY,
    shape=(25000,)
)
header.add_array(gene_info)

# Gene index for fast lookups
gene_index = FeatureIndexInfo(
    name="gene_index",
    length=25000,
    dtype=ArrayDType.STRING_ARRAY,
    index_files=["gene_symbols.idx", "gene_ensembl.idx"],
    shape=(25000, 2)
)
header.add_feature_index(gene_index)

# Cell barcode index
cell_index = FeatureIndexInfo(
    name="cell_barcode_index",
    length=50000,
    dtype=ArrayDType.STRING_ARRAY,
    index_files=["cell_barcodes.idx"]
)
header.add_feature_index(cell_index)

# Save the complete header
header.save("scrna_archive_header.bin")
print(f"Created scRNA-seq header with {len(header.arrays)} arrays and {len(header.feature_indices)} indices")
```

### Spatial Transcriptomics Archive

```python
# Spatial transcriptomics with coordinate information
header = SCDLHeader()

# Expression data
expression = ArrayInfo(
    name="spatial_expression.dat",
    length=500000000,  # 10k spots × 20k genes  
    dtype=ArrayDType.FLOAT32_ARRAY,
    shape=(10000, 20000)
)
header.add_array(expression)

# Spatial coordinates
coordinates = ArrayInfo(
    name="spot_coordinates.dat",
    length=20000,  # 10k spots × 2 coordinates
    dtype=ArrayDType.FLOAT32_ARRAY,
    shape=(10000, 2)
)
header.add_array(coordinates)

# Tissue image coordinates
image_coords = ArrayInfo(
    name="image_coordinates.dat",
    length=20000,
    dtype=ArrayDType.UINT32_ARRAY,
    shape=(10000, 2)  # Pixel coordinates
)
header.add_array(image_coords)

# Spatial index
spatial_index = FeatureIndexInfo(
    name="spatial_index",
    length=10000,
    dtype=ArrayDType.FLOAT32_ARRAY,
    index_files=["spatial_tree.idx"],  # Spatial tree for neighbor queries
    shape=(10000, 2)
)
header.add_feature_index(spatial_index)

header.save("spatial_archive_header.bin")
```

### Multi-Modal Archive

```python
# Multi-modal data (RNA + ATAC + Protein)
header = SCDLHeader()

# RNA expression
rna_expr = ArrayInfo(
    name="rna_expression.dat",
    length=625000000,  # 25k cells × 25k genes
    dtype=ArrayDType.FLOAT32_ARRAY, 
    shape=(25000, 25000)
)
header.add_array(rna_expr)

# ATAC peaks
atac_peaks = ArrayInfo(
    name="atac_peaks.dat",
    length=1250000000,  # 25k cells × 50k peaks
    dtype=ArrayDType.FLOAT32_ARRAY,
    shape=(25000, 50000)
)
header.add_array(atac_peaks)

# Protein expression
protein_expr = ArrayInfo(
    name="protein_expression.dat", 
    length=2500000,  # 25k cells × 100 proteins
    dtype=ArrayDType.FLOAT32_ARRAY,
    shape=(25000, 100)
)
header.add_array(protein_expr)

# Shared cell index
cell_index = FeatureIndexInfo(
    name="cell_index",
    length=25000,
    dtype=ArrayDType.STRING_ARRAY,
    index_files=["cell_barcodes.idx"]
)
header.add_feature_index(cell_index)

# Modality-specific indices
gene_index = FeatureIndexInfo(
    name="gene_index", 
    length=25000,
    dtype=ArrayDType.STRING_ARRAY,
    index_files=["gene_symbols.idx"]
)
header.add_feature_index(gene_index)

peak_index = FeatureIndexInfo(
    name="peak_index",
    length=50000, 
    dtype=ArrayDType.STRING_ARRAY,
    index_files=["peak_coordinates.idx"]
)
header.add_feature_index(peak_index)

protein_index = FeatureIndexInfo(
    name="protein_index",
    length=100,
    dtype=ArrayDType.STRING_ARRAY,
    index_files=["protein_names.idx"]
)
header.add_feature_index(protein_index)

header.save("multimodal_archive_header.bin")
```

---

This guide provides comprehensive coverage of the SCDL header system. For additional questions or advanced use cases, refer to the source code documentation or the SCDL schema specification.